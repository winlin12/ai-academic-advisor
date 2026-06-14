from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from requirementsync.catalog_scraper import (
    DEFAULT_BASE_URL,
    CatalogFetcher,
    discover_catalog_entrypoint,
    discover_program_urls,
    parse_program_page,
    parsed_programs_to_json,
)


DEFAULT_DATABASE_URL = "postgresql://purdueio:purdueio@postgres:5432/purdueio"


def env_value(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except ValueError:
        return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync Purdue degree requirements into the advisor Postgres schema."
    )
    parser.add_argument(
        "--database-url",
        default=env_value("REQUIREMENTS_DB_CONNECTION", DEFAULT_DATABASE_URL),
        help="Postgres connection URL.",
    )
    parser.add_argument(
        "--catalog-base-url",
        default=env_value("CATALOG_BASE_URL", DEFAULT_BASE_URL),
        help="Official Purdue catalog entrypoint URL.",
    )
    parser.add_argument(
        "--catalog-year",
        default=env_value("CATALOG_YEAR", "current"),
        help="Catalog start year, or 'current'.",
    )
    parser.add_argument(
        "--max-crawl-pages",
        type=int,
        default=env_int("MAX_CRAWL_PAGES", 150),
        help="Maximum catalog pages to crawl while discovering programs.",
    )
    parser.add_argument(
        "--max-programs",
        type=int,
        default=env_int("MAX_PROGRAMS", 0),
        help="Maximum program pages to load; 0 means no limit.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=env_int("CATALOG_TIMEOUT_SECONDS", 60),
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scrape and print parsed programs without writing to Postgres.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fetcher = CatalogFetcher(timeout=args.timeout)
    errors: list[dict[str, Any]] = []

    try:
        catalog_url, catalog_year, catalog_label = discover_catalog_entrypoint(
            fetcher,
            base_url=args.catalog_base_url,
            catalog_year=args.catalog_year,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[error] failed to discover catalog entrypoint: {exc}", file=sys.stderr)
        return 1

    if catalog_year <= 0:
        print(
            "[error] could not determine catalog year; pass --catalog-year YYYY",
            file=sys.stderr,
        )
        return 1

    print(f"Catalog: {catalog_label} ({catalog_year})")
    print(f"Entrypoint: {catalog_url}")

    try:
        program_urls = discover_program_urls(
            fetcher,
            catalog_url=catalog_url,
            max_pages=args.max_crawl_pages,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[error] failed to discover program URLs: {exc}", file=sys.stderr)
        return 1

    if args.max_programs > 0:
        program_urls = program_urls[: args.max_programs]

    print(f"Program URLs discovered: {len(program_urls)}")

    programs = []
    for idx, url in enumerate(program_urls, start=1):
        try:
            page = fetcher.fetch(url)
            program = parse_program_page(page, catalog_year=catalog_year)
            if program is None:
                continue
            programs.append(program)
            print(f"[{idx}/{len(program_urls)}] parsed {program.program_title}")
        except Exception as exc:  # noqa: BLE001
            errors.append({"url": url, "error": str(exc)})
            print(f"[warn] failed program page {url}: {exc}", file=sys.stderr)

    if args.dry_run:
        print(parsed_programs_to_json(programs))
        return 0 if programs else 2

    from requirementsync.db import (  # noqa: PLC0415
        apply_migrations,
        build_course_map,
        connect,
        finish_sync_run,
        json_dumps,
        replace_catalog_year,
        start_sync_run,
    )

    conn = connect(args.database_url)
    run_id = None
    try:
        apply_migrations(conn)
        run_id = start_sync_run(
            conn,
            catalog_year=catalog_year,
            catalog_label=catalog_label,
            catalog_base_url=args.catalog_base_url,
            metadata={
                "catalog_url": catalog_url,
                "program_urls_discovered": len(program_urls),
                "max_crawl_pages": args.max_crawl_pages,
                "max_programs": args.max_programs,
            },
        )
        course_map = build_course_map(conn)
        print(f"PurdueIO course links available: {len(course_map)}")

        loaded = replace_catalog_year(
            conn,
            catalog_year=catalog_year,
            run_id=run_id,
            programs=programs,
            course_map=course_map,
        )
        status = "completed" if loaded else "failed"
        finish_sync_run(
            conn,
            run_id=run_id,
            status=status,
            programs_found=len(program_urls),
            programs_loaded=loaded,
            errors=errors,
        )
        print("Requirement sync complete.")
        print(
            json_dumps(
                {
                    "run_id": run_id,
                    "catalog_year": catalog_year,
                    "programs_found": len(program_urls),
                    "programs_loaded": loaded,
                    "errors": len(errors),
                }
            )
        )
        return 0 if loaded else 2
    except Exception as exc:  # noqa: BLE001
        if run_id is not None:
            finish_sync_run(
                conn,
                run_id=run_id,
                status="failed",
                programs_found=len(program_urls),
                programs_loaded=len(programs),
                errors=[*errors, {"error": str(exc)}],
            )
        print(f"[error] requirement sync failed: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
