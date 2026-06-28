"""Command-line interface for the Purdue catalog ingestion pipeline."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from catalog_ingestion.config import get_settings

app = typer.Typer(
    name="catalog-ingest",
    help="Purdue catalog ingestion pipeline.",
    add_completion=False,
)
console = Console()


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


@app.command("db-init")
def db_init():
    """Create all database tables (idempotent)."""
    _setup_logging()
    from catalog_ingestion.db.session import check_connection, create_all_tables

    settings = get_settings()
    rprint(f"[bold]Connecting to:[/bold] {settings.database_url}")

    if not check_connection():
        rprint("[red]ERROR: Cannot connect to database.[/red]")
        raise typer.Exit(1)

    create_all_tables()
    rprint("[green]Database tables created (or already exist).[/green]")


@app.command("discover-years")
def discover_years(
    dry_run: bool = typer.Option(False, "--dry-run", help="Print results without writing to DB"),
):
    """Discover catalog years from the catalog index page."""
    _setup_logging()
    settings = get_settings()

    from catalog_ingestion.fetch.client import HttpxFetcher, WafChallengeError
    from catalog_ingestion.fetch.cache import DiskCache
    from catalog_ingestion.discover.catalog_years import discover_catalog_years

    cache = DiskCache(settings.page_cache_dir / "index")
    fetcher = HttpxFetcher(
        user_agent=settings.catalog_user_agent,
        base_url=settings.catalog_base_url,
        crawl_delay=1.0,
        cache=cache,
    )

    rprint(f"Fetching catalog index: {settings.catalog_base_url}/index.php")
    try:
        page = fetcher.fetch(f"{settings.catalog_base_url}/index.php")
    except WafChallengeError as exc:
        rprint(f"[red]WAF challenge on index page: {exc}[/red]")
        raise typer.Exit(1)

    years = discover_catalog_years(page, settings.catalog_base_url)

    table = Table(title="Catalog Years")
    table.add_column("Label")
    table.add_column("catoid")
    table.add_column("Archived?")
    table.add_column("URL")
    for year in years:
        table.add_row(
            year.label,
            str(year.catoid),
            "yes" if year.is_archived else "no",
            year.catalog_url,
        )
    console.print(table)

    if dry_run:
        rprint("[yellow]Dry run — not saving to database.[/yellow]")
        return

    from catalog_ingestion.db.session import get_session
    from catalog_ingestion.db.models import CatalogYear

    with get_session() as session:
        for info in years:
            existing = session.query(CatalogYear).filter_by(catoid=info.catoid).first()
            if existing:
                existing.label = info.label
                existing.start_year = info.start_year
                existing.end_year = info.end_year
                existing.is_archived = info.is_archived
                existing.catalog_url = info.catalog_url
            else:
                session.add(
                    CatalogYear(
                        label=info.label,
                        catoid=info.catoid,
                        start_year=info.start_year,
                        end_year=info.end_year,
                        is_archived=info.is_archived,
                        catalog_url=info.catalog_url,
                    )
                )

    rprint(f"[green]Saved {len(years)} catalog years to database.[/green]")


@app.command("sync-courses")
def sync_courses(
    year: str = typer.Option(..., "--year", help="Catalog year label, e.g. '2026-2027'"),
    subject: Optional[str] = typer.Option(None, "--subject", help="Subject code, e.g. 'CS'"),
    all_subjects: bool = typer.Option(False, "--all-subjects"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    max_courses: int = typer.Option(0, "--max-courses", help="Limit (0 = unlimited)"),
):
    """Scrape and ingest courses for a catalog year."""
    _setup_logging(get_settings().log_level)
    settings = get_settings()

    from catalog_ingestion.db.session import get_session
    from catalog_ingestion.db.models import CatalogYear
    from catalog_ingestion.fetch.client import build_fetcher, WafChallengeError
    from catalog_ingestion.discover.subjects import (
        KNOWN_COURSES_NAVOIDS,
        build_courses_page_url,
        build_subject_filter_url,
        discover_prefix_filter_param,
        discover_subjects_from_filter_page,
    )
    from catalog_ingestion.parse.courses import (
        parse_course_page,
        extract_course_links_from_listing,
    )
    from catalog_ingestion.ingest.courses import ingest_course, upsert_source_page

    PAGE_SIZE = 100
    MAX_LISTING_PAGES = 60  # safety cap; ~6000 courses per subject

    with get_session() as session:
        db_year = session.query(CatalogYear).filter_by(label=year).first()
        if not db_year:
            rprint(f"[red]Catalog year {year!r} not found. Run 'discover-years' first.[/red]")
            raise typer.Exit(1)
        catoid = db_year.catoid
        year_id = db_year.id

    navoid = KNOWN_COURSES_NAVOIDS.get(catoid)
    if not navoid:
        rprint(f"[red]No courses navoid known for catoid={catoid}. Add it to discover/subjects.py[/red]")
        raise typer.Exit(1)

    # The fetcher always fetches (read-only); --dry-run only suppresses DB writes.
    fetcher = build_fetcher(
        backend=settings.fetcher_backend,
        user_agent=settings.catalog_user_agent,
        base_url=settings.catalog_base_url,
        crawl_delay=settings.crawl_delay_seconds,
        cache_dir=settings.page_cache_dir,
        dry_run=False,
        browser_executable=settings.playwright_browser,
    )

    def _collect_course_links(fetcher_obj, subj_code: str, prefix_param: str):
        """Paginate a subject's course listing, returning unique (url, coid) pairs."""
        collected: list[tuple[str, int]] = []
        seen: set[int] = set()
        for cpage in range(1, MAX_LISTING_PAGES + 1):
            url = build_subject_filter_url(
                settings.catalog_base_url, catoid, navoid, subj_code,
                cpage=cpage, prefix_param=prefix_param,
            )
            listing_page = fetcher_obj.fetch(url)
            page_links = extract_course_links_from_listing(
                listing_page.html, settings.catalog_base_url, catoid
            )
            new = [(u, c) for (u, c) in page_links if c not in seen]
            for _, c in new:
                seen.add(c)
            collected.extend(new)
            # Last page (short) or nothing new → stop. Also stop early if we already
            # have enough to satisfy a --max-courses cap.
            if len(page_links) < PAGE_SIZE or not new:
                break
            if max_courses > 0 and len(collected) >= max_courses:
                break
        return collected

    def run_sync(fetcher_obj):
        # One fetch of the courses page to discover the prefix filter param (and, for
        # --all-subjects, the subject list).
        courses_page = fetcher_obj.fetch(
            build_courses_page_url(settings.catalog_base_url, catoid, navoid)
        )
        prefix_param = discover_prefix_filter_param(courses_page.html)
        rprint(f"Course-prefix filter param: [cyan]{prefix_param}[/cyan]")

        if subject:
            subjects_to_scrape = [subject.upper()]
        elif all_subjects:
            subject_infos = discover_subjects_from_filter_page(courses_page.html)
            subjects_to_scrape = [s.code for s in subject_infos]
            rprint(f"Found {len(subjects_to_scrape)} subjects")
        else:
            rprint("[red]Specify --subject SUBJ or --all-subjects[/red]")
            raise typer.Exit(1)

        total_ok = 0
        total_err = 0

        for subj_code in subjects_to_scrape:
            try:
                course_links = _collect_course_links(fetcher_obj, subj_code, prefix_param)
            except Exception as exc:
                rprint(f"  [red]Failed to list courses for {subj_code}: {exc}[/red]")
                total_err += 1
                continue

            if max_courses > 0:
                course_links = course_links[:max_courses]
            rprint(f"  Subject {subj_code}: {len(course_links)} courses")

            for course_url, coid in course_links:
                try:
                    course_page = fetcher_obj.fetch(course_url)
                    parsed = parse_course_page(course_page.html, course_url)
                    if parsed is None:
                        rprint(f"    [yellow]Could not parse {course_url}[/yellow]")
                        total_err += 1
                        continue

                    if not dry_run:
                        with get_session() as session:
                            db_year_obj = session.query(CatalogYear).filter_by(
                                label=year
                            ).first()
                            sp = upsert_source_page(
                                session,
                                url=course_url,
                                html=course_page.html,
                                http_status=course_page.http_status,
                                catalog_year_id=db_year_obj.id,
                                page_type="course",
                            )
                            ingest_course(
                                session,
                                parsed=parsed,
                                catalog_year_id=db_year_obj.id,
                                source_page_id=sp.id,
                            )
                    rprint(f"    [green]OK[/green] {parsed.course_code}: {parsed.title}")
                    total_ok += 1
                except Exception as exc:
                    rprint(f"    [red]ERR {course_url}: {exc}[/red]")
                    total_err += 1

        rprint(f"\nDone: {total_ok} ok, {total_err} errors")

    if isinstance(fetcher, __import__("catalog_ingestion.fetch.client", fromlist=["PlaywrightFetcher"]).PlaywrightFetcher):
        with fetcher:
            run_sync(fetcher)
    else:
        run_sync(fetcher)


@app.command("sync-program")
def sync_program(
    year: str = typer.Option(..., "--year"),
    program_url: Optional[str] = typer.Option(None, "--url", help="Direct preview_program.php URL"),
    poid: Optional[int] = typer.Option(None, "--poid", help="Program poid"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Fetch and ingest one program page."""
    _setup_logging(get_settings().log_level)
    settings = get_settings()

    from catalog_ingestion.db.session import get_session
    from catalog_ingestion.db.models import CatalogYear
    from catalog_ingestion.fetch.client import build_fetcher
    from catalog_ingestion.parse.programs import parse_program_page
    from catalog_ingestion.parse.requirements import parse_requirement_sections
    from catalog_ingestion.ingest.programs import ingest_program

    with get_session() as session:
        db_year = session.query(CatalogYear).filter_by(label=year).first()
        if not db_year:
            rprint(f"[red]Catalog year {year!r} not in database. Run 'discover-years' first.[/red]")
            raise typer.Exit(1)
        catoid = db_year.catoid

    if not program_url and poid:
        program_url = (
            f"{settings.catalog_base_url}/preview_program.php?catoid={catoid}&poid={poid}"
        )

    if not program_url:
        rprint("[red]Provide --url or --poid[/red]")
        raise typer.Exit(1)

    fetcher = build_fetcher(
        backend=settings.fetcher_backend,
        user_agent=settings.catalog_user_agent,
        base_url=settings.catalog_base_url,
        crawl_delay=settings.crawl_delay_seconds,
        cache_dir=settings.page_cache_dir,
        dry_run=dry_run,
        browser_executable=settings.playwright_browser,
    )

    def run():
        page = fetcher.fetch(program_url)
        parsed_prog = parse_program_page(page.html, program_url)
        if not parsed_prog:
            rprint(f"[red]Could not parse program at {program_url}[/red]")
            raise typer.Exit(1)

        req_groups = parse_requirement_sections(page.html)
        rprint(f"Program: [bold]{parsed_prog.name}[/bold] ({parsed_prog.degree_type})")
        rprint(f"College: {parsed_prog.college_name}")
        rprint(f"Total credits: {parsed_prog.total_credits_min} ({parsed_prog.total_credits_raw})")
        rprint(f"Requirement groups: {len(req_groups)}")
        for g in req_groups:
            rprint(f"  [{g.requirement_type or '?'}] {g.name} — {len(g.options)} options")

        if not dry_run:
            with get_session() as session:
                db_year_obj = session.query(CatalogYear).filter_by(label=year).first()
                ingest_program(
                    session,
                    parsed=parsed_prog,
                    requirement_groups=req_groups,
                    catalog_year=db_year_obj,
                    html=page.html,
                )
            rprint("[green]Program saved to database.[/green]")

    PlaywrightFetcher = __import__(
        "catalog_ingestion.fetch.client", fromlist=["PlaywrightFetcher"]
    ).PlaywrightFetcher
    if isinstance(fetcher, PlaywrightFetcher):
        with fetcher:
            run()
    else:
        run()


@app.command("sync-programs")
def sync_programs(
    year: str = typer.Option(..., "--year"),
    limit: int = typer.Option(0, "--limit", help="Max programs (0 = all)"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Discover and ingest all programs for a catalog year."""
    _setup_logging(get_settings().log_level)
    settings = get_settings()

    from catalog_ingestion.db.session import get_session
    from catalog_ingestion.db.models import CatalogYear
    from catalog_ingestion.fetch.client import build_fetcher
    from catalog_ingestion.discover.programs import (
        build_programs_list_url,
        discover_program_links,
    )
    from catalog_ingestion.parse.programs import parse_program_page
    from catalog_ingestion.parse.requirements import parse_requirement_sections
    from catalog_ingestion.ingest.programs import ingest_program

    with get_session() as session:
        db_year = session.query(CatalogYear).filter_by(label=year).first()
        if not db_year:
            rprint(f"[red]Catalog year {year!r} not in database.[/red]")
            raise typer.Exit(1)
        catoid = db_year.catoid

    programs_list_url = build_programs_list_url(settings.catalog_base_url, catoid)
    if not programs_list_url:
        rprint(f"[red]No programs list URL known for catoid={catoid}[/red]")
        raise typer.Exit(1)

    fetcher = build_fetcher(
        backend=settings.fetcher_backend,
        user_agent=settings.catalog_user_agent,
        base_url=settings.catalog_base_url,
        crawl_delay=settings.crawl_delay_seconds,
        cache_dir=settings.page_cache_dir,
        dry_run=dry_run,
        browser_executable=settings.playwright_browser,
    )

    def run():
        rprint(f"Fetching programs list: {programs_list_url}")
        list_page = fetcher.fetch(programs_list_url)
        links = discover_program_links(list_page.html, settings.catalog_base_url, catoid)

        if limit > 0:
            links = links[:limit]

        rprint(f"Found {len(links)} program links")
        ok = err = 0
        for link in links:
            try:
                page = fetcher.fetch(link.url)
                parsed = parse_program_page(page.html, link.url)
                if not parsed:
                    rprint(f"  [yellow]No parse: {link.url}[/yellow]")
                    continue
                groups = parse_requirement_sections(page.html)
                if not dry_run:
                    with get_session() as session:
                        db_year_obj = session.query(CatalogYear).filter_by(label=year).first()
                        ingest_program(session, parsed=parsed, requirement_groups=groups,
                                       catalog_year=db_year_obj, html=page.html)
                rprint(f"  [green]OK[/green] {parsed.name} ({parsed.degree_type})")
                ok += 1
            except Exception as exc:
                rprint(f"  [red]ERR {link.url}: {exc}[/red]")
                err += 1
        rprint(f"\nDone: {ok} ok, {err} errors")

    PlaywrightFetcher = __import__(
        "catalog_ingestion.fetch.client", fromlist=["PlaywrightFetcher"]
    ).PlaywrightFetcher
    if isinstance(fetcher, PlaywrightFetcher):
        with fetcher:
            run()
    else:
        run()


@app.command("validate")
def validate_cmd(
    year: str = typer.Option(..., "--year"),
    output: Optional[str] = typer.Option(None, "--output", help="Output file path for report"),
):
    """Run validation checks for a catalog year."""
    _setup_logging(get_settings().log_level)

    from catalog_ingestion.db.session import get_session
    from catalog_ingestion.audit.validation import validate_catalog_year
    from catalog_ingestion.audit.reports import write_validation_report

    with get_session() as session:
        results = validate_catalog_year(session, catalog_year_label=year)

    table = Table(title=f"Validation — {year}")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Count")
    table.add_column("Notes")

    for r in results:
        status = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
        table.add_row(r.check_name, status, str(r.count), r.notes)

    console.print(table)

    if output:
        write_validation_report(results, Path(output))
        rprint(f"Report written to {output}")


@app.command("import-purdueapi")
def import_purdueapi(
    entity: str = typer.Option("all", "--entity", help="subjects|courses|terms|all"),
    subject: Optional[str] = typer.Option(None, "--subject"),
):
    """Import data from PurdueAPI OData endpoint into staging tables."""
    _setup_logging(get_settings().log_level)
    settings = get_settings()

    from catalog_ingestion.db.session import get_session
    from catalog_ingestion.ingest.purdueapi_import import (
        import_courses,
        import_subjects,
        import_terms,
    )

    subject_codes = [subject.upper()] if subject else None

    with get_session() as session:
        if entity in ("subjects", "all"):
            n = import_subjects(session, odata_url=settings.purdueapi_odata_url)
            rprint(f"[green]Imported {n} subjects[/green]")
        if entity in ("courses", "all"):
            n = import_courses(
                session,
                odata_url=settings.purdueapi_odata_url,
                subject_codes=subject_codes,
            )
            rprint(f"[green]Imported {n} courses[/green]")
        if entity in ("terms", "all"):
            n = import_terms(session, odata_url=settings.purdueapi_odata_url)
            rprint(f"[green]Imported {n} terms[/green]")


@app.command("report-summary")
def report_summary(year: str = typer.Option(..., "--year")):
    """Print an ingestion summary for a catalog year."""
    _setup_logging(get_settings().log_level)

    from catalog_ingestion.db.session import get_session
    from catalog_ingestion.audit.reports import report_catalog_summary

    with get_session() as session:
        summary = report_catalog_summary(session, catalog_year_label=year)

    rprint(json.dumps(summary, indent=2))


@app.command("fetch-url")
def fetch_url_cmd(
    url: str = typer.Argument(...),
    output: Optional[str] = typer.Option(None, "--output", "-o"),
):
    """Fetch a single URL and print or save the HTML (for inspection)."""
    _setup_logging(get_settings().log_level)
    settings = get_settings()

    from catalog_ingestion.fetch.client import build_fetcher

    fetcher = build_fetcher(
        backend=settings.fetcher_backend,
        user_agent=settings.catalog_user_agent,
        base_url=settings.catalog_base_url,
        crawl_delay=1.0,
        cache_dir=settings.page_cache_dir,
        dry_run=False,
        browser_executable=settings.playwright_browser,
    )

    PlaywrightFetcher = __import__(
        "catalog_ingestion.fetch.client", fromlist=["PlaywrightFetcher"]
    ).PlaywrightFetcher

    def run():
        page = fetcher.fetch(url)
        rprint(f"Status: {page.http_status}  Bytes: {len(page.html)}  Cache: {page.from_cache}")
        if output:
            Path(output).write_text(page.html)
            rprint(f"Saved to {output}")
        else:
            print(page.html[:2000])

    if isinstance(fetcher, PlaywrightFetcher):
        with fetcher:
            run()
    else:
        run()


if __name__ == "__main__":
    app()
