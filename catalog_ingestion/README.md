# AI Academic Advisor — Data Stack Runbook

Everything you need to **run the database (and backend API) yourself**. The stack runs
under rootless **podman** — no `sudo`, no host installs. All commands are run from this
directory (`catalog_ingestion/`) via `make`.

```
┌─────────────────────┐     ┌──────────────────────┐     ┌───────────────────────────┐
│ Postgres            │ <── │ Backend (FastAPI)    │ <── │ Web client (clients/web)  │
│ catalog_ingestion   │     │ http://localhost:8000│     │ (run separately)          │
│ host port 5433      │     └──────────────────────┘     └───────────────────────────┘
└─────────────────────┘            ▲
        ▲                          │
        │ ingestion CLI (one-shot) │
        └──────────────────────────┘
   PurdueIO API → courses   |   catalog scrape → degree requirements
```

## Prerequisites (already set up on this server)
- Rootless podman with the user API socket enabled (one-time, already done):
  ```
  systemctl --user enable --now podman.socket
  ```
- `make` and `podman`. That's it.

## Quick start
```bash
cd catalog_ingestion
make up        # start Postgres + backend  → http://localhost:8000
make counts    # see how much data is loaded
make api       # print the API URLs
```
Open the interactive API docs at **http://localhost:8000/docs**.

The database already contains data (it persists in a podman volume across restarts):

| table | rows | source |
|-------|------|--------|
| catalog_years | 13 | catalog index |
| courses | ~10,600 | **PurdueIO API** |
| subjects | ~208 | PurdueIO API |
| programs | 1 | catalog scrape |
| requirement_groups / requirement_options | 41 / 96 | catalog scrape |

## Inspect the data
```bash
make psql      # open a SQL shell:  SELECT * FROM programs;
make counts    # row counts for the main tables

# Or hit the API:
curl 'http://localhost:8000/academic/facets'
curl 'http://localhost:8000/academic/programs'
curl 'http://localhost:8000/academic/courses/search?subject=CS'
```
Connect an external GUI (DBeaver / pgAdmin / TablePlus) with:
```
host=localhost  port=5433  db=catalog_ingestion  user=catalog  password=catalog
```

## Load / refresh data
Courses come from the **PurdueIO API** (fast, no scraping). Degree **requirements** come
from scraping the Purdue catalog (that is the only place they exist).
```bash
make load-years                      # populate catalog_years (13 years)
make load-courses                    # ~10k courses from PurdueIO for 2026-2027
make load-courses YEAR=2025-2026     # a different year
make sync-program POID=34693         # scrape ONE program
make sync-programs                   # scrape ALL ~959 programs (long; see note)
make validate                        # data-quality report
```
**Crawl delay:** `sync-programs` honors Purdue's `robots.txt` (120 s/page by default).
For a faster full run at your own discretion, lower it per-run — single-threaded, off-peak:
```bash
podman compose run --rm -e CRAWL_DELAY_SECONDS=20 ingestion sync-programs --year 2026-2027
```

## Lifecycle
```bash
make up        # start
make stop      # stop (keeps containers + data)
make down      # remove containers (DATA PERSISTS in the volume)
make status    # what's running
make logs      # follow backend logs
make backup    # dump DB → catalog_db_backup.sql.gz
make restore   # restore DB from catalog_db_backup.sql.gz
make reset     # ⚠ DESTROY all data + volumes, recreate empty tables
```
`make help` lists everything.

> **Note — harmless podman warning.** On this host, tearing down the last container's
> network prints `rootless netns: kill network process: permission denied`. The container
> still stops; `make stop/down` use direct `podman` calls that complete and exit cleanly,
> so you can ignore it. (It does not affect data or `make up`.)
