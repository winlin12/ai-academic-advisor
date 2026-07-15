# Runbook for the BoilerAdvisor data stack (rootless podman).
# Run all targets from this directory:  make <target>
#
#   make up            # start the database + backend API
#   make psql          # open a SQL shell into the database
#   make api           # print the API docs URL
#   make help          # list everything
#
# Override the catalog year on any data target:  make load-courses YEAR=2025-2026

# Container engine: rootless podman where available (the original target host), else docker
# (e.g. macOS with Docker Desktop). All targets work identically under both.
ENGINE     := $(shell command -v podman >/dev/null 2>&1 && echo podman || echo docker)
COMPOSE    := $(ENGINE) compose
PG         := catalog-ingestion-postgres
CONTAINERS := catalog-ingestion-backend catalog-ingestion-postgres catalog-ingestion-adminer
VOLUMES    := catalog_ingestion_catalog-pg-data catalog_ingestion_catalog-page-cache
YEAR       ?= 2026-2027

# Note: this host's rootless podman prints a harmless "rootless netns: kill network
# process: permission denied" when the LAST container's network is torn down, and
# `podman compose down/stop` treats that as fatal. Direct `podman stop`/`rm` complete
# the operation and exit 0, so stop/down/reset use those instead.

.DEFAULT_GOAL := help

.PHONY: help up down stop status logs psql api adminer db-init load-years load-courses \
        sync-program sync-programs validate counts backup restore reset

help: ## List available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-15s\033[0m %s\n",$$1,$$2}'

## --- run the stack ---------------------------------------------------------
up: ## Start the database + backend API (http://localhost:8000)
	$(COMPOSE) up -d postgres backend
	@echo "API:  http://localhost:8000   (docs: http://localhost:8000/docs)"
	@echo "DB:   postgresql://catalog:catalog@localhost:5433/catalog_ingestion"

stop: ## Stop the containers (keeps them and the data)
	@$(ENGINE) stop -t 5 $(CONTAINERS) 2>/dev/null || true
	@echo "Stopped (data preserved). Start again with: make up"

down: ## Stop & remove the containers (DATA PERSISTS in the volume)
	@$(ENGINE) rm -f $(CONTAINERS) 2>/dev/null || true
	@echo "Removed containers (data volume preserved). Recreate with: make up"

status: ## Show stack status
	$(COMPOSE) ps

logs: ## Follow the backend logs (Ctrl-C to exit)
	$(COMPOSE) logs -f backend

## --- inspect the database --------------------------------------------------
psql: ## Open a psql shell into the database
	$(ENGINE) exec -it $(PG) psql -U catalog -d catalog_ingestion

adminer: ## Start the Adminer DB browser (http://localhost:8081, server "postgres", user/pass "catalog")
	$(COMPOSE) up -d --no-deps adminer
	@echo "Adminer: http://localhost:8081  (System: PostgreSQL, Server: postgres, User: catalog, Password: catalog, DB: catalog_ingestion)"

counts: ## Print row counts for the main tables
	@$(ENGINE) exec $(PG) psql -U catalog -d catalog_ingestion -At -c "\
	  SELECT 'catalog_years', count(*) FROM catalog_years \
	  UNION ALL SELECT 'programs', count(*) FROM programs \
	  UNION ALL SELECT 'requirement_groups', count(*) FROM requirement_groups \
	  UNION ALL SELECT 'requirement_options', count(*) FROM requirement_options \
	  UNION ALL SELECT 'courses', count(*) FROM courses \
	  UNION ALL SELECT 'subjects', count(*) FROM subjects;" | column -t -s '|'

api: ## Print the API URLs
	@echo "Root:  http://localhost:8000/"
	@echo "Docs:  http://localhost:8000/docs"
	@echo "Try:   curl 'http://localhost:8000/academic/courses/search?subject=CS'"

## --- load / refresh data ---------------------------------------------------
db-init: ## Create the database tables (idempotent)
	$(COMPOSE) run --rm ingestion db-init

load-years: ## Discover & store the catalog years
	$(COMPOSE) run --rm ingestion discover-years

load-courses: ## Load ALL courses from PurdueIO for YEAR (fast, no scraping)
	$(COMPOSE) run --rm ingestion import-courses --year $(YEAR) --all-subjects

sync-program: ## Scrape ONE program: make sync-program POID=34693
	$(COMPOSE) run --rm ingestion sync-program --year $(YEAR) --poid $(POID)

sync-programs: ## Scrape ALL degree programs for YEAR (long; honors crawl-delay)
	$(COMPOSE) run --rm ingestion sync-programs --year $(YEAR)

validate: ## Run data validation for YEAR
	$(COMPOSE) run --rm ingestion validate --year $(YEAR)

## --- backup / reset --------------------------------------------------------
backup: ## Dump the database to catalog_db_backup.sql.gz
	$(ENGINE) exec $(PG) pg_dump -U catalog -d catalog_ingestion | gzip > catalog_db_backup.sql.gz
	@echo "Wrote catalog_db_backup.sql.gz"

restore: ## Restore the database from catalog_db_backup.sql.gz
	gunzip -c catalog_db_backup.sql.gz | $(ENGINE) exec -i $(PG) psql -U catalog -d catalog_ingestion

reset: ## DESTROY all data + volumes
	@$(ENGINE) rm -f $(CONTAINERS) 2>/dev/null || true
	@$(ENGINE) volume rm $(VOLUMES) 2>/dev/null || true
