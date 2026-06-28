# Ingestion service image.
#
# Based on Microsoft's official Playwright image, which ships Chromium plus every
# required system library (libatk, libnss3, libasound, libgbm, ...) preinstalled and
# pinned to the matching Playwright driver. This is the reason we containerize: the
# host (Ubuntu 26.04) cannot install those libraries without interactive sudo, and the
# AWS WAF JavaScript challenge on catalog.purdue.edu requires a real browser to solve
# legitimately.
#
# The tag MUST match the `playwright` version resolved by pip so the bundled browser
# build is compatible. We pin to 1.60.0 (see pyproject: playwright>=1.44, host has 1.60.0).
FROM mcr.microsoft.com/playwright/python:v1.60.0-noble

# Browsers live here in the base image; never re-download them.
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first for layer caching, then the package as editable so a
# bind-mounted ./src reflects host edits without a rebuild.
COPY pyproject.toml ./
COPY src ./src
RUN pip install -e .

# Default working dir for the CLI; page cache and any outputs land under /data.
ENV PAGE_CACHE_DIR=/data/page_cache
RUN mkdir -p /data/page_cache

ENTRYPOINT ["catalog-ingest"]
CMD ["--help"]
