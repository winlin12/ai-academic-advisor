#!/usr/bin/env bash
set -euo pipefail

# Run the full Purdue catalog data pipeline:
# 1) Parse catalog PDF into JSON
# 2) Normalize courses + prerequisites
# 3) Build catalog SQLite DB
# 4) Extract degrees + degree requirements
# 5) Build degree requirements SQLite DB
# 6) Build unified academic SQLite DB
#
# Usage:
#   scripts/build_all_databases.sh [path-to-catalog-pdf]
#
# Example:
#   scripts/build_all_databases.sh "2025-26+University+Catalog-Final.pdf"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
PDF_PATH="${1:-${ROOT_DIR}/2025-26+University+Catalog-Final.pdf}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Error: Python not found at ${PYTHON_BIN}"
  echo "Create the venv first or update this script to use your Python path."
  exit 1
fi

if [[ ! -f "${PDF_PATH}" ]]; then
  echo "Error: Catalog PDF not found: ${PDF_PATH}"
  exit 1
fi

echo "==> Using Python: ${PYTHON_BIN}"
echo "==> Using catalog PDF: ${PDF_PATH}"
echo

echo "==> [1/6] Parsing catalog PDF to JSON..."
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/parse_purdue_catalog.py" \
  --input "${PDF_PATH}" \
  --output-dir "${ROOT_DIR}/data"

echo
echo "==> [2/6] Normalizing courses JSON..."
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/normalize_courses_json.py" \
  --input "${ROOT_DIR}/data/courses.json" \
  --output-dir "${ROOT_DIR}/data/normalized"

echo
echo "==> [3/6] Loading catalog data into SQLite..."
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/load_catalog_to_sqlite.py" \
  --courses "${ROOT_DIR}/data/normalized/courses_normalized.json" \
  --edges "${ROOT_DIR}/data/normalized/prerequisite_edges.json" \
  --issues "${ROOT_DIR}/data/normalized/normalization_issues.json" \
  --snippets "${ROOT_DIR}/data/degree_requirement_snippets.json" \
  --db "${ROOT_DIR}/data/purdue_catalog.db"

echo
echo "==> [4/6] Extracting degrees and requirements..."
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/extract_degree_requirements.py" \
  --input-pdf "${PDF_PATH}" \
  --courses-json "${ROOT_DIR}/data/courses.json" \
  --output-dir "${ROOT_DIR}/data/degree_extracted"

echo
echo "==> [5/6] Loading degree requirements into SQLite..."
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/load_degree_requirements_to_sqlite.py" \
  --degrees "${ROOT_DIR}/data/degree_extracted/degrees.json" \
  --requirements "${ROOT_DIR}/data/degree_extracted/degree_requirements.json" \
  --requirement-courses "${ROOT_DIR}/data/degree_extracted/degree_requirement_courses.json" \
  --issues "${ROOT_DIR}/data/degree_extracted/degree_extraction_issues.json" \
  --summary "${ROOT_DIR}/data/degree_extracted/degree_extraction_summary.json" \
  --catalog-db "${ROOT_DIR}/data/purdue_catalog.db" \
  --db "${ROOT_DIR}/data/degree_requirements.db"

echo
echo "==> [6/6] Building unified academic SQLite DB..."
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/build_unified_academic_db.py" \
  --catalog-db "${ROOT_DIR}/data/purdue_catalog.db" \
  --degree-db "${ROOT_DIR}/data/degree_requirements.db" \
  --output-db "${ROOT_DIR}/data/purdue_academic.db"

echo
echo "Done."
echo "Generated DBs:"
echo "  - ${ROOT_DIR}/data/purdue_catalog.db"
echo "  - ${ROOT_DIR}/data/degree_requirements.db"
echo "  - ${ROOT_DIR}/data/purdue_academic.db"
