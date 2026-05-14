-- Usage:
-- sqlite3 data/degree_requirements.db < scripts/join_degree_and_courses.sql

ATTACH DATABASE 'data/purdue_catalog.db' AS catalog;

-- 1) Degree -> requirement -> course links with course title from catalog DB
SELECT
  d.degree_name,
  COALESCE(r.requirement_name, 'UNASSIGNED') AS requirement_name,
  drc.course_code,
  c.title AS catalog_course_title
FROM degree_requirement_courses drc
JOIN degrees d
  ON d.degree_id = drc.degree_id
LEFT JOIN degree_requirements r
  ON r.requirement_id = drc.requirement_id
LEFT JOIN catalog.courses c
  ON c.course_code = drc.course_code
ORDER BY d.degree_name, requirement_name, drc.course_code
LIMIT 200;

-- 2) Coverage summary
SELECT
  COUNT(*) AS total_degree_course_links,
  SUM(CASE WHEN c.course_code IS NOT NULL THEN 1 ELSE 0 END) AS links_found_in_catalog
FROM degree_requirement_courses drc
LEFT JOIN catalog.courses c
  ON c.course_code = drc.course_code;

