from requirementsync.catalog_scraper import CatalogPage, extract_course_refs, parse_program_page


SAMPLE_PROGRAM_HTML = """
<html>
  <head><title>Program: Computer Science, BS - Purdue University</title></head>
  <body>
    <h1>Program: Computer Science, BS</h1>
    <nav>College of Science</nav>
    <h2>Degree Requirements</h2>
    <h3>Departmental/Program Major Courses</h3>
    <p>CS 18000 - Problem Solving And Object-Oriented Programming, 4 credits</p>
    <p>Choose one of CS 18200 or CS 24000, 3 credits</p>
    <h3>Mathematics Requirement</h3>
    <p>MA 16100 - Plane Analytic Geometry And Calculus I, 5 credits</p>
    <p>Print Degree Planner</p>
  </body>
</html>
"""


def test_parse_program_page_extracts_requirement_hierarchy():
    page = CatalogPage(
        "https://catalog.purdue.edu/preview_program.php?catoid=99&poid=1",
        SAMPLE_PROGRAM_HTML,
    )

    program = parse_program_page(page, catalog_year=2026)

    assert program is not None
    assert program.program_title == "Computer Science"
    assert program.degree_code == "BS"
    assert program.school == "College of Science"
    assert program.parser_status == "parsed"
    assert [block["title"] for block in program.blocks] == [
        "Degree Requirements",
        "Departmental/Program Major Courses",
        "Mathematics Requirement",
    ]
    assert program.blocks[1]["rules"][1]["rule_type"] == "choice_group"
    assert len(program.blocks[1]["rules"][1]["options"]) == 2


def test_extract_course_refs_normalizes_common_purdue_code_shapes():
    refs = extract_course_refs("Take CS18000, MA 16100, and ECE-20001.")

    assert [ref["normalized_code"] for ref in refs] == ["CS18000", "MA16100", "ECE20001"]
