"""Tests for course page parser."""

from catalog_ingestion.parse.courses import parse_course_page

# Mirrors the real Acalog preview_course_nopop.php structure (verified live against
# catalog.purdue.edu, catoid=19): content in td.block_content, course code+title in
# <h1 id="course_preview_title">, inline "Credit Hours: 3.00." then the description,
# then labeled fields and a Learning Outcomes section.
SAMPLE_COURSE_HTML = """
<html><head><title>Program: CS 18000 - Purdue University</title></head>
<body>
<table><tr><td class="block_content">
<a href="#">Share this Page</a>
<h1 id="course_preview_title">CS 18000 - Problem Solving And Object-Oriented Programming</h1>
<hr/>
Credit Hours: 3.00.  Introduction to object-oriented programming and problem solving.
Topics include data types, control structures, methods, arrays, and classes.
Restrictions: Must be admitted to the College of Science or College of Engineering.
Attributes: ENG
<hr/>
<strong>Learning Outcomes</strong><br/> 1. Write working programs.
</td></tr></table>
</body></html>
"""


def test_parse_basic_course():
    result = parse_course_page(SAMPLE_COURSE_HTML, "https://catalog.purdue.edu/preview_course_nopop.php?catoid=19&coid=175649")
    assert result is not None
    assert result.subject_code == "CS"
    assert result.course_number == "18000"
    assert result.course_code == "CS 18000"
    assert result.coid == 175649


def test_parse_course_credits():
    result = parse_course_page(SAMPLE_COURSE_HTML, "https://catalog.purdue.edu/preview_course_nopop.php?catoid=19&coid=175649")
    assert result is not None
    assert result.credit_hours_min == 3.0


def test_parse_course_restrictions():
    result = parse_course_page(SAMPLE_COURSE_HTML, "https://catalog.purdue.edu/preview_course_nopop.php?catoid=19&coid=175649")
    assert result is not None
    assert result.restrictions_raw is not None
    assert "College" in result.restrictions_raw


def test_description_excludes_learning_outcomes():
    result = parse_course_page(SAMPLE_COURSE_HTML, "https://catalog.purdue.edu/preview_course_nopop.php?catoid=19&coid=175649")
    assert result.description is not None
    assert "object-oriented programming" in result.description
    # Description must stop before the Learning Outcomes / Restrictions sections.
    assert "Learning Outcomes" not in result.description
    assert "Restrictions" not in result.description


def test_no_prerequisite_hallucination():
    """Purdue catalog pages usually omit prerequisites; the parser must not invent any."""
    html = """
    <table><tr><td class="block_content">
    <h1 id="course_preview_title">CS 25100 - Data Structures And Algorithms</h1>
    <hr/>
    Credit Hours: 3.00.  Running time analysis of algorithms and data structures.
    <hr/><strong>Learning Outcomes</strong><br/> 1. Understand data structures.
    </td></tr></table>
    """
    result = parse_course_page(html, "https://catalog.purdue.edu/preview_course_nopop.php?catoid=19&coid=226823")
    assert result is not None
    assert result.credit_hours_min == 3.0
    assert result.prerequisites_raw is None
    assert result.corequisites_raw is None


def test_returns_none_for_empty_html():
    result = parse_course_page("", "https://catalog.purdue.edu/preview_course_nopop.php?catoid=19&coid=1")
    assert result is None
