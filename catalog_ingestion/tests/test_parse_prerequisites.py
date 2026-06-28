"""Tests for prerequisite parser."""

from catalog_ingestion.parse.prerequisites import parse_prerequisite_text, CONFIDENCE_HIGH, CONFIDENCE_LOW


def test_single_course():
    result = parse_prerequisite_text("Prerequisite: CS 18000.")
    assert result.parsed_json is not None
    assert result.parsed_json["type"] == "COURSE"
    assert result.parsed_json["course"] == "CS 18000"
    assert result.parse_confidence == CONFIDENCE_HIGH


def test_and_expression():
    result = parse_prerequisite_text("Prerequisite: CS 18000 and MA 16100.")
    assert result.parsed_json is not None
    assert result.parsed_json["type"] == "AND"
    assert len(result.parsed_json["children"]) == 2


def test_or_expression():
    result = parse_prerequisite_text("Prerequisite: MA 16100 or MA 16500.")
    assert result.parsed_json is not None
    assert result.parsed_json["type"] == "OR"


def test_nested_expression():
    raw = "Prerequisite: CS 18000 and (MA 16100 or MA 16500)."
    result = parse_prerequisite_text(raw)
    assert result.raw_text == raw
    assert result.parsed_json is not None
    root = result.parsed_json
    assert root["type"] == "AND"
    children = root["children"]
    assert any(c["type"] == "COURSE" and c["course"] == "CS 18000" for c in children)
    or_child = next(c for c in children if c["type"] == "OR")
    or_courses = {c["course"] for c in or_child["children"]}
    assert "MA 16100" in or_courses
    assert "MA 16500" in or_courses


def test_empty_preserves_raw():
    result = parse_prerequisite_text("")
    assert result.raw_text == ""
    assert result.parsed_json is None


def test_no_courses_low_confidence():
    result = parse_prerequisite_text("Must be admitted to the college.")
    assert result.parse_confidence == CONFIDENCE_LOW


def test_raw_text_always_preserved():
    raw = "Prerequisite: Some complex text that cannot be parsed [x12] 99."
    result = parse_prerequisite_text(raw)
    assert result.raw_text == raw
