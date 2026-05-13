import json
from functools import lru_cache
from pathlib import Path

from app.models.schemas import Course

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "courses.json"


@lru_cache
def load_catalog() -> list[Course]:
    with DATA_PATH.open("r", encoding="utf-8") as f:
        raw_courses = json.load(f)
    return [Course(**course) for course in raw_courses]
