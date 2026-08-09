"""Tests for the read-only admin browsing service — pure parts only (no DB required)."""

import pytest

from app.services import admin_db
from app.services.admin_db import UnknownTableError, fetch_rows, row_select_expression


def test_fetch_rows_rejects_unknown_table():
    with pytest.raises(UnknownTableError):
        fetch_rows("pg_shadow")


def test_fetch_rows_rejects_sql_injection_shaped_names():
    with pytest.raises(UnknownTableError):
        fetch_rows("programs; DROP TABLE programs--")


def test_row_select_expression_plain_table():
    assert row_select_expression("programs") == "to_jsonb(t)"


def test_row_select_expression_strips_hidden_columns():
    assert row_select_expression("academic_rules") == "to_jsonb(t) - 'embedding'"


def test_every_hidden_column_table_is_browsable():
    # A HIDDEN_COLUMNS entry for a non-whitelisted table would be dead config.
    for table in admin_db.HIDDEN_COLUMNS:
        assert table in admin_db.BROWSABLE_TABLES


def test_fetch_rows_clamps_page_size(monkeypatch):
    captured: dict = {}

    class FakeCursor:
        def execute(self, query, params=None):
            captured["query"] = query
            captured["params"] = params

        def fetchone(self):
            return {"n": 0}

        def fetchall(self):
            return []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(admin_db, "_connect", lambda: FakeConnection())
    result = fetch_rows("programs", limit=9999, offset=-5)
    assert result.limit == admin_db.MAX_PAGE_SIZE
    assert result.offset == 0
    assert captured["params"] == (admin_db.MAX_PAGE_SIZE, 0)
