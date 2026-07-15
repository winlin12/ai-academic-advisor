"""Shared HTML parsing utilities."""

from __future__ import annotations

import html as html_module

from bs4 import BeautifulSoup


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    value = html_module.unescape(value)
    value = value.replace("\xa0", " ")
    return " ".join(value.split()).strip()


def soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")
