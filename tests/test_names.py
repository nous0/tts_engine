"""Tests for guessing a speaker's gender from their name."""

from __future__ import annotations

import pytest

from app.core.names import guess_gender, normalize


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Alice", "female"),
        ("Bob", "male"),
        ("  emma  ", "female"),
        ("MICHAEL", "male"),
        ("Alice Johnson", "female"),
        ("Dr. David Smith", "male"),
        # Vietnamese: given name last, with or without diacritics.
        ("Lan", "female"),
        ("Hương", "female"),
        ("Huong", "female"),
        ("Nguyễn Thị Hoa", "female"),
        ("Trần Văn Minh", "male"),
        ("Lê Tuấn", "male"),
        ("Đức", "male"),
        # Titles.
        ("Mr Nobody", "male"),
        ("Mrs Robinson", "female"),
        ("Chị Minh", "female"),
        ("Ông Minh", "male"),
    ],
)
def test_guess_gender_known_names(name, expected):
    assert guess_gender(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "Host",
        "Guest",
        "Speaker",
        "Speaker 1",
        "Narrator",
        "Minh",  # unisex in Vietnamese
        "Linh",
        "Dung",  # Dung (female) vs Dũng (male): ambiguous once diacritics are dropped
        "Sam",
        "",
        "   ",
    ],
)
def test_guess_gender_unknown_or_ambiguous(name):
    assert guess_gender(name) is None


def test_guess_gender_conflicting_first_and_last_names_is_unknown():
    assert guess_gender("Alice Tuan") is None


def test_a_lone_title_word_is_not_a_title():
    # "Co"/"Ba" only mark gender in front of a name.
    assert guess_gender("Ba") is None


def test_normalize_strips_diacritics_and_splits():
    assert normalize("Nguyễn Thị Hương") == ["nguyen", "thi", "huong"]
    assert normalize("Đặng_Văn-Lâm") == ["dang", "van", "lam"]
