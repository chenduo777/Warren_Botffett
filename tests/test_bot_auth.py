"""Hermetic unit tests for src.bot.auth.parse_allowed_user_ids.

No API calls; no external services; safe for CI without credentials.
"""

import pytest
from src.bot.auth import parse_allowed_user_ids


class TestParseAllowedUserIdsEmpty:
    def test_none_returns_empty_set(self):
        assert parse_allowed_user_ids(None) == set()

    def test_empty_string_returns_empty_set(self):
        assert parse_allowed_user_ids("") == set()

    def test_whitespace_only_returns_empty_set(self):
        assert parse_allowed_user_ids("   ") == set()

    def test_single_comma_returns_empty_set(self):
        assert parse_allowed_user_ids(",") == set()

    def test_multiple_commas_returns_empty_set(self):
        assert parse_allowed_user_ids(",,,") == set()


class TestParseAllowedUserIdsValid:
    def test_three_ids_with_whitespace(self):
        """Whitespace around IDs should be stripped."""
        assert parse_allowed_user_ids("1, 2 ,3") == {1, 2, 3}

    def test_single_id(self):
        assert parse_allowed_user_ids("42") == {42}

    def test_duplicates_are_deduplicated(self):
        """Duplicate IDs must collapse to a unique set."""
        assert parse_allowed_user_ids("1,1,2") == {1, 2}


class TestParseAllowedUserIdsInvalid:
    def test_pure_alpha_raises(self):
        with pytest.raises(ValueError):
            parse_allowed_user_ids("abc")

    def test_mixed_valid_and_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_allowed_user_ids("1,foo,3")

    def test_float_string_raises(self):
        """'1.5' is not a valid int; int('1.5') raises ValueError."""
        with pytest.raises(ValueError):
            parse_allowed_user_ids("1.5")
