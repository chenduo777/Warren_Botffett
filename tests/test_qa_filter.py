"""Hermetic unit tests for src.qa._build_filter.

_build_filter returns a Milvus boolean expression string (or None).
No API calls; no external services; safe for CI without credentials.
"""

import pytest
from src.qa import _build_filter


class TestBuildFilterNoConstraints:
    def test_all_none_returns_none(self):
        """No args → no filter → None."""
        assert _build_filter(None, None, None) is None

    def test_default_args_returns_none(self):
        assert _build_filter() is None


class TestBuildFilterYearOnly:
    def test_year_only(self):
        """year=N → equality expression."""
        assert _build_filter(year=2008) == "year == 2008"

    def test_year_zero(self):
        """Edge: year=0 is a valid int (unlikely domain value, but shouldn't crash)."""
        assert _build_filter(year=0) == "year == 0"


class TestBuildFilterRanges:
    def test_start_year_only(self):
        assert _build_filter(start_year=1980) == "year >= 1980"

    def test_end_year_only(self):
        assert _build_filter(end_year=2019) == "year <= 2019"

    def test_both_start_and_end(self):
        assert _build_filter(start_year=1980, end_year=2019) == "year >= 1980 && year <= 2019"


class TestBuildFilterConflict:
    def test_year_with_start_year_raises(self):
        with pytest.raises(ValueError):
            _build_filter(year=2008, start_year=1980)

    def test_year_with_end_year_raises(self):
        with pytest.raises(ValueError):
            _build_filter(year=2008, end_year=2019)

    def test_year_with_both_raises(self):
        with pytest.raises(ValueError):
            _build_filter(year=2008, start_year=1980, end_year=2019)
