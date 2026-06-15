from __future__ import annotations

import pytest

from heist.config import _coerce_value


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1", True),
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
    ],
)
def test_env_var_bool_coercion(raw: str, expected: bool) -> None:
    assert _coerce_value("progress", raw) is expected
    assert _coerce_value("exit_on_failure", raw) is expected


@pytest.mark.parametrize("raw", ["", "anything-else"])
def test_env_var_bool_rejects_invalid_values(raw: str) -> None:
    with pytest.raises(ValueError, match="must be a boolean"):
        _coerce_value("progress", raw)


@pytest.mark.parametrize("raw,expected", [("1", 1), ("42", 42)])
def test_env_var_int_coercion(raw: str, expected: int) -> None:
    assert _coerce_value("jobs", raw) == expected
    assert _coerce_value("timeout_s", raw) == expected


def test_env_var_int_rejects_non_integer() -> None:
    with pytest.raises(ValueError, match="expected int"):
        _coerce_value("jobs", "high")


@pytest.mark.parametrize("raw", ["0", "-1"])
def test_env_var_int_rejects_non_positive_values(raw: str) -> None:
    with pytest.raises(ValueError, match="jobs must be >= 1"):
        _coerce_value("jobs", raw)


def test_env_var_unknown_key_passes_through_as_string() -> None:
    # Current behaviour: unknown keys fall through as strings. This pins it so
    # a regression (e.g. accidentally coercing every key as bool) is caught.
    assert _coerce_value("output_dir", "/tmp/heist-runs") == "/tmp/heist-runs"
