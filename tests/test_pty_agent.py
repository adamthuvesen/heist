from __future__ import annotations

import sys

from heist import pty_agent


def test_main_without_argv_returns_usage_exit_code(capsys: object) -> None:
    original = sys.argv
    try:
        sys.argv = ["pty_agent"]
        assert pty_agent.main() == 2
    finally:
        sys.argv = original
