"""Run a command under a pseudo-terminal, forwarding its output to this process.

Some agent CLIs (notably ``opencode run``) only stream when their stdout is a
TTY and hang indefinitely when it is a plain file — which is exactly how the
harness captures output (``subprocess.Popen(stdout=<file>)``). Wrapping the CLI
in this shim gives it a PTY so it streams normally, while the harness still
captures the combined output to its stdout file.

Usage: ``python -m heist.pty_agent <command> [args...]``
"""

from __future__ import annotations

import os
import pty
import sys


def main() -> int:
    argv = sys.argv[1:]
    if not argv:
        print("usage: python -m heist.pty_agent <command> [args...]", file=sys.stderr)
        return 2
    status = pty.spawn(argv)
    return os.waitstatus_to_exitcode(status)


if __name__ == "__main__":
    raise SystemExit(main())
