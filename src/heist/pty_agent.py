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
    exit_code = os.waitstatus_to_exitcode(status)
    # A signal death comes back negative (e.g. -15 for SIGTERM). Returning it
    # as-is makes SystemExit wrap it (256 + code → 241), which downstream reads
    # as a meaningless exit code. Re-encode as the shell convention 128 + signum
    # so the runner can recognise a signal kill (e.g. a harness abort).
    if exit_code < 0:
        return 128 - exit_code
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
