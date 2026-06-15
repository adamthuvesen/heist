from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def main() -> None:
    mode = sys.argv[1]
    if mode == "delayed_pass":
        time.sleep(0.3)
        Path("answer.txt").write_text("yes\n")
    elif mode in {"pass", "reported", "slow_usage", "exit_nonzero"}:
        Path("answer.txt").write_text("yes\n")
    elif mode == "fail":
        Path("answer.txt").write_text("no\n")
    elif mode == "cheat":
        # Simulates an agent that read the answer key: it solves the task AND its
        # transcript references the hidden path (injected via HEIST_CHEAT_ECHO).
        Path("answer.txt").write_text("yes\n")
        print(os.environ.get("HEIST_CHEAT_ECHO", ""))
    elif mode == "sleep":
        time.sleep(10)
    elif mode in {"stderr_only", "multiline_stream"}:
        Path("answer.txt").write_text("yes\n")
    else:
        raise SystemExit(f"unknown fake mode: {mode}")

    if mode == "reported":
        print('{"usage": {"input_tokens": 100, "output_tokens": 20}, "total_cost_usd": 1.23}')
    elif mode == "stderr_only":
        # Real agents (codex, claude) sometimes emit usage to stderr. Make sure
        # the runner's combined-stream parse covers this.
        print('{"usage": {"input_tokens": 100, "output_tokens": 20}}', file=sys.stderr)
    elif mode == "multiline_stream":
        # Per-turn usage events. Important for C1 (max-vs-sum) verification:
        # cumulative-stream providers emit growing totals; delta-stream
        # providers emit per-turn slices. This fake mimics deltas.
        for _i in range(3):
            print('{"usage": {"input_tokens": 50, "output_tokens": 10}}', flush=True)
    else:
        # Default `pass`/`fail`/`delayed_pass`/`slow_usage`: single usage line.
        # Also writes usage to HEIST_USAGE_FILE if set, so tests can read it
        # deterministically without depending on pipe-buffer flush timing.
        usage_line = '{"usage": {"input_tokens": 100, "output_tokens": 20}}'
        print(usage_line, flush=True)
        usage_path = os.environ.get("HEIST_USAGE_FILE")
        if usage_path:
            Path(usage_path).write_text(usage_line + "\n")

    if mode == "slow_usage":
        time.sleep(10)
    if mode == "exit_nonzero":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
