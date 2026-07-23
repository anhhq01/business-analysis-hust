"""
Module 7 periodic scheduler.

Runs Module_07/monitoring.py every N minutes ONLY when Module 6 has appended
new scoring events to Module_07/inputs/module6_scored_events.jsonl.

Usage (from repo root):
  python Module_07/scheduler.py --interval-seconds 300

One-shot mode (single check then exit):
  python Module_07/scheduler.py --once
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENTS_LOG = ROOT / "Module_07" / "inputs" / "module6_scored_events.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "Module_07" / "outputs"
DEFAULT_STATE_PATH = DEFAULT_OUTPUT_DIR / "monitor_scheduler_state.json"
DEFAULT_MONITOR_SCRIPT = ROOT / "Module_07" / "monitoring.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Periodic trigger for Module 7 monitoring")
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--events-log", type=Path, default=DEFAULT_EVENTS_LOG)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--monitor-script", type=Path, default=DEFAULT_MONITOR_SCRIPT)
    parser.add_argument("--dashboard-max-rows", type=int, default=50000)
    parser.add_argument("--rolling-window", type=int, default=168)
    parser.add_argument("--reference-max-step", type=int, default=354)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--run-on-start", action="store_true")
    return parser.parse_args()


def load_state(state_file: Path) -> dict:
    if not state_file.exists():
        return {"last_size": 0, "last_mtime": 0.0, "last_run_at": None}
    try:
        return json.loads(state_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"last_size": 0, "last_mtime": 0.0, "last_run_at": None}


def save_state(state_file: Path, state: dict) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")


def log_status(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def has_new_events(events_log: Path, state: dict) -> tuple[bool, dict]:
    if not events_log.exists():
        return False, state

    stat = events_log.stat()
    size = int(stat.st_size)
    mtime = float(stat.st_mtime)

    last_size = int(state.get("last_size", 0))
    last_mtime = float(state.get("last_mtime", 0.0))
    changed = (size > last_size) or (mtime > last_mtime and size != last_size)

    next_state = dict(state)
    next_state["last_size"] = size
    next_state["last_mtime"] = mtime
    return changed, next_state


def run_monitoring(args: argparse.Namespace) -> bool:
    cmd = [
        sys.executable,
        str(args.monitor_script),
        "--events-log",
        str(args.events_log),
        "--dashboard-max-rows",
        str(args.dashboard_max_rows),
        "--rolling-window",
        str(args.rolling_window),
        "--reference-max-step",
        str(args.reference_max_step),
    ]
    log_status("Triggering Module 7 monitoring run...")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        log_status("Monitoring run failed")
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr)
        return False

    if proc.stdout:
        print(proc.stdout.strip())
    log_status("Monitoring run completed")
    return True


def main() -> None:
    args = parse_args()
    state = load_state(args.state_file)

    if args.run_on_start:
        ok = run_monitoring(args)
        if ok:
            state["last_run_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            save_state(args.state_file, state)

    log_status(
        f"Scheduler started: interval={args.interval_seconds}s, events_log={args.events_log}"
    )

    while True:
        changed, state = has_new_events(args.events_log, state)
        if changed:
            ok = run_monitoring(args)
            if ok:
                state["last_run_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            save_state(args.state_file, state)
        else:
            log_status("No new Module 6 events - skip this cycle")

        if args.once:
            break

        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
