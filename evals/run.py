"""Eval runner: grade recorded turns and report metric pass rates.

Usage:
    python -m evals.run runs/turns.jsonl
    python -m evals.run runs/turns.jsonl --min-pass 1.0
"""

import json
import statistics
import sys

from evals.metrics import METRICS


def load_turns(path: str) -> list[dict]:
    turns = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                turns.append(json.loads(line))
    return turns


def main(argv: list[str]) -> int:
    path = argv[1] if len(argv) > 1 else "runs/turns.jsonl"
    try:
        turns = load_turns(path)
    except FileNotFoundError:
        print(f"No turns file at {path}; run the agent first.")
        return 2
    if not turns:
        print("No turns recorded yet.")
        return 2

    totals = {name: [0, 0] for name in METRICS}
    failures: list[tuple[str, str, str]] = []
    for turn in turns:
        turn_id = turn.get("turn_id") or turn.get("started_at", "?")
        for name, func in METRICS.items():
            try:
                passed, detail = func(turn)
            except Exception as error:
                passed, detail = False, f"metric crashed: {error}"
            totals[name][1] += 1
            if passed:
                totals[name][0] += 1
            else:
                failures.append((str(turn_id), name, detail))

    print(f"Graded {len(turns)} turns from {path}\n")
    all_pass = True
    for name, (passed, total) in totals.items():
        status = "PASS" if passed == total else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"[{status}] {name}: {passed}/{total} ({100 * passed / total:.1f}%)")
    if failures:
        print("\nFailures:")
        for turn_id, name, detail in failures:
            print(f"  turn {turn_id} {name}: {detail}")

    _print_stats(turns)
    return 0 if all_pass else 1


def _pct(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(pct / 100 * len(ordered)), len(ordered) - 1)
    return ordered[index]


def _print_stats(turns: list[dict]) -> None:
    tokens_in = [
        sum((call.get("prompt_tokens") or 0) for call in turn.get("model_calls", []))
        for turn in turns
    ]
    model_lat = [
        call.get("latency_s", 0) or 0
        for turn in turns
        for call in turn.get("model_calls", [])
    ]
    tool_lat = [
        trace.get("latency_s", 0) or 0
        for turn in turns
        for trace in turn.get("tool_trace", [])
    ]
    verdicts: dict[str, int] = {}
    for turn in turns:
        for trace in turn.get("tool_trace", []):
            verdict = trace.get("verdict", "unknown")
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
    executed = verdicts.get("executed", 0)
    n_turns = len(turns)

    print("\nNumbers:")
    print(f"  turns graded: {n_turns}")
    print(f"  input tokens/turn: p50 {statistics.median(tokens_in):.0f}  p95 {_pct(tokens_in, 95):.0f}")
    print(f"  model latency/call (s): p50 {statistics.median(model_lat) if model_lat else 0:.2f}  p95 {_pct(model_lat, 95):.2f}")
    print(f"  tool latency/call (s): p50 {statistics.median(tool_lat) if tool_lat else 0:.2f}  p95 {_pct(tool_lat, 95):.2f}")
    print(f"  tool calls executed: {executed}  ({executed / n_turns:.1f}/turn)")
    print(f"  verdicts: {verdicts}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
