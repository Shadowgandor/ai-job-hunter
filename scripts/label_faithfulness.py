#!/usr/bin/env python3
"""Interactive CLI for hand-labeling (snippet, reason) pairs as faithful / not-faithful.

For each pair, decide: does the Claude-generated reason make claims that are
actually supported by the source snippet? Reasonable inferences are allowed;
invented specifics (tech, perks, seniority not mentioned) are not faithful.

Aim for 20-30 labeled examples. Focus on jobs where Claude gave a reason —
very low-scoring jobs with terse reasons are less interesting.

Usage:
    python scripts/label_faithfulness.py                  # reads last_run.json
    python scripts/label_faithfulness.py path/to/run.json
"""

import json
import sys
import textwrap
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval_harness import load_scored_jobs

LABELS_FILE = Path("data/labels/faithfulness_labels.json")


def _item_id(sj) -> str:
    """Stable ID for a (job, reason) pair."""
    return f"{sj.job.id}:{abs(hash(sj.reason)) % 10 ** 8}"


def _load_existing() -> dict[str, dict]:
    if not LABELS_FILE.exists():
        return {}
    return {item["id"]: item for item in json.loads(LABELS_FILE.read_text())}


def _save(labels: dict[str, dict]) -> None:
    LABELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    LABELS_FILE.write_text(json.dumps(list(labels.values()), indent=2))


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else "last_run.json"
    if not Path(source).exists():
        print(f"Source file '{source}' not found. Run the pipeline first.")
        sys.exit(1)

    scored_jobs = load_scored_jobs(source)
    # Skip jobs with no real reason (parse errors, unscored)
    scored_jobs = [
        sj for sj in scored_jobs
        if sj.reason and len(sj.reason) > 20 and "error" not in sj.reason.lower()
    ]
    # Sort highest-scoring first — more interesting faithfulness cases
    scored_jobs.sort(key=lambda sj: sj.score, reverse=True)

    labels = _load_existing()
    to_label = [sj for sj in scored_jobs if _item_id(sj) not in labels]

    print(f"\n=== FAITHFULNESS LABELING ===")
    print(f"Already labeled : {len(labels)}")
    print(f"Remaining       : {len(to_label)}")
    print(f"Question: is the REASON grounded in the SNIPPET?")
    print(f"Keys: y=faithful  n=not faithful  q=quit\n")

    try:
        for i, sj in enumerate(to_label, 1):
            item_id = _item_id(sj)
            job = sj.job
            print(
                f"[{i}/{len(to_label)}] Score: {sj.score}/10 — "
                f"{job.title} @ {job.company or '?'}"
            )
            print("─" * 70)
            print("SNIPPET:")
            for line in textwrap.wrap(job.snippet[:600], width=70):
                print(f"  {line}")
            print()
            print("REASON:")
            for line in textwrap.wrap(sj.reason, width=70):
                print(f"  {line}")
            print()

            while True:
                raw = input("Faithful? [y/n/q]: ").strip().lower()
                if raw == "q":
                    raise KeyboardInterrupt
                if raw in ("y", "n"):
                    notes = input("Notes (optional, Enter to skip): ").strip()
                    labels[item_id] = {
                        "id": item_id,
                        "job_title": job.title,
                        "company": job.company or "",
                        "snippet": job.snippet,
                        "reason": sj.reason,
                        "llm_score": sj.score,
                        "faithful": raw == "y",
                        "notes": notes,
                        "labeled_at": datetime.now().isoformat(),
                    }
                    break
                print("  → Enter y, n, or q")

            print()

    except KeyboardInterrupt:
        pass

    _save(labels)
    n_faithful = sum(1 for item in labels.values() if item["faithful"])
    print(f"\nSaved {len(labels)} labels to {LABELS_FILE}")
    print(f"  faithful     {n_faithful}")
    print(f"  not-faithful {len(labels) - n_faithful}")


if __name__ == "__main__":
    main()
