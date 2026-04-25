#!/usr/bin/env python3
"""Interactive CLI for hand-labeling job postings as ai-native / sprinkles / unclear.

For each posting, decide:
  ai-native  — the core job is about helping dev teams adopt AI tools/practices
  sprinkles  — a standard software role that happens to use AI in the product
  unclear    — genuinely ambiguous; the posting doesn't give enough signal

Unclear labels are excluded from binary metrics but captured for inspection.
Aim for 40-50 labeled examples with roughly balanced classes.

Usage:
    python scripts/label_classifier.py                  # reads last_run.json
    python scripts/label_classifier.py path/to/run.json
"""

import json
import sys
import textwrap
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval_harness import load_scored_jobs

LABELS_FILE = Path("data/labels/classifier_labels.json")
VALID = {"n": "ai-native", "s": "sprinkles", "u": "unclear"}


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
    labels = _load_existing()
    to_label = [sj for sj in scored_jobs if sj.job.id not in labels]

    print(f"\n=== CLASSIFIER LABELING ===")
    print(f"Already labeled : {len(labels)}")
    print(f"Remaining       : {len(to_label)}")
    print(f"Keys: n=ai-native  s=sprinkles  u=unclear  q=quit\n")

    try:
        for i, sj in enumerate(to_label, 1):
            job = sj.job
            header = (
                f"[{i}/{len(to_label)}] {job.title} @ {job.company or '?'} "
                f"({job.location or '?'})"
            )
            print(header)
            print(f"  embed_sim={sj.embed_sim:.3f}  ai_native_score={sj.ai_native_score:.3f}")
            print(f"  {job.url}")
            print("─" * 70)
            for line in textwrap.wrap(job.snippet[:600], width=70):
                print(f"  {line}")
            print()

            while True:
                raw = input("Label [n/s/u/q]: ").strip().lower()
                if raw == "q":
                    raise KeyboardInterrupt
                if raw in VALID:
                    notes = input("Notes (optional, Enter to skip): ").strip()
                    labels[job.id] = {
                        "id": job.id,
                        "title": job.title,
                        "company": job.company or "",
                        "location": job.location or "",
                        "snippet": job.snippet,
                        "url": job.url,
                        "label": VALID[raw],
                        "notes": notes,
                        "labeled_at": datetime.now().isoformat(),
                    }
                    break
                print("  → Enter n, s, u, or q")

            print()

    except KeyboardInterrupt:
        pass

    _save(labels)
    counts = {}
    for item in labels.values():
        counts[item["label"]] = counts.get(item["label"], 0) + 1
    print(f"\nSaved {len(labels)} labels to {LABELS_FILE}")
    for k, v in sorted(counts.items()):
        print(f"  {k:<12} {v}")


if __name__ == "__main__":
    main()
