#!/usr/bin/env python3
"""Evaluate the prototype classifier against hand-labeled examples.

Loads data/labels/classifier_labels.json, runs classify_ai_native() on each
entry, and reports accuracy, precision, recall, F1, and a threshold sweep.
'unclear' labels are excluded from binary metrics but counted separately.

Usage:
    python scripts/evaluate_classifier.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers import Job
from classifier import classify_ai_native

LABELS_FILE = Path("data/labels/classifier_labels.json")


def _load_labels() -> list[dict]:
    if not LABELS_FILE.exists():
        print(f"No labels found at {LABELS_FILE}. Run scripts/label_classifier.py first.")
        sys.exit(1)
    data = json.loads(LABELS_FILE.read_text())
    if not data:
        print("Label file is empty. Run scripts/label_classifier.py first.")
        sys.exit(1)
    return data


def _binary_metrics(
    actuals: list[bool], scores: list[float], threshold: float
) -> tuple[float, float, float, float]:
    """Return (accuracy, precision, recall, f1) at a given threshold."""
    tp = sum(1 for a, s in zip(actuals, scores) if a and s >= threshold)
    tn = sum(1 for a, s in zip(actuals, scores) if not a and s < threshold)
    fp = sum(1 for a, s in zip(actuals, scores) if not a and s >= threshold)
    fn = sum(1 for a, s in zip(actuals, scores) if a and s < threshold)
    n = len(actuals)
    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return accuracy, precision, recall, f1


def main() -> None:
    all_labels = _load_labels()
    print(f"\n=== CLASSIFIER EVALUATION ===")
    print(f"Loaded {len(all_labels)} labeled examples")

    counts: dict[str, int] = {}
    for l in all_labels:
        counts[l["label"]] = counts.get(l["label"], 0) + 1
    print()
    for k in ("ai-native", "sprinkles", "unclear"):
        print(f"  {k:<12} {counts.get(k, 0)}")

    # Binary evaluation: exclude unclear
    binary = [l for l in all_labels if l["label"] != "unclear"]
    if len(binary) < 5:
        print(f"\nNeed at least 5 binary-labeled examples (have {len(binary)}). Keep labeling.")
        return

    print(f"\n{'─' * 60}")
    print(f"Binary eval (ai-native vs sprinkles, n={len(binary)})")
    print(f"  'unclear' examples excluded: {counts.get('unclear', 0)}")
    print()

    jobs = [
        Job(
            title=l["title"],
            company=l["company"],
            location=l["location"],
            url=l["url"],
            snippet=l["snippet"],
            source="labeled",
        )
        for l in binary
    ]
    raw_scores = classify_ai_native(jobs)
    actuals = [l["label"] == "ai-native" for l in binary]

    # Metrics at default threshold
    acc, prec, rec, f1 = _binary_metrics(actuals, raw_scores, 0.5)
    print(f"Metrics at threshold=0.5 (positive = ai-native):")
    print(f"  Accuracy:  {acc:.1%}")
    print(f"  Precision: {prec:.1%}")
    print(f"  Recall:    {rec:.1%}")
    print(f"  F1:        {f1:.1%}")
    print()

    # Confusion matrix at 0.5
    tp = sum(1 for a, s in zip(actuals, raw_scores) if a and s >= 0.5)
    tn = sum(1 for a, s in zip(actuals, raw_scores) if not a and s < 0.5)
    fp = sum(1 for a, s in zip(actuals, raw_scores) if not a and s >= 0.5)
    fn = sum(1 for a, s in zip(actuals, raw_scores) if a and s < 0.5)
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(f"             ai-native  sprinkles")
    print(f"  ai-native     {tp:<4}       {fn}")
    print(f"  sprinkles     {fp:<4}       {tn}")
    print()

    # Threshold sweep to show precision-recall tradeoff
    print(f"Threshold sweep:")
    print(f"  {'Threshold':>9}  {'Accuracy':>9}  {'Precision':>9}  {'Recall':>8}  {'F1':>6}")
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        a, p, r, f = _binary_metrics(actuals, raw_scores, t)
        print(f"  {t:>9.1f}  {a:>9.1%}  {p:>9.1%}  {r:>8.1%}  {f:>6.1%}")
    print()

    # Sorted prediction list — good for spotting patterns in misclassifications
    ranked = sorted(
        zip(binary, raw_scores),
        key=lambda x: x[1],
        reverse=True,
    )
    print("All predictions (sorted by score, ✓=correct ✗=wrong):")
    for l, score in ranked:
        actual = l["label"] == "ai-native"
        predicted = score >= 0.5
        correct = actual == predicted
        flag = "✓" if correct else "✗"
        tag = "ai-native" if predicted else "sprinkles"
        print(
            f"  {flag} [{score:.3f}→{tag:<10}] actual={l['label']:<12} {l['title'][:50]}"
        )
        if not correct and l.get("notes"):
            print(f"      note: {l['notes']}")
    print()

    # Summary of misclassifications with notes
    wrong = [(l, s) for l, s in ranked if (l["label"] == "ai-native") != (s >= 0.5)]
    if wrong:
        print(f"Misclassified ({len(wrong)}) — review these to improve prototypes:")
        for l, score in wrong:
            direction = (
                "sprinkles predicted as ai-native"
                if score >= 0.5
                else "ai-native predicted as sprinkles"
            )
            print(f"  [{score:.3f}] {l['title']} @ {l['company']} — {direction}")
            if l.get("notes"):
                print(f"    note: {l['notes']}")
        print()


if __name__ == "__main__":
    main()
