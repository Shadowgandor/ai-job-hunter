#!/usr/bin/env python3
"""Evaluate the LLM-as-judge against hand-labeled faithfulness examples.

Loads data/labels/faithfulness_labels.json, runs the judge on each case,
then reports raw agreement rate and Cohen's kappa with your labels.

Also mixes in deterministic checks (does the reason cite a technology that
actually appears in the snippet?) to demonstrate a hybrid eval approach.

Usage:
    python scripts/evaluate_judge.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers import Job
from analyzer import ScoredJob
from eval_harness import run_eval, EvalResult

LABELS_FILE = Path("data/labels/faithfulness_labels.json")

# Tech tokens to look for in deterministic substring check.
# If the reason mentions one of these but the snippet doesn't, flag it.
_TECH_TOKENS = [
    "copilot", "cursor", "codeium", "tabnine", "claude", "chatgpt", "gpt",
    "openai", "llm", "mcp", "langchain", "tensorflow", "pytorch", "spark",
    "kubernetes", "docker", "react", "angular", "vue", "node", "python",
    "typescript", "java", "kotlin", "swift", "go", "rust",
]


def _load_labels() -> list[dict]:
    if not LABELS_FILE.exists():
        print(f"No labels at {LABELS_FILE}. Run scripts/label_faithfulness.py first.")
        sys.exit(1)
    data = json.loads(LABELS_FILE.read_text())
    if not data:
        print("Label file is empty. Run scripts/label_faithfulness.py first.")
        sys.exit(1)
    return data


def _cohen_kappa(human: list[bool], judge: list[bool]) -> float:
    n = len(human)
    if n == 0:
        return 0.0
    p_o = sum(h == j for h, j in zip(human, judge)) / n
    n_h_yes = sum(human)
    n_j_yes = sum(judge)
    p_e = (n_h_yes * n_j_yes + (n - n_h_yes) * (n - n_j_yes)) / n ** 2
    return (p_o - p_e) / (1 - p_e) if p_e < 1.0 else 1.0


def _kappa_label(k: float) -> str:
    if k < 0:
        return "worse than chance"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def _tech_check(reason: str, snippet: str) -> list[str]:
    """Deterministic check: return tech tokens cited in reason but absent from snippet."""
    reason_lower = reason.lower()
    snippet_lower = snippet.lower()
    return [t for t in _TECH_TOKENS if t in reason_lower and t not in snippet_lower]


def main() -> None:
    labels = _load_labels()
    print(f"\n=== JUDGE EVALUATION ===")
    print(f"Loaded {len(labels)} labeled examples")

    n_human_faithful = sum(1 for l in labels if l["faithful"])
    print(f"  human faithful:     {n_human_faithful}")
    print(f"  human not-faithful: {len(labels) - n_human_faithful}")
    print()

    # Reconstruct ScoredJob objects for the judge
    cases = [
        ScoredJob(
            job=Job(
                title=l["job_title"],
                company=l.get("company", ""),
                location="",
                url="",
                snippet=l["snippet"],
                source="labeled",
            ),
            score=l["llm_score"],
            reason=l["reason"],
        )
        for l in labels
    ]

    print("Running judge...")
    results = run_eval(cases)
    if not results:
        print("No results from judge.")
        return

    human_labels = [l["faithful"] for l in labels]
    judge_labels = [r.faithful for r in results]

    # ── Agreement metrics ──────────────────────────────────────────────────
    n = len(human_labels)
    n_agree = sum(h == j for h, j in zip(human_labels, judge_labels))
    kappa = _cohen_kappa(human_labels, judge_labels)

    tp = sum(1 for h, j in zip(human_labels, judge_labels) if h and j)
    tn = sum(1 for h, j in zip(human_labels, judge_labels) if not h and not j)
    fp = sum(1 for h, j in zip(human_labels, judge_labels) if not h and j)
    fn = sum(1 for h, j in zip(human_labels, judge_labels) if h and not j)

    print(f"Agreement:     {n_agree}/{n} ({n_agree / n:.1%})")
    print(f"Cohen's kappa: {kappa:.2f} ({_kappa_label(kappa)})")
    print()
    print("Confusion matrix (rows=human, cols=judge):")
    print(f"               faithful  not-faithful")
    print(f"  faithful        {tp:<4}      {fn}")
    print(f"  not-faithful    {fp:<4}      {tn}")
    print()

    # ── Deterministic checks ───────────────────────────────────────────────
    tech_flags: list[tuple[dict, list[str]]] = []
    for label, result in zip(labels, results):
        flagged = _tech_check(result.reason, label["snippet"])
        if flagged:
            tech_flags.append((label, flagged))

    print(f"Deterministic tech-citation check ({len(_TECH_TOKENS)} tokens watched):")
    if tech_flags:
        print(f"  {len(tech_flags)} reason(s) cite tech not found in snippet:")
        for label, flagged in tech_flags:
            human_mark = "✓" if label["faithful"] else "✗"
            print(
                f"  {human_mark} '{label['job_title']}' — "
                f"reason mentions {flagged} but snippet doesn't"
            )
    else:
        print("  No tech-citation mismatches found.")
    print()

    # ── Disagreements ──────────────────────────────────────────────────────
    disagreements = [
        (labels[i], results[i])
        for i in range(n)
        if human_labels[i] != judge_labels[i]
    ]

    if not disagreements:
        print("No disagreements — judge matches your labels on every case.")
    else:
        print(f"Disagreements ({len(disagreements)}):")
        for label_item, result in disagreements:
            direction = (
                "Human=faithful, Judge=not-faithful"
                if label_item["faithful"]
                else "Human=not-faithful, Judge=faithful"
            )
            print(f"\n  [{direction}]")
            print(f"  Title:  {result.job_title} (LLM score: {result.llm_score}/10)")
            print(f"  Reason: {result.reason[:120]}...")
            print(f"  Judge:  {result.critique}")
            if label_item.get("notes"):
                print(f"  Yours:  {label_item['notes']}")

    print()


if __name__ == "__main__":
    main()
