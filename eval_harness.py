"""Eval harness: verifies that Claude's match reasons are grounded in source postings.

Uses an LLM-as-judge approach — a second Claude call checks whether the
generated reason makes claims that are actually supported by the job snippet
(faithfulness check, not a quality/relevance check).

Usage:
    python eval_harness.py                  # Evaluate last_run.json
    python eval_harness.py results.json     # Evaluate a specific file
    python eval_harness.py --synthetic      # Run on built-in test cases (no file needed)
"""

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import anthropic

import config
from analyzer import ScoredJob
from scrapers import Job

logger = logging.getLogger(__name__)

_EVAL_SYSTEM_PROMPT = """You are a factual-consistency evaluator for job-match summaries.

Given a job posting (title, company, location, snippet) and an AI-generated match reason,
judge whether the reason is FAITHFUL to the source text.

A reason is faithful when:
- Every specific claim it makes can be verified or reasonably inferred from the posting
- It does not invent technologies, requirements, or perks not mentioned in the posting
- It does not misrepresent the role's focus or seniority level

Reasonable inferences are allowed (e.g., inferring "senior" from "10 years required").
Vague statements like "good culture" that go beyond the snippet are mildly unfaithful.
Invented details ("requires Copilot certification", "offers equity") are clearly unfaithful.

Respond ONLY with JSON — no markdown, no extra text:
{"faithful": true, "score": 0.85, "critique": "No issues found"}

score: 0.0 = completely hallucinated, 1.0 = perfectly faithful."""


@dataclass
class EvalResult:
    job_title: str
    llm_score: int        # Original 0-10 relevance score from Claude
    reason: str           # The reason string being evaluated
    faithful: bool
    faithfulness_score: float  # 0.0–1.0
    critique: str


def run_eval(
    scored_jobs: list[ScoredJob],
    client: anthropic.Anthropic | None = None,
) -> list[EvalResult]:
    """Evaluate faithfulness of Claude's reasons against their source postings."""
    if not client:
        if not config.ANTHROPIC_API_KEY:
            logger.error("ANTHROPIC_API_KEY not set — cannot run eval")
            return []
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    results = []
    for sj in scored_jobs:
        result = _eval_one(client, sj)
        results.append(result)
        flag = "FAITHFUL" if result.faithful else "NOT FAITHFUL"
        logger.info(f"  [{flag} {result.faithfulness_score:.2f}] {result.job_title}")

    return results


def _eval_one(client: anthropic.Anthropic, sj: ScoredJob) -> EvalResult:
    job = sj.job
    user_msg = (
        f"## Job Posting\n"
        f"Title: {job.title}\n"
        f"Company: {job.company or 'Unknown'}\n"
        f"Location: {job.location or 'Unknown'}\n"
        f"Snippet: {job.snippet}\n\n"
        f"## AI-Generated Match Reason\n{sj.reason}"
    )

    try:
        resp = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=256,
            system=_EVAL_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = resp.content[0].text.strip()
        # Strip any stray markdown fences
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(text)
        return EvalResult(
            job_title=job.title,
            llm_score=sj.score,
            reason=sj.reason,
            faithful=bool(data.get("faithful", False)),
            faithfulness_score=float(data.get("score", 0.0)),
            critique=data.get("critique", ""),
        )

    except (json.JSONDecodeError, anthropic.APIError) as e:
        logger.error(f"Eval failed for '{job.title}': {e}")
        return EvalResult(
            job_title=job.title,
            llm_score=sj.score,
            reason=sj.reason,
            faithful=False,
            faithfulness_score=0.0,
            critique=f"Eval error: {e}",
        )


# ---------- Report ----------

def print_report(results: list[EvalResult]) -> None:
    if not results:
        print("No results to report.")
        return

    n = len(results)
    n_faithful = sum(1 for r in results if r.faithful)
    mean_score = sum(r.faithfulness_score for r in results) / n

    print(f"\n{'=' * 60}")
    print(f"FAITHFULNESS EVAL — {n} reasons evaluated")
    print(f"{'=' * 60}")
    print(f"Faithful:   {n_faithful}/{n} ({100 * n_faithful / n:.0f}%)")
    print(f"Mean score: {mean_score:.3f}")
    print()

    for r in sorted(results, key=lambda x: x.faithfulness_score):
        flag = "✓" if r.faithful else "✗"
        print(f"  {flag} [{r.faithfulness_score:.2f}] (LLM score: {r.llm_score}/10) {r.job_title}")
        if r.critique and r.critique != "No issues found":
            print(f"       {r.critique}")
    print()


# ---------- Synthetic test cases ----------

def _synthetic_cases() -> list[ScoredJob]:
    """Three built-in cases: faithful, hallucinated, and borderline."""
    return [
        # Case 1: reason is grounded — should be rated faithful
        ScoredJob(
            job=Job(
                title="AI Enablement Engineer",
                company="FinTech BV",
                location="Amsterdam",
                url="https://example.com/job/1",
                snippet=(
                    "Help our engineering teams adopt GitHub Copilot and Claude Code. "
                    "You'll run workshops, measure developer productivity, and work with "
                    "our platform team to embed AI tooling into CI/CD pipelines. "
                    "Based in Amsterdam, hybrid."
                ),
                source="synthetic",
            ),
            score=9,
            reason=(
                "Near-perfect match — focuses on GitHub Copilot and Claude Code adoption, "
                "with workshops and productivity measurement. Amsterdam (Utrecht area). "
                "Exactly the AI tooling enablement role described in the profile."
            ),
        ),
        # Case 2: reason invents details not in snippet — should be NOT faithful
        ScoredJob(
            job=Job(
                title="Senior Full-Stack Developer",
                company="Startup NL",
                location="Amsterdam",
                url="https://example.com/job/2",
                snippet=(
                    "We're looking for a senior developer to join our product team. "
                    "React frontend, Node.js backend. Experience with REST APIs required. "
                    "Nice to have: some AI experience."
                ),
                source="synthetic",
            ),
            score=4,
            reason=(
                "Partial match — the role includes a dedicated AI tooling mentorship track, "
                "requires GitHub Copilot expertise, and offers significant AI adoption budget. "
                "Good location but requires ML experience."
            ),
        ),
        # Case 3: borderline — vague but not obviously wrong
        ScoredJob(
            job=Job(
                title="AI Platform Engineer",
                company="Scale-Up NL",
                location="Utrecht",
                url="https://example.com/job/3",
                snippet=(
                    "Join our AI platform team to build internal developer tooling. "
                    "You'll integrate AI assistants and automate repetitive engineering tasks. "
                    "Experience with LLMs and API integrations required."
                ),
                source="synthetic",
            ),
            score=7,
            reason=(
                "Strong match — building internal developer AI tooling in Utrecht. "
                "Focuses on AI integration for engineering teams without requiring deep ML expertise."
            ),
        ),
    ]


# ---------- Persistence ----------

def load_scored_jobs(path: str) -> list[ScoredJob]:
    """Load a ScoredJob list from a JSON file written by main.py."""
    raw = json.loads(Path(path).read_text())
    jobs = []
    for item in raw:
        jd = item["job"]
        job = Job(
            title=jd["title"],
            company=jd["company"],
            location=jd["location"],
            url=jd["url"],
            snippet=jd["snippet"],
            source=jd["source"],
            date_found=jd.get("date_found", ""),
        )
        jobs.append(
            ScoredJob(
                job=job,
                score=item["score"],
                reason=item["reason"],
                embed_sim=item.get("embed_sim", 0.0),
                ai_native_score=item.get("ai_native_score", 0.0),
            )
        )
    return jobs


# ---------- CLI ----------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if "--synthetic" in sys.argv:
        cases = _synthetic_cases()
        print(f"Running faithfulness eval on {len(cases)} synthetic test cases...")
    else:
        result_file = next(
            (a for a in sys.argv[1:] if not a.startswith("--")), "last_run.json"
        )
        p = Path(result_file)
        if not p.exists():
            print(
                f"File '{result_file}' not found. "
                "Run the main pipeline first, or use --synthetic for built-in test cases."
            )
            sys.exit(1)
        cases = load_scored_jobs(str(p))
        print(f"Loaded {len(cases)} scored jobs from {result_file}")

    results = run_eval(cases)
    print_report(results)
