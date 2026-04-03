"""
Analyze job relevance using Claude API.

Sends batches of jobs to Claude Haiku for fast, cheap relevance scoring
against the configured job profile.
"""

import json
import logging
from dataclasses import dataclass

import anthropic

import config
from scrapers import Job

logger = logging.getLogger(__name__)

BATCH_SIZE = 10  # Jobs per Claude API call


@dataclass
class ScoredJob:
    job: Job
    score: int  # 0-10
    reason: str  # Why it matches or doesn't


SCORING_SYSTEM_PROMPT = """You are a job matching assistant. You will receive a candidate's job profile and a batch of job listings. Score each job's relevance to the candidate on a 0-10 scale.

CRITICAL DISTINCTIONS:
- "Help teams ADOPT AI tools" (what the candidate wants) ≠ "Build/train ML models" (what the candidate does NOT want)
- Roles requiring ML expertise (TensorFlow, PyTorch, model training, MLOps) are NOT a match even if the title says "AI Engineer"
- Roles requiring a Master's/PhD in AI, Computer Science, or Data Science are likely NOT a match
- The candidate is a senior WEB DEVELOPER who wants to champion AI tool adoption — not a data scientist or ML engineer

Scoring guide:
- 9-10: Near-perfect match — AI enablement/tooling/adoption role focused on helping dev teams use AI tools (Copilot, Claude, Cursor, MCP). Near Utrecht. Senior level. No ML expertise required.
- 7-8: Strong match — DevEx/platform eng with AI tool focus, or AI strategy/coaching role. Good location. Doesn't require ML/data science background.
- 5-6: Partial match — some AI enablement aspects but also requires significant ML knowledge, or right concept but wrong location
- 3-4: Weak match — AI-related but primarily about building ML models, data pipelines, or requires deep Python/ML stack experience
- 0-2: No match — pure ML/data science, wrong seniority, too far away, or clearly a different domain

ALSO: Score 0 if the listing appears to be CLOSED, EXPIRED, or UNAVAILABLE. Look for phrases like "no longer available", "vacature is vervuld", "not accepting applications", "expired", "gesloten", "verlopen".

Be strict. Most jobs should score below 5. The candidate's sweet spot is enabling/coaching/championing AI adoption in dev teams — not building AI systems.

Respond ONLY with a JSON array. No markdown, no backticks, no explanation outside the JSON.
Each element: {"index": 0, "score": 7, "reason": "Brief explanation"}
"""


def analyze_jobs(jobs: list[Job]) -> list[ScoredJob]:
    """Score a list of jobs against the configured profile using Claude."""
    if not config.ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEY not set — skipping analysis")
        return [ScoredJob(job=j, score=5, reason="No API key; unscored") for j in jobs]

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    scored: list[ScoredJob] = []

    # Process in batches
    for i in range(0, len(jobs), BATCH_SIZE):
        batch = jobs[i : i + BATCH_SIZE]
        batch_results = _score_batch(client, batch)
        scored.extend(batch_results)

    return scored


def _score_batch(client: anthropic.Anthropic, jobs: list[Job]) -> list[ScoredJob]:
    """Score a single batch of jobs."""
    # Build the job listings text
    listings = []
    for idx, job in enumerate(jobs):
        listings.append(
            f"[{idx}] Title: {job.title}\n"
            f"    Company: {job.company or 'Unknown'}\n"
            f"    Location: {job.location or 'Unknown'}\n"
            f"    Snippet: {job.snippet[:300]}\n"
            f"    URL: {job.url}"
        )

    user_message = (
        f"## Candidate Profile\n{config.JOB_PROFILE}\n\n"
        f"## Job Listings to Score\n" + "\n\n".join(listings)
    )

    try:
        response = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=1024,
            system=SCORING_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        response_text = response.content[0].text.strip()

        # Clean potential markdown fences
        if response_text.startswith("```"):
            response_text = response_text.split("\n", 1)[-1]
        if response_text.endswith("```"):
            response_text = response_text.rsplit("```", 1)[0]
        response_text = response_text.strip()

        scores = json.loads(response_text)
        score_map = {s["index"]: s for s in scores}

        results = []
        for idx, job in enumerate(jobs):
            if idx in score_map:
                results.append(
                    ScoredJob(
                        job=job,
                        score=score_map[idx]["score"],
                        reason=score_map[idx].get("reason", ""),
                    )
                )
            else:
                results.append(ScoredJob(job=job, score=0, reason="Not scored by Claude"))

        return results

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Claude response as JSON: {e}")
        logger.debug(f"Raw response: {response_text[:500]}")
        return [ScoredJob(job=j, score=0, reason="Parse error") for j in jobs]

    except anthropic.APIError as e:
        logger.error(f"Claude API error: {e}")
        return [ScoredJob(job=j, score=0, reason="API error") for j in jobs]
