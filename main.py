"""
AI Job Hunter — Daily automated job search for AI Native Engineer roles.

Usage:
    python main.py              # Full run: scrape → analyze → notify
    python main.py --dry-run    # Scrape only, no notifications
    python main.py --test-tg    # Send a test Telegram message
"""

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import config
from scrapers import Job
from scrapers.indeed import scrape_indeed
from scrapers.duckduckgo import scrape_duckduckgo
from analyzer import analyze_jobs, ScoredJob
from notifier import notify_matches, send_error_notification

try:
    from embeddings import embed_jobs
    from classifier import classify_ai_native
    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False
    logging.getLogger("ai-job-hunter").info(
        "sentence-transformers not installed — ML scoring disabled"
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ai-job-hunter")


# ---------- State Management ----------

def load_seen_jobs() -> dict[str, dict]:
    """Load previously seen job IDs with metadata."""
    path = Path(config.STATE_FILE)
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Could not load state file: {e}")
        return {}


def save_seen_jobs(seen: dict[str, dict]) -> None:
    """Save seen jobs to state file, pruning entries older than 60 days."""
    cutoff = (datetime.now() - timedelta(days=60)).isoformat()
    pruned = {
        job_id: meta
        for job_id, meta in seen.items()
        if meta.get("date", "") > cutoff
    }

    with open(config.STATE_FILE, "w") as f:
        json.dump(pruned, f, indent=2)

    logger.info(f"State saved: {len(pruned)} jobs tracked (pruned {len(seen) - len(pruned)} old entries)")


def filter_new_jobs(jobs: list[Job], seen: dict[str, dict]) -> list[Job]:
    """Return only jobs we haven't seen before."""
    new = [j for j in jobs if j.id not in seen]
    logger.info(f"Filtering: {len(jobs)} total → {len(new)} new")
    return new


def mark_as_seen(jobs: list[Job], seen: dict[str, dict]) -> None:
    """Add jobs to the seen set."""
    for job in jobs:
        seen[job.id] = {
            "title": job.title,
            "company": job.company,
            "url": job.url,
            "date": job.date_found,
        }


# ---------- Filters ----------

CLOSED_INDICATORS = [
    "no longer available", "not accepting applications",
    "vacature is vervuld", "niet meer beschikbaar",
    "deze vacature is gesloten", "verlopen", "expired",
    "helaas al vervuld", "not found", "pagina bestaat niet",
]


def _filter_closed_listings(jobs: list[Job]) -> list[Job]:
    """Remove listings that appear closed/expired based on title or snippet."""
    open_jobs = []
    for job in jobs:
        text = f"{job.title} {job.snippet}".lower()
        if any(indicator in text for indicator in CLOSED_INDICATORS):
            logger.info(f"Filtered closed listing: {job.title}")
            continue
        open_jobs.append(job)

    filtered_count = len(jobs) - len(open_jobs)
    if filtered_count:
        logger.info(f"Removed {filtered_count} closed/expired listings")
    return open_jobs


# ---------- Main Pipeline ----------

def run(dry_run: bool = False) -> None:
    """Main job hunting pipeline."""
    logger.info("=" * 60)
    logger.info("AI Job Hunter — starting daily scan")
    logger.info("=" * 60)

    # 1. Load state
    seen = load_seen_jobs()
    logger.info(f"Loaded {len(seen)} previously seen jobs")

    # 2. Scrape from all sources
    all_queries = config.SEARCH_QUERIES + config.SEARCH_QUERIES_NL
    all_jobs: list[Job] = []

    # Indeed.nl RSS
    try:
        indeed_jobs = scrape_indeed(
            queries=config.SEARCH_QUERIES[:6],  # Top queries only for Indeed
            locations=config.LOCATIONS[:2],      # Utrecht + Amsterdam
            radius_km=config.RADIUS_KM,
        )
        all_jobs.extend(indeed_jobs)
    except Exception as e:
        logger.error(f"Indeed scraper failed: {e}")

    # DuckDuckGo web search
    try:
        ddg_jobs = scrape_duckduckgo(
            queries=all_queries[:8],  # Limit to avoid rate limiting
            max_results_per_query=5,
        )
        all_jobs.extend(ddg_jobs)
    except Exception as e:
        logger.error(f"DuckDuckGo scraper failed: {e}")

    # 3. Deduplicate across sources
    unique_jobs: dict[str, Job] = {}
    for job in all_jobs:
        if job.id not in unique_jobs:
            unique_jobs[job.id] = job
    all_jobs = list(unique_jobs.values())
    logger.info(f"Total unique jobs from all sources: {len(all_jobs)}")

    # 4. Filter out previously seen jobs
    new_jobs = filter_new_jobs(all_jobs, seen)

    if not new_jobs:
        logger.info("No new jobs found. Done.")
        if not dry_run:
            notify_matches([])  # Send "no matches" summary
        return

    # Cap to avoid excessive API usage
    if len(new_jobs) > config.MAX_JOBS_PER_RUN:
        logger.info(f"Capping at {config.MAX_JOBS_PER_RUN} jobs (had {len(new_jobs)})")
        new_jobs = new_jobs[: config.MAX_JOBS_PER_RUN]

    # 4b. Pre-filter obviously closed/expired listings
    new_jobs = _filter_closed_listings(new_jobs)

    # 5. Analyze with Claude
    logger.info(f"Analyzing {len(new_jobs)} new jobs with Claude...")
    scored_jobs = analyze_jobs(new_jobs)

    # 5b. Enrich with ML scores (embedding similarity + AI-native classifier)
    if _ML_AVAILABLE:
        try:
            sims = embed_jobs(new_jobs)
            probs = classify_ai_native(new_jobs)
            for sj, sim, prob in zip(scored_jobs, sims, probs):
                sj.embed_sim = sim
                sj.ai_native_score = prob
        except Exception as e:
            logger.warning(f"ML scoring failed (non-fatal): {e}")

    # Log results
    for sj in sorted(scored_jobs, key=lambda s: s.score, reverse=True):
        ml_suffix = (
            f" | sim={sj.embed_sim:.2f} ai_native={sj.ai_native_score:.2f}"
            if _ML_AVAILABLE else ""
        )
        logger.info(
            f"  [{sj.score}/10] {sj.job.title} @ {sj.job.company or '?'}"
            f" — {sj.reason}{ml_suffix}"
        )

    # Save results for eval harness
    _save_last_run(scored_jobs, accumulate="--accumulate" in sys.argv)

    # 6. Notify
    if not dry_run:
        try:
            notify_matches(scored_jobs)
        except Exception as e:
            logger.error(f"Notification failed: {e}")
            send_error_notification(str(e))

    # 7. Mark all new jobs as seen (even low-scoring ones)
    mark_as_seen(new_jobs, seen)
    save_seen_jobs(seen)

    # Summary
    matches = [sj for sj in scored_jobs if sj.score >= config.RELEVANCE_THRESHOLD]
    logger.info(f"Done! {len(matches)} matches out of {len(new_jobs)} new jobs.")


def _save_last_run(scored_jobs: list[ScoredJob], accumulate: bool = False) -> None:
    """Persist scored jobs to scored_jobs.json for the eval harness.

    With accumulate=True, merges with any existing file (deduplicating by job id)
    so data builds up across runs without losing previous results.
    """
    path = Path("scored_jobs.json")
    existing: dict[str, dict] = {}
    if accumulate and path.exists():
        try:
            for item in json.loads(path.read_text()):
                existing[item["job"]["id"]] = item
        except (json.JSONDecodeError, KeyError):
            pass  # Corrupt file — start fresh

    new_entries = {sj.job.id: sj.to_dict() for sj in scored_jobs}
    merged = {**existing, **new_entries}  # New run wins on collision

    try:
        path.write_text(json.dumps(list(merged.values()), indent=2))
        added = len(new_entries)
        total = len(merged)
        logger.info(
            f"Saved {added} scored jobs → {path}"
            + (f" (total accumulated: {total})" if accumulate else "")
        )
    except IOError as e:
        logger.warning(f"Could not save scored_jobs.json: {e}")


def run_eval_mode() -> None:
    """Run the faithfulness eval harness on the last pipeline results."""
    from eval_harness import load_scored_jobs, run_eval, print_report

    path = "scored_jobs.json"
    if not Path(path).exists():
        print(f"No {path} found. Run the main pipeline first.")
        return

    cases = load_scored_jobs(path)
    print(f"Evaluating faithfulness of {len(cases)} scored jobs from {path}...")
    results = run_eval(cases)
    print_report(results)


def test_telegram() -> None:
    """Send a test message to verify Telegram setup."""
    from notifier import _send_message
    ok = _send_message("🤖 *AI Job Hunter* — test message\\! Your bot is working\\.")
    if ok:
        print("✅ Test message sent successfully!")
    else:
        print("❌ Failed to send test message. Check your TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")


# ---------- CLI ----------

if __name__ == "__main__":
    if "--test-tg" in sys.argv:
        test_telegram()
    elif "--eval" in sys.argv:
        run_eval_mode()
    elif "--dry-run" in sys.argv or "--accumulate" in sys.argv:
        run(dry_run="--dry-run" in sys.argv)
    else:
        try:
            run()
        except Exception as e:
            logger.exception(f"Fatal error: {e}")
            send_error_notification(str(e))
            sys.exit(1)
