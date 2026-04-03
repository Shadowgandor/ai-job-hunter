"""
Send job match notifications via Telegram Bot API.

Setup:
1. Message @BotFather on Telegram to create a bot and get your token
2. Message your bot, then visit:
   https://api.telegram.org/bot<TOKEN>/getUpdates
   to find your chat_id
3. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID as env vars
"""

import logging
from datetime import datetime

import requests

import config
from analyzer import ScoredJob

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


def notify_matches(scored_jobs: list[ScoredJob]) -> None:
    """Send Telegram notifications for matching jobs."""
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        logger.warning("Telegram not configured — printing to console instead")
        _print_matches(scored_jobs)
        return

    # Filter to relevant matches
    matches = [sj for sj in scored_jobs if sj.score >= config.RELEVANCE_THRESHOLD]

    if not matches:
        _send_message("🔍 *AI Job Hunter — no new matches today.*\n"
                      f"Scanned {len(scored_jobs)} new listings.")
        logger.info("No matches above threshold — sent summary.")
        return

    # Sort by score descending
    matches.sort(key=lambda sj: sj.score, reverse=True)

    # Header message
    header = (
        f"🎯 *AI Job Hunter — {len(matches)} match{'es' if len(matches) != 1 else ''}!*\n"
        f"_{datetime.now().strftime('%A %d %B %Y')}_\n"
        f"Scanned {len(scored_jobs)} new listings total."
    )
    _send_message(header)

    # Individual job messages
    for sj in matches:
        score_emoji = "🟢" if sj.score >= 9 else "🟡" if sj.score >= 7 else "🟠"
        company = sj.job.company or "Unknown company"
        location = sj.job.location or "Unknown location"

        msg = (
            f"{score_emoji} *Score: {sj.score}/10*\n"
            f"*{_escape_md(sj.job.title)}*\n"
            f"🏢 {_escape_md(company)} · 📍 {_escape_md(location)}\n"
            f"💡 _{_escape_md(sj.reason)}_\n"
            f"[Open →]({sj.job.url})"
        )
        _send_message(msg)

    logger.info(f"Sent {len(matches)} match notifications to Telegram")


def send_error_notification(error: str) -> None:
    """Send an error notification."""
    if config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID:
        _send_message(f"⚠️ *AI Job Hunter error:*\n`{_escape_md(error[:200])}`")


def _send_message(text: str) -> bool:
    """Send a single Telegram message."""
    url = TELEGRAM_API.format(token=config.TELEGRAM_BOT_TOKEN)
    try:
        resp = requests.post(
            url,
            json={
                "chat_id": config.TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
        if not resp.ok:
            logger.error(f"Telegram API error: {resp.status_code} — {resp.text[:200]}")
            return False
        return True
    except requests.RequestException as e:
        logger.error(f"Telegram request failed: {e}")
        return False


def _escape_md(text: str) -> str:
    """Escape Markdown special characters for Telegram."""
    for char in ["_", "*", "[", "]", "(", ")", "~", "`", ">", "#", "+", "-", "=", "|", "{", "}", ".", "!"]:
        text = text.replace(char, f"\\{char}")
    return text


def _print_matches(scored_jobs: list[ScoredJob]) -> None:
    """Fallback: print matches to console."""
    matches = [sj for sj in scored_jobs if sj.score >= config.RELEVANCE_THRESHOLD]
    if not matches:
        print(f"\n✅ No matches above threshold ({config.RELEVANCE_THRESHOLD}) "
              f"in {len(scored_jobs)} jobs scanned.\n")
        return

    matches.sort(key=lambda sj: sj.score, reverse=True)
    print(f"\n🎯 {len(matches)} match(es) found!\n")
    for sj in matches:
        print(f"  [{sj.score}/10] {sj.job.title}")
        print(f"         {sj.job.company or '?'} — {sj.job.location or '?'}")
        print(f"         {sj.reason}")
        print(f"         {sj.job.url}")
        print()
