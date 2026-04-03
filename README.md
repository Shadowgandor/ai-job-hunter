# 🤖 AI Job Hunter

Automated daily job search for AI Native Engineer roles in the Netherlands. Runs on GitHub Actions, scores relevance with Claude, notifies via Telegram.

## Architecture

```
GitHub Actions (daily cron)
    ↓
┌─────────────────────────────┐
│  Scrapers                   │
│  • Indeed.nl RSS feeds      │
│  • DuckDuckGo web search    │
│    (LinkedIn, Glassdoor,    │
│     Magnet.me, etc.)        │
└────────────┬────────────────┘
             ↓
     Deduplicate against
     seen_jobs.json
             ↓
┌─────────────────────────────┐
│  Claude Haiku               │
│  Score each job 0-10        │
│  against your profile       │
└────────────┬────────────────┘
             ↓
     Filter score ≥ 7
             ↓
┌─────────────────────────────┐
│  Telegram Bot               │
│  Send matching jobs         │
└─────────────────────────────┘
```

## Setup

### 1. Create a Telegram Bot

1. Open Telegram and message [@BotFather](https://t.me/BotFather)
2. Send `/newbot` and follow the prompts
3. Copy the **bot token** (looks like `123456789:ABCdefGhIjKlMnOpQrStUvWxYz`)
4. **Message your new bot** (send it any message like "hello")
5. Visit `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
6. Find your `chat_id` in the response JSON (under `result[0].message.chat.id`)

### 2. Get a Claude API Key

1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Create an API key
3. Add some credits (this uses Haiku — extremely cheap, ~$0.01-0.05/day)

### 3. Set Up the GitHub Repository

```bash
# Clone or create repo
git clone <your-repo-url>
cd ai-job-hunter

# Or initialize fresh
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-repo-url>
git push -u origin main
```

### 4. Add GitHub Secrets

Go to your repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

Add these three secrets:

| Secret Name | Value |
|---|---|
| `ANTHROPIC_API_KEY` | Your Claude API key |
| `TELEGRAM_BOT_TOKEN` | Your Telegram bot token |
| `TELEGRAM_CHAT_ID` | Your Telegram chat ID |

### 5. Test Locally (Optional)

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export ANTHROPIC_API_KEY="sk-ant-..."
export TELEGRAM_BOT_TOKEN="123456789:ABC..."
export TELEGRAM_CHAT_ID="your-chat-id"

# Test Telegram connection
python main.py --test-tg

# Do a dry run (scrape + analyze, no notifications)
python main.py --dry-run

# Full run
python main.py
```

### 6. Enable the Workflow

The GitHub Action runs automatically at **08:00 CET daily**. You can also trigger it manually:

Repo → **Actions** → **Daily AI Job Search** → **Run workflow**

## Customization

### Edit search queries

In `config.py`, modify `SEARCH_QUERIES` and `SEARCH_QUERIES_NL` to adjust what gets searched.

### Change scoring threshold

In `config.py`, adjust `RELEVANCE_THRESHOLD` (default: 7). Lower it to see more results, raise it for only the strongest matches.

### Edit your job profile

In `config.py`, modify `JOB_PROFILE` to fine-tune what Claude considers a match.

### Add a new scraper

1. Create a new file in `scrapers/` that returns `list[Job]`
2. Import and call it in `main.py`'s `run()` function

## Cost

- **GitHub Actions**: Free (runs ~3 min/day, well within free tier)
- **Claude Haiku**: ~$0.01-0.05/day depending on job volume
- **Telegram**: Free
- **Total**: Essentially free — maybe €1-2/month for Claude API

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point and orchestration |
| `config.py` | All configuration (queries, profile, API keys) |
| `analyzer.py` | Claude-based relevance scoring |
| `notifier.py` | Telegram notifications |
| `scrapers/indeed.py` | Indeed.nl RSS scraper |
| `scrapers/duckduckgo.py` | DuckDuckGo multi-site search |
| `seen_jobs.json` | State file tracking seen jobs (auto-updated) |
| `.github/workflows/daily-search.yml` | GitHub Actions schedule |
