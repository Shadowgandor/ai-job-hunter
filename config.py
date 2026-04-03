"""
Configuration for the AI Job Hunter.
All settings can be overridden via environment variables.
"""

import os

# ---------- Job Search Configuration ----------

SEARCH_QUERIES = [
    "AI enablement engineer",
    "AI tooling engineer",
    "AI adoption specialist",
    "developer experience AI",
    "AI way of working",
    "AI integration engineer",
    "AI specialist SDLC",
    "GenAI engineer tooling",
    "AI coach developer",
    "platform engineer AI tools",
    "AI transformation engineer",
]

# Dutch-language queries (many NL roles are posted in Dutch)
SEARCH_QUERIES_NL = [
    "AI specialist developer teams",
    "AI adoptie engineer",
    "AI tooling ontwikkelaar",
    "generatieve AI engineer",
    "AI adviseur development",
]

LOCATIONS = ["utrecht", "amsterdam", "nieuwegein", "amersfoort"]
RADIUS_KM = 40
COUNTRY = "nl"

# ---------- Job Profile for Claude Matching ----------

JOB_PROFILE = """
I'm looking for a role as an "AI Native Engineer" — though the exact title varies wildly.

Core of what I'm looking for:
- Implement AI practices in the SDLC (software development lifecycle)
- Help development teams adopt AI tooling (GitHub Copilot, Claude Code, Cursor, MCP servers, etc.)
- Embed AI tools and best practices into engineering workflows
- Stay up to date on latest AI features, updates, and capabilities
- Translate AI developments into internal opportunities and concrete improvements
- Champion AI adoption within engineering organizations
- Measure and improve developer productivity through AI

Also acceptable if the role includes:
- Working as part of a platform/DevEx team focused on AI tooling
- Solo AI champion or specialist within an engineering department
- Consulting/advisory roles that embed AI practices at client organizations
- Strategic AI coordination with hands-on technical aspects
- Workshops, demos, guidelines creation for AI tool usage
- Security/compliance considerations for AI tools in enterprise settings

My background:
- 13 years professional web development experience (Angular, TypeScript primary stack)
- Background in native iOS/Android development (Swift, Kotlin)
- Currently working in enterprise financial services (compliance-heavy, banking)
- Hands-on experience advocating for AI tooling adoption, writing proposals to CISO/leadership
- Practical experience with GitHub Copilot, Claude Code, MCP servers, AI coding agents
- Built developer productivity measurement frameworks with ROI calculations
- Experience with accessibility (EAA/WCAG), CVE triage, Angular/NX monorepos, Azure DevOps

Location requirement:
- Based in Nieuwegein, Netherlands
- Prefer roles within 40km of Utrecht (Utrecht, Amsterdam, Amersfoort, Hilversum, etc.)
- Remote/hybrid is fine too

What does NOT match:
- Pure ML/Data Science roles (training models, building pipelines)
- Academic/PhD research positions
- Junior/intern/traineeship positions
- Roles requiring 5+ years of Python/ML-specific experience
- Roles focused solely on building AI products (vs. enabling teams to use AI)
- Roles far from Utrecht (Eindhoven, Groningen, Enschede, etc.) unless fully remote
"""

RELEVANCE_THRESHOLD = 7  # Minimum Claude score (0-10) to notify

# ---------- API Configuration ----------

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ---------- State ----------

STATE_FILE = os.environ.get("STATE_FILE", "seen_jobs.json")
MAX_JOBS_PER_RUN = int(os.environ.get("MAX_JOBS_PER_RUN", "50"))
