"""Base types and utilities for job scrapers."""

from dataclasses import dataclass, field
from datetime import datetime
import hashlib


@dataclass
class Job:
    title: str
    company: str
    location: str
    url: str
    snippet: str
    source: str
    date_found: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def id(self) -> str:
        """Generate a stable ID from URL (primary) or title+company hash."""
        if self.url:
            return hashlib.md5(self.url.encode()).hexdigest()
        raw = f"{self.title}|{self.company}|{self.location}".lower()
        return hashlib.md5(raw.encode()).hexdigest()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "company": self.company,
            "location": self.location,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "date_found": self.date_found,
        }
