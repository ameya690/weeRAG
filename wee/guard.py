# wee/guard.py
from __future__ import annotations
from typing import Dict, Any, List
import re

EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b")
CREDIT = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
PII_HINTS = [EMAIL, PHONE, CREDIT]

# Very small bad-word list (extensible)
BAD_WORDS = {"idiot","stupid","dumb"}

# Prompt-injection heuristics
INJECTION_PATTERNS = [
    r"ignore (the )?(previous|above) instructions",
    r"disregard your rules",
    r"reveal your (system|hidden) prompt",
    r"act as (an? )?(administrator|developer|root)",
    r"self-destruct|format drive|exfiltrate",
]

class Guard:
    """
    Lightweight guardrail checks: PII, profanity, injection-y phrases, link allowlist.
    """
    def __init__(self, allowed_domains: List[str] = None):
        self.allowed = set(allowed_domains or [])
        self.injection_rx = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

    def check_pii(self, text: str) -> List[str]:
        issues = []
        for rx in PII_HINTS:
            for _ in rx.finditer(text):
                if rx is EMAIL: issues.append("email")
                elif rx is PHONE: issues.append("phone")
                else: issues.append("credit_card_like")
        return list(sorted(set(issues)))

    def check_profanity(self, text: str) -> List[str]:
        toks = set(w.lower() for w in re.findall(r"[A-Za-z']+", text))
        return sorted(list(BAD_WORDS & toks))

    def check_injection(self, text: str) -> List[str]:
        hits = []
        for rx in self.injection_rx:
            if rx.search(text):
                hits.append(rx.pattern)
        return hits

    def check_links(self, text: str) -> List[str]:
        if not self.allowed:
            return []
        hits = []
        for url in re.findall(r"https?://[^\s)]+", text):
            m = re.search(r"https?://([^/]+)/?", url)
            if m:
                dom = m.group(1).lower()
                if not any(dom.endswith(ad.lower()) for ad in self.allowed):
                    hits.append(url)
        return hits

    def score(self, text: str) -> float:
        """
        Heuristic risk score in [0,1].
        """
        s = 0.0
        if self.check_pii(text): s += 0.4
        if self.check_injection(text): s += 0.4
        if self.check_profanity(text): s += 0.1
        if self.check_links(text): s += 0.1
        return min(1.0, s)

    def sanitize(self, text: str) -> str:
        t = EMAIL.sub("[email]", text)
        t = PHONE.sub("[phone]", t)
        t = CREDIT.sub("[card]", t)
        return t

    def check(self, text: str) -> Dict[str, Any]:
        return {
            "pii": self.check_pii(text),
            "profanity": self.check_profanity(text),
            "injection": self.check_injection(text),
            "unallowed_links": self.check_links(text),
            "risk": self.score(text),
        }
