"""AI Security Layer (spec Part 20).

Deterministic, regex/heuristic-based checks - not another LLM call, per
the "verification should use deterministic checks whenever possible"
principle (Part 22). This is a real, testable first line of defense,
not a claim of complete jailbreak-proofing: heuristic detectors have
false negatives against novel phrasing, which is stated plainly in the
registry rather than glossed over.
"""

import re
from dataclasses import dataclass, field

# ----------------------------------------------------------------------
# Prompt injection / jailbreak heuristics
# ----------------------------------------------------------------------

INJECTION_PATTERNS = [
    (re.compile(r"ignore (all|the|any) (previous|prior|above) instructions?", re.I),
     "instruction override attempt"),
    (re.compile(r"disregard (all|the|any) (previous|prior|above)", re.I),
     "instruction override attempt"),
    (re.compile(r"you are now (in )?(developer|debug|admin|dan|unrestricted) mode", re.I),
     "role/mode override attempt"),
    (re.compile(r"pretend (you|to) (are|be) (an? )?(unfiltered|unrestricted|jailbroken)", re.I),
     "jailbreak persona request"),
    (re.compile(r"reveal (your|the) (system prompt|instructions|hidden prompt)", re.I),
     "system prompt exfiltration attempt"),
    (re.compile(r"act as (if )?(you have )?no (restrictions|rules|limits|guidelines)", re.I),
     "restriction-bypass request"),
    (re.compile(r"\bDAN\b.{0,20}\b(mode|prompt)\b", re.I), "known jailbreak persona (DAN)"),
    (re.compile(r"bypass (your |the )?(safety|content) (filter|policy|guidelines)", re.I),
     "safety-bypass request"),
]

# ----------------------------------------------------------------------
# PII patterns (deliberately conservative - flags likely PII, not proof)
# ----------------------------------------------------------------------

PII_PATTERNS = [
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("ssn_like", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit_card_like", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("phone_like", re.compile(r"\b(?:\+?\d{1,3}[ -]?)?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}\b")),
]


@dataclass
class SecurityFinding:
    category: str          # "prompt_injection" | "pii"
    detail: str
    span: str = ""


@dataclass
class SecurityReport:
    blocked: bool
    findings: list = field(default_factory=list)

    def to_dict(self):
        return {"blocked": self.blocked,
                "findings": [f.__dict__ for f in self.findings]}


def check_prompt_injection(text: str) -> list:
    findings = []
    for pattern, label in INJECTION_PATTERNS:
        m = pattern.search(text or "")
        if m:
            findings.append(SecurityFinding("prompt_injection", label, m.group(0)))
    return findings


def check_pii(text: str) -> list:
    findings = []
    for label, pattern in PII_PATTERNS:
        for m in pattern.finditer(text or ""):
            # Mask the match in the finding so the report itself doesn't
            # persist raw PII.
            masked = m.group(0)[:2] + "*" * max(0, len(m.group(0)) - 4) + m.group(0)[-2:]
            findings.append(SecurityFinding("pii", label, masked))
    return findings


def check_input(text: str) -> SecurityReport:
    """Input-side security check (Part 20 pipeline: Input -> Security Check)."""
    injections = check_prompt_injection(text)
    pii = check_pii(text)
    # Prompt injection blocks the request; PII in a question is logged but
    # not blocking (a user is allowed to ask about their own data).
    return SecurityReport(blocked=bool(injections), findings=injections + pii)


def check_output(text: str) -> SecurityReport:
    """Output-side security check (Part 20 pipeline: Output -> Security Check).
    Flags PII the model may have echoed or fabricated in its response."""
    pii = check_pii(text)
    return SecurityReport(blocked=False, findings=pii)


def redact_pii(text: str) -> str:
    """Return `text` with detected PII spans replaced by a redaction marker."""
    result = text or ""
    for label, pattern in PII_PATTERNS:
        result = pattern.sub(f"[REDACTED_{label.upper()}]", result)
    return result
