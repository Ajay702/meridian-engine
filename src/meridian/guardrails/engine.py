"""Guardrail sub-engine.

Input stage:  size limits → prompt-injection scoring → PII redaction.
Output stage: PII redaction (the model may echo PII from retrieved context) and
              citation validation for grounded answers.

Each check returns a structured `Verdict` (allow / redact / block + reasons) that
is logged, counted and attached to the trace, so policy decisions are auditable.
Checks are pluggable: anything implementing `Check` can be appended.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from meridian.core.config import GuardrailConfig


class Action(StrEnum):
    ALLOW = "allow"
    REDACT = "redact"
    BLOCK = "block"


@dataclass
class Verdict:
    action: Action = Action.ALLOW
    text: str = ""
    reasons: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)

    def merge(self, other: Verdict) -> Verdict:
        rank = {Action.ALLOW: 0, Action.REDACT: 1, Action.BLOCK: 2}
        action = max(self.action, other.action, key=rank.__getitem__)
        return Verdict(action, other.text, self.reasons + other.reasons,
                       {**self.scores, **other.scores})


class Check(Protocol):
    name: str

    def __call__(self, text: str) -> Verdict: ...


# --------------------------------------------------------------------------- PII
def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return total % 10 == 0


class PIIRedactor:
    name = "pii"
    PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
        ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
        ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
        ("PHONE", re.compile(
            r"(?<!\w)(?:\+?\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}\b")),
        ("IPV4", re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")),
        ("API_KEY", re.compile(
            r"\b(?:sk|pk|rk)[-_](?:live|test|ant|proj)?[-_]?[A-Za-z0-9]{16,}\b")),
    )

    def __call__(self, text: str) -> Verdict:
        found: list[str] = []

        def sub(label: str):
            def repl(m: re.Match[str]) -> str:
                if label == "CARD":
                    digits = re.sub(r"\D", "", m.group())
                    if not (13 <= len(digits) <= 19 and _luhn_ok(digits)):
                        return m.group()
                found.append(label)
                return f"[{label}]"
            return repl

        out = text
        for label, pat in self.PATTERNS:
            out = pat.sub(sub(label), out)
        if not found:
            return Verdict(Action.ALLOW, text)
        return Verdict(Action.REDACT, out, [f"pii:{t.lower()}" for t in sorted(set(found))])


# --------------------------------------------------------------------------- injection
class InjectionDetector:
    """Weighted signal scoring for prompt-injection / jailbreak attempts.

    Heuristics are a first line of defence, cheap enough to run on every request;
    the score is exposed so a classifier model can be layered on for the grey zone.
    """

    name = "injection"
    SIGNALS: tuple[tuple[float, re.Pattern[str]], ...] = (
        (0.6, re.compile(r"\b(ignore|disregard|forget)\b.{0,30}\b(previous|prior|above|all)\b"
                         r".{0,20}\b(instructions?|rules|prompts?|context)\b", re.I | re.S)),
        (0.5, re.compile(r"\b(reveal|print|show|repeat|leak)\b.{0,30}\b(system prompt|hidden"
                         r" instructions?|your instructions)\b", re.I | re.S)),
        (0.4, re.compile(r"\byou are (now )?(DAN|in developer mode|unrestricted|jailbroken)\b",
                         re.I)),
        (0.3, re.compile(r"\b(act|pretend|roleplay) as\b.{0,40}\b(no|without) (rules|limits|"
                         r"restrictions|filters)\b", re.I | re.S)),
        (0.3, re.compile(r"</?(system|assistant|im_start|im_end)>|\[/?INST\]", re.I)),
        (0.2, re.compile(r"\b(base64|rot13)\b.{0,20}\b(decode|execute|follow)\b", re.I | re.S)),
    )

    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold

    def score(self, text: str) -> tuple[float, list[str]]:
        total, hits = 0.0, []
        for weight, pat in self.SIGNALS:
            if pat.search(text):
                total += weight
                hits.append(pat.pattern[:40])
        return min(1.0, total), hits

    def __call__(self, text: str) -> Verdict:
        s, _ = self.score(text)
        if s >= self.threshold:
            return Verdict(Action.BLOCK, text, ["prompt_injection"], {"injection": s})
        return Verdict(Action.ALLOW, text, [], {"injection": s})


class LengthLimit:
    name = "length"

    def __init__(self, max_chars: int) -> None:
        self.max_chars = max_chars

    def __call__(self, text: str) -> Verdict:
        if len(text) > self.max_chars:
            return Verdict(Action.BLOCK, text, [f"input_too_long:{len(text)}>{self.max_chars}"])
        return Verdict(Action.ALLOW, text)


_CITE = re.compile(r"\[(\d+)\]")


def validate_citations(answer: str, n_sources: int) -> list[str]:
    """Return problems with an answer's [n] citations (out-of-range or missing)."""
    cited = {int(c) for c in _CITE.findall(answer)}
    problems = [f"citation_out_of_range:{c}" for c in sorted(cited) if not 1 <= c <= n_sources]
    if n_sources and not cited and "don't have enough information" not in answer:
        problems.append("uncited_answer")
    return problems


class GuardrailEngine:
    def __init__(self, config: GuardrailConfig | None = None) -> None:
        self.config = config or GuardrailConfig()
        self.input_checks: list[Check] = [LengthLimit(self.config.max_input_chars)]
        if self.config.block_injection:
            self.input_checks.append(InjectionDetector(self.config.injection_threshold))
        self.output_checks: list[Check] = []
        if self.config.redact_pii:
            self.input_checks.append(PIIRedactor())
            self.output_checks.append(PIIRedactor())

    @staticmethod
    def _run(checks: list[Check], text: str) -> Verdict:
        verdict = Verdict(Action.ALLOW, text)
        for check in checks:
            v = check(verdict.text)
            verdict = verdict.merge(v)
            if verdict.action is Action.BLOCK:
                break
        return verdict

    def check_input(self, text: str) -> Verdict:
        return self._run(self.input_checks, text)

    def check_output(self, text: str) -> Verdict:
        return self._run(self.output_checks, text)
