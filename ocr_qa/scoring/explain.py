"""Deterministic, templated problem statements (Sec 5).

``build_problems`` turns the metrics that pushed a page to 'review' into concise,
NUMERIC, human-readable problem lines. Never an LLM — same inputs always yield
the same wording.
"""

from __future__ import annotations

from ocr_qa.models import MetricResult, PageReport

# Order in which problems are surfaced (most decisive first).
_PRIORITY = ["IFR", "CED", "XSA", "LVR", "BCC", "OOV", "HMW", "TSI", "BGC",
             "PPL", "TTR", "WSA", "PSW", "SFC"]


def _problem_for(m: MetricResult, prefix: str = "") -> str:
    """A compact numeric problem line derived from the metric's justification."""
    return f"[{m.key}] {prefix}{m.justification}"


def build_problems(metrics: list[MetricResult], t_page: float) -> list[str]:
    """Collect problems from metrics that caused (or contributed to) a review.

    DEFINITE (hard) failures first, then the SOFT signals that failed
    (supporting evidence; only decisive in aggregate). Tagged so a reader can
    tell a definite fault from a soft signal.
    """
    by_key = {m.key: m for m in metrics}
    problems: list[str] = []
    seen: set[str] = set()

    # 1) DEFINITE / hard fails, in priority order
    for key in _PRIORITY:
        m = by_key.get(key)
        if m and m.hard_fail and m.applicable:
            problems.append(_problem_for(m, "DEFINITE: "))
            seen.add(key)

    # 2) SOFT signals that failed (status 'bad', text-rich page)
    for key in _PRIORITY:
        if key in seen:
            continue
        m = by_key.get(key)
        if not m or not m.applicable:
            continue
        if m.tier == "soft" and m.status == "bad" and m.reliability == "high":
            problems.append(_problem_for(m, "soft: "))
            seen.add(key)

    # 3) any remaining clearly-failing (e.g. low-PQS backstop context)
    for key in _PRIORITY:
        if key in seen:
            continue
        m = by_key.get(key)
        if m and m.applicable and (m.status == "bad" or m.score < min(t_page, 50.0)):
            problems.append(_problem_for(m))
            seen.add(key)

    return problems


def top_problem(problems: list[str]) -> str:
    return problems[0] if problems else "—"
