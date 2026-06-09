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


def _problem_for(m: MetricResult) -> str:
    """A compact numeric problem line derived from the metric's justification."""
    return f"[{m.key}] {m.justification}"


def build_problems(metrics: list[MetricResult], t_page: float) -> list[str]:
    """Collect problems from metrics that caused (or contributed to) a review.

    Hard-fail metrics first (in priority order), then non-hard metrics whose
    status is 'bad' or whose score is clearly below the page threshold.
    """
    by_key = {m.key: m for m in metrics}
    problems: list[str] = []
    seen: set[str] = set()

    # 1) hard fails, in priority order
    for key in _PRIORITY:
        m = by_key.get(key)
        if m and m.hard_fail and m.applicable:
            problems.append(_problem_for(m))
            seen.add(key)

    # 2) bad-status / clearly-failing metrics
    for key in _PRIORITY:
        if key in seen:
            continue
        m = by_key.get(key)
        if not m or not m.applicable:
            continue
        if m.status == "bad" or m.score < min(t_page, 60.0):
            problems.append(_problem_for(m))
            seen.add(key)

    return problems


def top_problem(problems: list[str]) -> str:
    return problems[0] if problems else "—"
