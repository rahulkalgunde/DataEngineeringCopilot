"""Deterministic robustness probes derived from golden QA rows (RGB-inspired).

noise probe      : gold contexts + 2 cross-source distractors; expect an answer.
rejection probe  : gold evidence removed; expect a refusal.
"""

from __future__ import annotations

_REFUSAL_MARKERS = (
    "insufficient",
    "cannot answer",
    "do not have",
    "does not contain",
    "not enough information",
    # JSON-structured refusals (structured output with missing_info/answer: null)
    '"missing_info": true',
    '"answer": null',
    '"answer": null,',
    '"missing_info": true',
    '"answer": "null"',
)


def looks_like_refusal(answer: str) -> bool:
    a = (answer or "").lower()
    return any(m in a for m in _REFUSAL_MARKERS)


def build_probes(qa_rows: list[dict]) -> list[dict]:
    others_by_row: dict[str, list[str]] = {}
    ids = [r.get("id", f"row{i}") for i, r in enumerate(qa_rows)]
    for i, _r in enumerate(qa_rows):
        pool: list[str] = []
        for j, o in enumerate(qa_rows):
            if j == i:
                continue
            pool.extend(o.get("contexts", []))
        others_by_row[ids[i]] = pool

    probes: list[dict] = []
    for i, r in enumerate(qa_rows):
        rid = ids[i]
        others = others_by_row[rid]
        distractors = (
            [others[(i * 2 + k) % max(1, len(others))] for k in range(2)]
            if others
            else ["no external context available"]
        )
        probes.append(
            {
                "id": f"{rid}-noise",
                "question": r["question"],
                "contexts": list(r.get("contexts", [])) + distractors,
                "ground_truth": r.get("ground_truth", ""),
                "probe": "noise",
                "expect_refusal": False,
            }
        )
        replacements = distractors if others else ["no external context available"]
        probes.append(
            {
                "id": f"{rid}-rejection",
                "question": r["question"],
                "contexts": replacements[:2],
                "ground_truth": r.get("ground_truth", ""),
                "probe": "rejection",
                "expect_refusal": True,
            }
        )
    return probes
