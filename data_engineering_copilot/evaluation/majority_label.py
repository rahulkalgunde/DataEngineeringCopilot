"""Majority-vote LLM labeling for judge calibration rows.

Replaces manual labeling with a 3-annotator ensemble: each row is judged by
three independently pinned providers and the binary majority (2-of-3; ties
impossible) becomes the label. ``label_note`` records unanimity vs split so
downstream consumers know label confidence.

Honesty note: these are LLM labels, not human labels. ``dec
eval-judge-calibrate`` against them measures cross-model agreement, which is
necessary but NOT sufficient for the industry calibration gate — flip
``judge_cascade_enabled`` only after a human pass via ``make
label-calibration`` confirms kappa >= 0.60.
"""

from __future__ import annotations

import asyncio
import json
import pathlib

from data_engineering_copilot.evaluation.generation_eval import _judge_call_with_retry
from data_engineering_copilot.factory import build_llm_fallback_chain

DEFAULT_PATH = "tests/evaluation/golden/judge_calibration.jsonl"
# Free-tier providers that held up under batch load (groq/openrouter/zai 429'd).
DEFAULT_ANNOTATORS = ("cerebras", "sambanova", "mistral")


def majority_vote(votes: list[int]) -> tuple[int, str]:
    """Binary majority verdict. Returns (label, agreement) where agreement is
    'unanimous' (3-0) or 'majority' (2-1). Raises on empty/non-binary input."""
    if not votes or any(v not in (0, 1) for v in votes):
        raise ValueError("votes must be a non-empty list of 0/1 ints")
    label = 1 if sum(votes) * 2 > len(votes) else 0
    agreement = "unanimous" if len(set(votes)) == 1 else "majority"
    return label, agreement


def _faithfulness_prompt(row: dict) -> str:
    ctx = "\n\n---\n\n".join(row.get("contexts") or [])[:5000]
    return (
        "You are a strict faithfulness grader. The CONTEXT is the only truth.\n"
        f"CONTEXT:\n{ctx}\n\n"
        f"QUESTION:\n{row.get('question', '')}\n\n"
        f"ANSWER:\n{(row.get('answer') or '')[:1500]}\n"
        'Is the ANSWER fully supported by the CONTEXT? Output ONLY JSON: {"score": <0 or 1>}'
    )


async def label_row(row: dict, judges: list) -> dict:
    """Judge one row with every annotator; attach majority verdict."""
    prompt = _faithfulness_prompt(row)
    votes = []
    for j in judges:
        score = await _judge_call_with_retry(j, prompt, 0.0, 1.0)
        votes.append(1 if score >= 0.5 else 0)  # threshold, never int-truncate
    label, agreement = majority_vote(votes)
    row["human_faithfulness"] = label
    row["human_relevance"] = label
    row["needs_label"] = False
    row["label_note"] = f"llm_majority_3way_{agreement}"
    return row


async def label_dataset(
    path: str = DEFAULT_PATH,
    annotators: tuple[str, ...] = DEFAULT_ANNOTATORS,
    judges: list | None = None,
) -> dict:
    """Label all rows in place. Inject ``judges`` for tests; else build chains."""
    rows_path = pathlib.Path(path)
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if judges is None:
        from data_engineering_copilot.config.settings import settings

        judges = [build_llm_fallback_chain("evaluation", app_settings=settings, purpose_provider=p) for p in annotators]
    for i, row in enumerate(rows):
        await label_row(row, judges)
        if (i + 1) % 20 == 0:
            faithful = sum(r["human_faithfulness"] for r in rows[: i + 1])
            print(f"[{i + 1}/{len(rows)}] labeled ({faithful} faithful)", flush=True)
    rows_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    summary = {
        "rows": len(rows),
        "faithful": sum(r["human_faithfulness"] for r in rows),
        "unanimous": sum(1 for r in rows if r["label_note"].endswith("unanimous")),
    }
    print(f"DONE: {summary['rows']} rows, {summary['faithful']} faithful, {summary['unanimous']} unanimous")
    return summary


if __name__ == "__main__":
    asyncio.run(label_dataset())
