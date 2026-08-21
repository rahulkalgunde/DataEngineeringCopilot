"""Generation-layer evaluation with retrieval frozen.

This harness evaluates ONLY the generation layer: it supplies an immutable
``(question, gold_context)`` pair to the answer LLM and measures how well the
generated answer is grounded, relevant, and correct — independent of retrieval
or prompt-assembly quality.

Metrics (per the generation-layer plan; latency/throughput B8 is intentionally
excluded):
  * B5 Faithfulness   — fraction of claims backed by the frozen context (gate >= 0.85)
  * B6 Answer relevance — how directly the answer addresses the question (gate >= 0.80)
  * B7 LLM-as-judge rubric — 1-5 completeness/accuracy/tone vs gold (gate >= 4.0)

Judge bias mitigation: judge runs at the chain's near-zero temperature, a
*different-family* judge (the ``evaluation`` purpose) scores against the
generator, and the rubric score is averaged over ``n_trials`` (>1) to dampen
judge variance. No latency measurement is performed.
"""

from __future__ import annotations

import json
import logging
import re
import statistics
from dataclasses import asdict, dataclass, field

from data_engineering_copilot.config.settings import AppSettings
from data_engineering_copilot.factory import build_llm_fallback_chain
from data_engineering_copilot.services.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)

FAITHFULNESS_GATE = 0.85
RELEVANCE_GATE = 0.80
RUBRIC_GATE = 4.0

_FAITHFULNESS_PROMPT = """You are a strict faithfulness grader for a RAG system.
The GROUNDING CONTEXT is the ONLY allowed source of truth. Given the QUESTION
and the ANSWER, estimate the fraction of the answer's factual claims that are
directly supported by the context. A faithful answer invents nothing.
Ignore style, length, and tone.

QUESTION:
{question}

GROUNDING CONTEXT:
{context}

ANSWER:
{answer}

Output ONLY a JSON object: {{"score": <float 0.0-1.0>, "reason": "<short>"}}."""

_RELEVANCE_PROMPT = """You are grading answer relevance (topicality only).
Given the QUESTION and the ANSWER, score how directly and completely the answer
addresses the question. 1.0 = directly and completely addresses the question;
0.0 = off-topic, evasive, or ignores the question. Ignore factual correctness.

QUESTION:
{question}

ANSWER:
{answer}

Output ONLY a JSON object: {{"score": <float 0.0-1.0>, "reason": "<short>"}}."""

_RUBRIC_PROMPT = """You are a strict answer-quality judge for a data-engineering
assistant. Given the QUESTION, the GROUNDING CONTEXT, the CANDIDATE ANSWER, and
the GOLD ANSWER, score the candidate on a 1-5 rubric:
  5 = complete, accurate vs context, expert tone, concise
  1 = incomplete, inaccurate, or verbose/off-topic
Weight grounding: the answer must stay within the context.

QUESTION:
{question}

GROUNDING CONTEXT:
{context}

CANDIDATE ANSWER:
{answer}

GOLD ANSWER:
{ground_truth}

Output ONLY a JSON object: {{"score": <int 1-5>, "reason": "<short>"}}."""


@dataclass
class GenerationEvalRow:
    question: str
    contexts: list[str]
    ground_truth: str
    id: str = ""


@dataclass
class GenerationEvalReport:
    rows: list[dict] = field(default_factory=list)
    faithfulness_mean: float = 0.0
    relevance_mean: float = 0.0
    rubric_mean: float = 0.0
    passed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            "# Generation-Layer Evaluation Report",
            "",
            f"- **Faithfulness** (gate >= {FAITHFULNESS_GATE}): {self.faithfulness_mean:.3f}",
            f"- **Answer relevance** (gate >= {RELEVANCE_GATE}): {self.relevance_mean:.3f}",
            f"- **Rubric correctness** (gate >= {RUBRIC_GATE}): {self.rubric_mean:.3f}",
            f"- **Passed gates:** {self.passed}",
            "",
            "## Per-row",
            "",
            "| id | faithfulness | relevance | rubric |",
            "|----|--------------|-----------|--------|",
        ]
        for r in self.rows:
            lines.append(
                f"| {r.get('id', '')} | {r.get('faithfulness', 0):.3f} | {r.get('relevance', 0):.3f} | {r.get('rubric', 0):.3f} |"
            )
        return "\n".join(lines)


def load_generation_dataset(path: str) -> list[GenerationEvalRow]:
    """Load a JSONL dataset of frozen (question, contexts, ground_truth) rows.

    Accepts either ``ground_truth`` or ``answer`` as the gold reference, and
    ``contexts`` as the frozen gold context list.
    """
    rows: list[GenerationEvalRow] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append(
                GenerationEvalRow(
                    question=d["question"],
                    contexts=list(d.get("contexts") or []),
                    ground_truth=d.get("ground_truth") or d.get("answer") or "",
                    id=d.get("id") or "",
                )
            )
    return rows


def _parse_score(text: str, lo: float, hi: float) -> float:
    """Extract a numeric score from a judge response (JSON or bare number)."""
    if not text:
        return 0.0
    m = re.search(r'\{[^{}]*"score"\s*:\s*(-?[0-9]+(?:\.[0-9]+)?)', text)
    if m:
        return min(hi, max(lo, float(m.group(1))))
    m = re.search(r"(-?[0-9]+(?:\.[0-9]+)?)", text)
    if m:
        return min(hi, max(lo, float(m.group(1))))
    return 0.0


async def _generate_answer(generator, prompt: str) -> str:
    return (await generator.generate(prompt)).strip()


async def score_faithfulness(judge, question: str, answer: str, contexts: list[str], n_trials: int = 1) -> float:
    context = "\n\n".join(contexts)
    prompt = _FAITHFULNESS_PROMPT.format(question=question, context=context, answer=answer)
    scores = [(_parse_score(await judge.generate(prompt), 0.0, 1.0)) for _ in range(max(1, n_trials))]
    return statistics.fmean(scores)


async def score_relevance(judge, question: str, answer: str, n_trials: int = 1) -> float:
    prompt = _RELEVANCE_PROMPT.format(question=question, answer=answer)
    scores = [(_parse_score(await judge.generate(prompt), 0.0, 1.0)) for _ in range(max(1, n_trials))]
    return statistics.fmean(scores)


async def score_rubric(
    judge, question: str, answer: str, ground_truth: str, contexts: list[str], n_trials: int = 3
) -> float:
    context = "\n\n".join(contexts)
    prompt = _RUBRIC_PROMPT.format(question=question, context=context, answer=answer, ground_truth=ground_truth)
    scores = [(_parse_score(await judge.generate(prompt), 1.0, 5.0)) for _ in range(max(1, n_trials))]
    return statistics.fmean(scores)


async def evaluate_generation(
    dataset_path: str,
    settings: AppSettings | None = None,
    *,
    generator=None,
    judge=None,
    n_trials: int = 3,
    intent: str = "factual",
) -> GenerationEvalReport:
    """Evaluate the generation layer alone on a frozen gold-context dataset.

    ``generator`` and ``judge`` are injectable for hermetic tests; when omitted
    they are built from ``settings`` (answer and evaluation purposes).
    """
    rows = load_generation_dataset(dataset_path)
    if (generator is None or judge is None) and settings is None:
        raise ValueError("settings must be provided when generator/judge are not injected")
    if generator is None:
        assert settings is not None
        generator = build_llm_fallback_chain(purpose="answer", app_settings=settings)
    if judge is None:
        assert settings is not None
        judge = build_llm_fallback_chain(purpose="evaluation", app_settings=settings)
    pb = PromptBuilder()

    results: list[dict] = []
    for row in rows:
        context_str = "\n\n".join(row.contexts)
        prompt = pb.build_rag_prompt(context=context_str, question=row.question, intent=intent)
        answer = await _generate_answer(generator, prompt)
        faith = await score_faithfulness(judge, row.question, answer, row.contexts, n_trials=n_trials)
        rel = await score_relevance(judge, row.question, answer, n_trials=n_trials)
        rubric = await score_rubric(judge, row.question, answer, row.ground_truth, row.contexts, n_trials=n_trials)
        results.append(
            {
                "id": row.id,
                "question": row.question,
                "answer": answer,
                "faithfulness": faith,
                "relevance": rel,
                "rubric": rubric,
            }
        )
        logger.info(
            "generation_eval_row id=%s faithfulness=%.3f relevance=%.3f rubric=%.3f",
            row.id,
            faith,
            rel,
            rubric,
        )

    faith_mean = statistics.fmean([r["faithfulness"] for r in results]) if results else 0.0
    rel_mean = statistics.fmean([r["relevance"] for r in results]) if results else 0.0
    rubric_mean = statistics.fmean([r["rubric"] for r in results]) if results else 0.0
    passed = faith_mean >= FAITHFULNESS_GATE and rel_mean >= RELEVANCE_GATE and rubric_mean >= RUBRIC_GATE
    return GenerationEvalReport(
        rows=results,
        faithfulness_mean=faith_mean,
        relevance_mean=rel_mean,
        rubric_mean=rubric_mean,
        passed=passed,
    )


if __name__ == "__main__":
    import asyncio

    from data_engineering_copilot.config.settings import settings as _settings

    async def _main() -> None:
        report = await evaluate_generation("tests/evaluation/eval_dataset.jsonl", _settings, n_trials=3)
        print(report.to_markdown())

    asyncio.run(_main())
