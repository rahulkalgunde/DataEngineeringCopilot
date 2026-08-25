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

import asyncio
import json
import logging
import re
import statistics
from dataclasses import asdict, dataclass, field

from data_engineering_copilot.config.settings import AppSettings
from data_engineering_copilot.evaluation.judge_cache import JudgeCache, judge_cache_key
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
    intent: str = "factual"
    probe: str = ""
    expect_refusal: bool = False


def stratified_sample(rows: list, n: int, key) -> list:
    """Deterministic round-robin sample across strata (rows sorted by id first)."""
    if n <= 0 or not rows:
        return []
    strata: dict[str, list] = {}
    for r in sorted(rows, key=lambda r: r.id or ""):
        strata.setdefault(key(r) or "default", []).append(r)
    out: list = []
    max_depth = max((len(b) for b in strata.values()), default=0)
    depth = 0
    while len(out) < n and depth < max_depth:
        for bucket in strata.values():
            if depth < len(bucket) and len(out) < n:
                out.append(bucket[depth])
        depth += 1
    return out


@dataclass
class GenerationEvalReport:
    rows: list[dict] = field(default_factory=list)
    faithfulness_mean: float = 0.0
    relevance_mean: float = 0.0
    rubric_mean: float = 0.0
    # Fraction of rows where a second independent judge's rubric score is
    # within ±1 of the primary judge (MT-Bench-style agreement check).
    # None when no second judge was supplied.
    judge_agreement: float | None = None
    llm_usage: dict | None = None
    pairwise: dict | None = None
    robustness: dict | None = None
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
            *(
                [
                    f"- **Pairwise vs baseline:** win={self.pairwise['win']} tie={self.pairwise['tie']} loss={self.pairwise['loss']}"
                ]
                if self.pairwise
                else []
            ),
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
                    intent=d.get("intent", "factual"),
                    probe=d.get("probe", ""),
                    expect_refusal=bool(d.get("expect_refusal", False)),
                )
            )
    return rows


def _parse_score(text: str, lo: float, hi: float) -> float | None:
    """Extract a numeric score from a judge response (JSON or bare number).

    Returns None when nothing parseable is found — distinct from a legitimate
    zero score, which callers must not retry on.
    """
    if not text:
        return None
    m = re.search(r'\{[^{}]*"score"\s*:\s*(-?[0-9]+(?:\.[0-9]+)?)', text)
    if m:
        return min(hi, max(lo, float(m.group(1))))
    m = re.search(r"(-?[0-9]+(?:\.[0-9]+)?)", text)
    if m:
        return min(hi, max(lo, float(m.group(1))))
    return None


async def _judge_call_with_retry(
    judge,
    prompt: str,
    lo: float,
    hi: float,
    *,
    max_retries: int = 3,
) -> float:
    """Call judge + parse with retry ONLY on unparseable output.

    A parsed 0 is a valid verdict (common for faithfulness refusals) and must
    return immediately — retrying it tripled spend for zero information.
    """
    for attempt in range(max_retries):
        raw = await judge.generate(prompt)
        s = _parse_score(raw, lo, hi)
        if s is not None:
            return s
        if attempt < max_retries - 1:
            print(f"    judge-retry {attempt + 2}/{max_retries}: unparseable (raw={raw[:80]!r})")
    return 0.0


def resolve_judge(*, local, cloud, enabled: bool, threshold: float, band: float):
    """Return the judge object used by score_* functions.

    Cascade semantics live at SCORING level: we wrap the LOCAL judge so that
    scores landing inside [threshold-band, threshold+band] are re-scored by the
    cloud chain. When disabled, the primary (cloud/evaluation) chain is used
    directly — identical to today's behavior.
    """
    if not enabled or local is None or cloud is None:
        return cloud if cloud is not None else local

    class _BandEscalating:
        async def generate(self, prompt: str) -> str:
            reply = await local.generate(prompt)
            try:
                obj = json.loads(reply)
                s = float(obj.get("score"))
            except Exception:  # noqa: BLE001 — unparseable => escalate
                return await cloud.generate(prompt)
            if abs(s - threshold) <= band:
                return await cloud.generate(prompt)
            return reply

    return _BandEscalating()


def pairwise_verdict(score_a_ord1: float, score_b_ord1: float, score_a_ord2: float, score_b_ord2: float) -> str:
    """MT-Bench-style swapped-order verdict. 'A'/'B' only on consistent wins.

    Scores are (judge_score_for_A, judge_score_for_B) in each ordering.
    Position-swap cancels positional bias: A wins only if preferred in BOTH orderings.
    """
    a_wins_1 = score_a_ord1 > score_b_ord1
    b_wins_1 = score_b_ord1 > score_a_ord1
    a_wins_2 = score_a_ord2 > score_b_ord2
    b_wins_2 = score_b_ord2 > score_a_ord2
    if a_wins_1 and a_wins_2:
        return "A"
    if b_wins_1 and b_wins_2:
        return "B"
    return "TIE"


def load_baseline_answers(path: str) -> dict[str, str]:
    """Load baseline answers JSONL ({id, answer}) -> id -> answer."""
    import pathlib as _pl

    p = _pl.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"baseline answers not found: {p}")
    out: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out[d["id"]] = d.get("answer", "")
    return out


async def _generate_answer(generator, prompt: str) -> str:
    return _extract_answer_text((await generator.generate(prompt)).strip())


def _extract_answer_text(raw: str) -> str:
    """Normalize generator output for scoring.

    Structured doc-intent answers arrive as raw JSON
    (``{"answer": ..., "missing_info": ...}``). Extract the human answer so
    faithfulness/relevance judge clean text; a null answer with missing_info
    collapses to a canonical refusal sentence that matches the refusal
    markers used by the robustness scorer. Non-JSON output passes through.
    """
    if not raw.startswith("{"):
        return raw
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw
    if not isinstance(data, dict) or "answer" not in data:
        return raw
    answer = data.get("answer")
    if answer is None or (isinstance(answer, str) and not answer.strip()):
        return "Insufficient context to answer."
    return str(answer).strip()


async def score_faithfulness(
    judge,
    question: str,
    answer: str,
    contexts: list[str],
    n_trials: int = 1,
    cache: JudgeCache | None = None,
    model_id: str = "",
) -> float:
    context = "\n\n".join(contexts)
    prompt = _FAITHFULNESS_PROMPT.format(question=question, context=context, answer=answer)
    scores = []
    for _ in range(max(1, n_trials)):
        ck = judge_cache_key(model_id, "faithfulness-v1", question, answer, context) if cache else ""
        hit = await cache.get(ck) if cache else None
        if hit is not None:
            scores.append(float(hit["score"]))
            continue
        s = await _judge_call_with_retry(judge, prompt, 0.0, 1.0)
        if cache:
            await cache.put(ck, {"score": s})
        scores.append(s)
    return statistics.fmean(scores)


async def score_relevance(
    judge,
    question: str,
    answer: str,
    n_trials: int = 1,
    cache: JudgeCache | None = None,
    model_id: str = "",
) -> float:
    prompt = _RELEVANCE_PROMPT.format(question=question, answer=answer)
    scores = []
    for _ in range(max(1, n_trials)):
        ck = judge_cache_key(model_id, "relevance-v1", question, answer, "") if cache else ""
        hit = await cache.get(ck) if cache else None
        if hit is not None:
            scores.append(float(hit["score"]))
            continue
        s = await _judge_call_with_retry(judge, prompt, 0.0, 1.0)
        if cache:
            await cache.put(ck, {"score": s})
        scores.append(s)
    return statistics.fmean(scores)


async def score_rubric(
    judge,
    question: str,
    answer: str,
    ground_truth: str,
    contexts: list[str],
    n_trials: int = 3,
    cache: JudgeCache | None = None,
    model_id: str = "",
    epsilon: float | None = None,
) -> float:
    """Rubric score with optional adaptive early-stop (epsilon on last two trials)."""
    context = "\n\n".join(contexts)
    prompt = _RUBRIC_PROMPT.format(question=question, context=context, answer=answer, ground_truth=ground_truth)
    scores: list[float] = []
    cap = max(1, n_trials)
    while len(scores) < cap:
        ck = judge_cache_key(model_id, "rubric-v1", question, f"{answer}\x00{ground_truth}", context) if cache else ""
        hit = await cache.get(ck) if cache else None
        if hit is not None:
            s = float(hit["score"])
        else:
            s = await _judge_call_with_retry(judge, prompt, 1.0, 5.0)
            if cache:
                await cache.put(ck, {"score": s})
        scores.append(s)
        if epsilon is not None and len(scores) >= 2 and abs(scores[-1] - scores[-2]) <= epsilon:
            break
    return statistics.fmean(scores)


async def evaluate_generation(
    dataset_path: str,
    settings: AppSettings | None = None,
    *,
    generator=None,
    judge=None,
    judge_b=None,
    judges: list | None = None,
    n_trials: int = 3,
    intent: str = "factual",
    sample: int = 0,
    stratify_by: str = "intent",
    compare_answers: dict[str, str] | None = None,
    row_concurrency: int = 1,
) -> GenerationEvalReport:
    """Evaluate the generation layer alone on a frozen gold-context dataset.

    ``generator`` and ``judge`` are injectable for hermetic tests; when omitted
    they are built from ``settings`` (answer and evaluation purposes). When
    ``judge_b`` is supplied, every row is additionally rubric-scored by it and
    the report carries ``judge_agreement`` — the fraction of rows where the two
    judges' scores differ by at most 1 point (LLM-judge reliability check).
    """
    rows = load_generation_dataset(dataset_path)
    if (generator is None or (judge is None and not judges)) and settings is None:
        raise ValueError("settings must be provided when generator/judge are not injected")
    empty = [r.id or r.question[:40] for r in rows if not (r.contexts and any(c.strip() for c in r.contexts))]
    if empty:
        raise ValueError(
            f"generation dataset has {len(empty)} rows with empty contexts (e.g. {empty[:2]!r}) — fix dataset before paid run"
        )
    from data_engineering_copilot.infrastructure.provider_fallback import UsageLedger

    UsageLedger.reset()
    probe_rows = [r for r in rows if r.expect_refusal]
    rows = [r for r in rows if not r.expect_refusal]
    if sample and sample > 0:
        keyfn = (lambda r: getattr(r, "source_name", "")) if stratify_by == "source_name" else (lambda r: r.intent)
        total = len(rows)
        rows = stratified_sample(rows, sample, keyfn)
        print(f"Stratified sample: {len(rows)} of {total} rows (by {stratify_by})")
    if generator is None:
        assert settings is not None
        generator = build_llm_fallback_chain(purpose="answer", app_settings=settings)
    if judge is None and not judges:
        assert settings is not None
        judge = build_llm_fallback_chain(purpose="evaluation", app_settings=settings)
    if settings is not None and getattr(settings, "judge_cascade_enabled", False):
        local_judge = build_llm_fallback_chain(purpose="evaluation", app_settings=settings)
        judge = resolve_judge(
            local=local_judge,
            cloud=judge,
            enabled=True,
            threshold=0.85,
            band=getattr(settings, "judge_cascade_band", 0.15),
        )
    pb = PromptBuilder()

    cache: JudgeCache | None = None
    if settings is not None and getattr(settings, "judge_cache_enabled", False):
        try:
            from data_engineering_copilot.factory import get_shared_redis_client

            cache = JudgeCache(
                enabled=True,
                ttl_days=getattr(settings, "judge_cache_ttl_days", 30),
                client=get_shared_redis_client(),
            )
        except Exception as exc:  # noqa: BLE001 - fail-open contract
            logger.warning("judge cache disabled for this run: %s", exc)
            cache = None
    model_id = str(getattr(judge, "model_id", "") or "evaluation-chain")
    if judges:
        n_trials = 1  # majority mode: breadth across judges replaces depth per judge

    sem = asyncio.Semaphore(max(1, int(row_concurrency)))

    async def _score_with(j, row, answer):
        faith = await score_faithfulness(
            j, row.question, answer, row.contexts, n_trials=n_trials, cache=cache, model_id=model_id
        )
        rel = await score_relevance(j, row.question, answer, n_trials=n_trials, cache=cache, model_id=model_id)
        rub = await score_rubric(
            j,
            row.question,
            answer,
            row.ground_truth,
            row.contexts,
            n_trials=n_trials,
            cache=cache,
            model_id=model_id,
            epsilon=getattr(settings, "adaptive_trial_epsilon", None) if settings else None,
        )
        return faith, rel, rub

    async def _eval_row(row) -> dict:
        async with sem:
            context_str = "\n\n".join(row.contexts)
            prompt = pb.build_rag_prompt(context=context_str, question=row.question, intent=intent)
            answer = await _generate_answer(generator, prompt)
            if judges:
                votes = await asyncio.gather(*[_score_with(j, row, answer) for j in judges])
                faith = statistics.median(v[0] for v in votes)
                rel = statistics.median(v[1] for v in votes)
                rubric = statistics.median(v[2] for v in votes)
                out = {
                    "id": row.id,
                    "question": row.question,
                    "answer": answer,
                    "faithfulness": faith,
                    "relevance": rel,
                    "rubric": rubric,
                    "judge_votes": [{"faithfulness": f, "relevance": r, "rubric": b} for f, r, b in votes],
                }
            else:
                faith, rel, rubric = await _score_with(judge, row, answer)
                out = {
                    "id": row.id,
                    "question": row.question,
                    "answer": answer,
                    "faithfulness": faith,
                    "relevance": rel,
                    "rubric": rubric,
                }
            logger.info(
                "generation_eval_row id=%s faithfulness=%.3f relevance=%.3f rubric=%.3f",
                row.id,
                faith,
                rel,
                rubric,
            )
            return out

    results = list(await asyncio.gather(*[_eval_row(r) for r in rows]))

    faith_mean = statistics.fmean([r["faithfulness"] for r in results]) if results else 0.0
    rel_mean = statistics.fmean([r["relevance"] for r in results]) if results else 0.0
    rubric_mean = statistics.fmean([r["rubric"] for r in results]) if results else 0.0

    judge_agreement: float | None = None
    robustness: dict | None = None
    if probe_rows:
        from data_engineering_copilot.evaluation.robustness_probes import looks_like_refusal

        refusals_ok = 0
        for r in probe_rows:
            ctx = "\n\n".join(r.contexts)
            prompt = f"Answer using ONLY this context.\nCONTEXT:\n{ctx}\n\nQUESTION:\n{r.question}"
            ans = await _generate_answer(generator, prompt)
            if looks_like_refusal(ans):
                refusals_ok += 1
        robustness = {"total": len(probe_rows), "refusal_correct": refusals_ok}
        print(f"Robustness probes: {refusals_ok}/{len(probe_rows)} refusals correct")

    if judge_b is not None and not judges and results:
        agreements: list[float] = []
        for row in rows:
            answer = next(r["answer"] for r in results if r["id"] == row.id)
            rubric_b = await score_rubric(judge_b, row.question, answer, row.ground_truth, row.contexts, n_trials=1)
            rubric_a = next(r["rubric"] for r in results if r["id"] == row.id)
            agreements.append(1.0 if abs(rubric_a - rubric_b) <= 1.0 else 0.0)
        judge_agreement = statistics.fmean(agreements)

    if judges and results:
        # majority mode: pairwise rubric agreement across all judge pairs,
        # averaged over rows (|Δ| <= 1 counts as agreement).
        per_row: list[float] = []
        for r in results:
            rubs = [v["rubric"] for v in r.get("judge_votes", [])]
            pairs = [(a, b) for i, a in enumerate(rubs) for b in rubs[i + 1 :]]
            if pairs:
                per_row.append(statistics.fmean([1.0 if abs(x - y) <= 1.0 else 0.0 for x, y in pairs]))
        judge_agreement = statistics.fmean(per_row) if per_row else None

    pairwise: dict | None = None
    if compare_answers and results and not judges:
        wins = ties = losses = 0
        for row in rows:
            if row.id not in compare_answers:
                continue
            cur_rubric = next((r["rubric"] for r in results if r["id"] == row.id), 0.0)
            base_answer = compare_answers[row.id]
            base_rubric = await score_rubric(
                judge, row.question, base_answer, row.ground_truth, row.contexts, n_trials=1
            )
            verdict = pairwise_verdict(cur_rubric, base_rubric, base_rubric, cur_rubric)
            if verdict == "A":
                wins += 1
            elif verdict == "B":
                losses += 1
            else:
                ties += 1
        pairwise = {"win": wins, "tie": ties, "loss": losses}
        print(f"Pairwise vs baseline: win={wins} tie={ties} loss={losses}")

    passed = faith_mean >= FAITHFULNESS_GATE and rel_mean >= RELEVANCE_GATE and rubric_mean >= RUBRIC_GATE
    return GenerationEvalReport(
        rows=results,
        faithfulness_mean=faith_mean,
        relevance_mean=rel_mean,
        rubric_mean=rubric_mean,
        judge_agreement=judge_agreement,
        llm_usage=UsageLedger.snapshot(),
        pairwise=pairwise,
        robustness=robustness,
        passed=passed,
    )


if __name__ == "__main__":
    import asyncio

    from data_engineering_copilot.config.settings import settings as _settings

    async def _main() -> None:
        report = await evaluate_generation("tests/evaluation/eval_dataset.jsonl", _settings, n_trials=3)
        print(report.to_markdown())

    asyncio.run(_main())
