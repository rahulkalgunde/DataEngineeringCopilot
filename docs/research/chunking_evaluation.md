# Evaluating Text Chunking Quality in Isolation

**Scope:** Evaluating *chunking* as a standalone stage — **before** retrieval, reranking, and generation — using gold spans/structure as ground truth. No retrieval, no ranking, no answer generation. All claims are sourced from primary documentation, framework source code, or landmark papers. Blog/secondary sources are flagged where used and are **not** cited as primary evidence.

**Headline finding (load-bearing):** The only published, runnable chunking-evaluation benchmark we found is Chroma's *Chunking Evaluation* technical report + code (`brandonstarxel/chunking_evaluation`). Its core metric is an **Intersection-over-Union over *character* offset ranges**, computed on **retrieved** chunks, not on chunking-in-isolation. It also ships a **synthetic gold-data generator** (GPT-4) with a whitespace/fuzzy-tolerant matcher. Several of the proposed metrics (Token-IoU-as-named, Excerpt Precision / "Precision_Ω", Boundary Integrity Rate, Structural Boundary Fracture Rate) are **not standard named metrics in the primary literature** — they are proposed here and we map them onto established primitives (IoU, segmentation boundary metrics, structural-parsing checks). Each is flagged below.

---

## 1. Token-level IoU (Intersection-over-Union on token spans)

**What it is:** For each gold span $g$ and each predicted chunk $c$, compute $\text{IoU} = |g \cap c| / |g \cup c|$ over **token indices**; aggregate per gold span (max over predicted chunks) then average over spans/queries. Higher = predicted chunk boundaries fit the gold span.

**Primary basis:**
- Chroma's *Chunking Evaluation* report defines the central measure as IoU of the **retrieved** chunk against the question's gold reference span, averaged (max-over-chunks, mean-over-queries). ([trychroma.com/research/evaluating-chunking](https://www.trychroma.com/research/evaluating-chunking)) Their code, however, computes IoU over **character** ranges via `union_ranges` / `intersect_two_ranges` in `base_evaluation.py` — *not* token indices. So "token-level IoU" as proposed here is a **reframing** of their character-level method onto the token space; the aggregation logic (max-over-predicted, mean-over-gold) carries over directly.
- For the **token space itself**, the canonical, dependency-free primitive is `tiktoken` (already a dependency of this repo: `tiktoken>=0.7` in `pyproject.toml`). **Critical implementation caveat (verified locally against `cl100k_base`):** `tiktoken` exposes **no** `return_offsets_mapping` / `char_to_token` API. Token→character alignment must be recovered by `enc.decode(ids[a:b])` (substring by decoding a token range) or by tracking **byte** offsets of the UTF-8 encoding. This makes a naive "tokenize each chunk separately then compare" approach wrong, because a chunk split mid-token/byte re-tokenizes differently. **Recommendation:** tokenize the **whole document once**, then map each gold char-span $[s,e)$ and each predicted chunk char-span $[s,e)$ to token-index intervals $[t_s, t_e)$ via byte-offset search; compute IoU on those intervals. This sidesteps char↔token ambiguity entirely.

**Implementation recommendation:**
1. `ids = enc.encode(doc)`; build cumulative byte-offset table from `doc.encode("utf-8")` so char span → byte span → token interval.
2. For each gold span and each predicted chunk, get token-interval IoU.
3. Aggregate `mean(max_iou_gold)` across spans (mirrors Chroma's max-over-predicted, mean-over-gold).

**Verdict (1):** Mature *methodology* (IoU is standard; tokenization is the only nuance), **Medium effort** (whole-doc tokenization + offset table), **Medium risk** (tokenizer drift between embedder/chunker and evaluator; char vs token off-by-one). Use the whole-doc tokenization pattern to remove the ambiguity.

---

## 2. Excerpt Precision (proposed "Precision_Ω")

**What it is (as proposed):** Fraction of predicted chunks (or of the union of predicted-chunk content) that is "on-topic"/contained within the gold excerpt set — i.e., do our chunks avoid dragging in irrelevant surrounding text?

**Primary basis & caveat:** "Excerpt Precision" / "Precision_Ω" is **not a standard named metric** in the primary chunking or segmentation literature we surveyed. The closest primary primitive is Chroma's IoU itself, which is symmetric (it already penalizes chunks that over-cover the gold span by shrinking the union denominator). A precision-directional variant (reward only overlap, ignore under-coverage) is a legitimate **derived** metric but should be defined explicitly and not attributed to an existing framework. Treat the name as **proposed/non-standard**.

**How to ground it in primary methods:**
- Repurpose the IoU overlap term as a *precision* signal: for a gold span $g$ and its best-matching predicted chunk $c^*$, define $P_\Omega = |g \cap c^*| / |c^*|$ (fraction of the chunk that is gold-relevant) and average over gold spans. This is exactly IoU with the union replaced by the chunk size — a precision-at-chunk measure.
- As a corpus-level check, the **`tiktoken` decode** recovery from §1 lets you measure how much *extra* token mass each chunk carries beyond its nearest gold span (the "irrelevant drag" Precision_Ω is meant to capture).

**Implementation recommendation:** Implement as `mean over gold spans of (|gold ∩ best_chunk| / |best_chunk|)`, computed in the token space per §1. Report alongside IoU so under-coverage (low IoU) vs over-coverage (low Precision_Ω) are separable.

**Verdict (2):** **Non-standard (proposed)**, Mature *primitive* (it's a precision morph of IoU), **Low effort**, **Low–Medium risk** (ambiguous definition if unnamed; pin the formula). Flag in docs as a project-defined metric, not a cited standard.

---

## 3. Span Recall / Boundary Integrity Rate

**What it is (as proposed):** Do predicted chunk boundaries land *near* the gold span boundaries? "Span Recall" = fraction of gold span covered by some predicted chunk; "Boundary Integrity Rate" = fraction of gold boundary points that are matched by a predicted chunk edge within a tolerance.

**Primary basis — segmentation-boundary literature (this *is* a well-established field, just under different names):**
- **SegEval** (`cfournie/segmentation.evaluation`, [segeval.readthedocs.org](https://segeval.readthedocs.org)) is the standard library for text-segmentation evaluation. Its metrics — **WindowDiff**, **Pk**, **Boundary Similarity (B)**, **Segmentation Similarity (S)**, **Boundary Edit Distance** — are exactly "how well do my boundaries align with gold boundaries." Foundational paper: Fournier (2013), *A Simple Window-Diff Based Evaluation Metric for Human/Synthetic Segmentation*, ACL. ([segeval.readthedocs.org](https://segeval.readthedocs.org); [aclanthology.org](https://aclanthology.org/))
- **Boundary Similarity (B)** and **WindowDiff/Pk** are the direct, citable equivalents of the proposed "Boundary Integrity Rate": they score boundary-position agreement between a hypothesized segmentation (your chunks' edges) and a reference segmentation (gold spans' edges), with a tunable $k$-boundary window.
- "Boundary Integrity Rate" as a literal named metric is **not** in the primary literature we found; it is the proposed name for a Boundary-Similarity-style measure. Cite SegEval/Boundary Similarity instead.

**Implementation recommendation:** Use SegEval's `boundary_similarity` / `pk` / `windowdiff` on the sequence of boundary positions (convert gold char-spans and predicted chunks to boundary position lists). For a chunking-tuned tolerance, wrap with a $k$-token window (e.g., $k = \text{chunk\_size}/2$ or a fixed 16-token slack).

**Verdict (3):** Mature & **directly supported by a primary library (SegEval)**, **Low–Medium effort** (boundary-list conversion + SegEval call), **Low risk** (well-validated metric). Strongly prefer SegEval's named metrics over the ad-hoc "Boundary Integrity Rate" label.

---

## 4. Structural Boundary Fracture Rate

**What it is (as proposed):** Fraction of *structural* boundaries (Markdown headers, code-fence boundaries, table boundaries, list items, function/class definitions) that a chunker **splits across** two chunks — i.e., does the chunker respect document structure?

**Primary basis:**
- **CommonMark Spec 0.31.2** and **GitHub Flavored Markdown (GFM)** are the authoritative definitions of the structural units to protect: fenced code blocks (` ``` `), indented code blocks, ATX/Setext headings, and **pipe tables** (GFM §4.10). ([spec.commonmark.org/0.31.2](https://spec.commonmark.org/0.31.2)); ([github.github.com/gfm](https://github.github.com/gfm))
- **cAST — AST-based Chunking for Code (arXiv 2506.15655, EMNLP 2025 Findings, [aclanthology.org/2025.findings-emnlp.430](https://aclanthology.org/2025.findings-emnlp.430)):** establishes that **line-based / character-based code chunking "break semantic structures, splitting functions"**, and that an AST-aware splitter preserves structural integrity far better. This is the primary justification for a *structural* (not just positional) boundary metric — and a ready implementation path: parse to AST / Markdown-parse tree, and count gold structural nodes intersected by a chunk boundary.
- For Markdown specifically, `langchain-text-splitters` `MarkdownHeaderTextSplitter` splits *on headers* by construction ([reference.langchain.com/python/langchain-text-splitters](https://python.langchain.com/docs/integrations/document_transformers/markdown_header_metadata/)), which is the reference behavior a "fracture rate" should reward.

**Implementation recommendation:** Build a structural-node inventory (Markdown: headers/code-fences/tables via a CommonMark+GFM parser such as `markdown-it-py`/`mistletoe`; code: AST nodes via `tree-sitter`). For each node spanning $[s,e)$, check whether any predicted chunk boundary falls strictly inside $(s,e)$. `Structural Boundary Fracture Rate = (# structural nodes cut) / (total structural nodes)`. Lower is better.

**Verdict (4):** Mature *concept* (cAST proves structure-aware splitting matters; CommonMark/GFM define the units), **Medium effort** (need a real parser/AST, not regex), **Medium risk** (parser coverage gaps for exotic formats; must pin the parser version). This is the highest-value *differentiator* metric for a docs/code RAG system.

---

## 5. Existing libraries & tools

| Tool | Type | Chunking-quality metrics? | Primary source |
|---|---|---|---|
| **`brandonstarxel/chunking_evaluation`** (Chroma) | Benchmark + code | Yes — char-IoU on retrieved chunks + synthetic gold generator | [trychroma.com/research/evaluating-chunking](https://www.trychroma.com/research/evaluating-chunking); code in `evaluation_framework/base_evaluation.py` |
| **`chonkie-inc/chonkie`** | Chunking **library** | **No** quality/recall metrics; only deterministic *invariant* tests (reconstruction + offset correctness) | [github.com/chonkie-inc/chonkie](https://github.com/chonkie-inc/chonkie); `tests/chunkers/test_recursive_chunker.py` |
| **`langchain-text-splitters`** | Chunking library | No metrics; provides `add_start_index=True` to recover char offsets (`text.find(chunk, offset)` with overlap-aware offset) | [reference.langchain.com/python/langchain-text-splitters](https://python.langchain.com/docs/integrations/document_transformers/) |
| **SegEval** (`cfournie/segmentation.evaluation`) | Segmentation-eval library | Yes — WindowDiff, Pk, Boundary Similarity, Segmentation Similarity, Boundary Edit Distance | [segeval.readthedocs.org](https://segeval.readthedocs.org) |
| **HiCBench / HiChunk** (arXiv 2509.11552, Tencent Youtu) | Benchmark + framework | Yes — manually annotated **multi-level chunking points** + evidence-dense QA; evaluates chunking across the full RAG pipeline | [arxiv.org/abs/2509.11552](https://arxiv.org/abs/2509.11552) |
| **RAGAS / DeepEval** | RAG eval | **No** chunking-in-isolation metrics (faithfulness, answer relevancy, etc. — retrieval/generation level) | [docs.ragas.io](https://docs.ragas.io); [deepeval.com](https://deepeval.com) |

**Key takeaways:**
- There is **no off-the-shelf "chunking quality" library** that measures span/boundary/structural fitness in isolation. Chonkie and LangChain provide the *chunkers* and (crucially) the **offset-recovery** primitives (`add_start_index`, Chonkie's `start_index/end_index` invariants) you need to *build* the metrics.
- **HiCBench (primary)** is the only benchmark we found that ships **human-annotated chunking-point ground truth** — directly relevant to §6.
- A **secondary** engineering overview (alphaXiv summary of HiChunk) is useful for orientation but is **not** primary; cite the arXiv paper. ([alphaxiv.org](https://alphaxiv.org)) — flagged secondary.

---

## 6. Ground-truth dataset construction

**Two viable primary-backed strategies:**

**(a) Synthetic gold generation (Chroma's approach — primary, runnable):**
- Chroma's `synthetic_evaluation.py` uses GPT-4 with two prompt stages: `_tag_text` embeds `<start>…<end>`-style **100-character tagged chunks** into a document, then `rigorous_document_search` recovers the exact character offsets of each tagged excerpt using a **whitespace-tolerant + fuzzy-tolerant** matcher (`utils.py`). The gold dataset (`questions_df.csv`) stores `question`, `references` (a JSON list of `{content, start_index, end_index}` **character** offsets), and `corpus_id`. (Code: `evaluation_framework/synthetic_evaluation.py`, `utils.py`, `general_evaluation_data/questions_df.csv`.)
- Scale in their release: **472 gold rows** across wikitexts (144) / pubmed (99) / finance (97) / state_of_the_union (76) / chatlogs (56). This is a concrete template for a buildable gold set.
- **Caveat:** offsets are *character* ranges; to get *token* ranges (§1) you must re-derive via §1's tokenization step. The `rigorous_document_search` tolerance means gold spans can be fuzzy — keep the matcher's tolerance explicit and version-pinned.

**(b) Human-annotated multi-level chunking points (HiCBench — primary):**
- HiCBench provides **manually annotated multi-level document chunking points** plus synthesized evidence-dense QA pairs and their evidence sources. The paper's core argument: existing RAG benchmarks are inadequate for assessing chunking quality due to **evidence sparsity**, which is exactly why a chunking-specific gold set is needed. ([arxiv.org/abs/2509.11552](https://arxiv.org/abs/2509.11552))
- This is the stronger ground truth for the *structural* metrics in §3–§4 (real human boundaries at multiple granularities).

**Construction recommendation for this repo:**
1. Start from Chroma's synthetic generator (GPT-4 tagging + tolerant matcher) for volume; **bootstrap a small HiCBench-style human-annotated slice** (50–100 docs) for structural-truth calibration.
2. Store gold as **character offsets** `{content, start_index, end_index, structural_type}` (extend Chroma's schema with a `structural_type` enum: header/code_fence/table/function/paragraph) so §3–§4 can consume it directly.
3. Keep the document **verbatim** (no normalization) so char offsets stay valid; apply normalization only at metric time, recording the transform.

**Verdict (6):** Primary-backed and **buildable** (Chroma generator is runnable; HiCBench schema is documented), **Medium–High effort** (LLM generation + tolerant matcher + a human-annotated calibration slice), **Medium risk** (LLM-generated gold can be wrong → validate a sample with humans; offset drift if docs are re-normalized).

---

## 7. Precise IoU / boundary computation

**Character-range IoU (Chroma's actual method, primary code):**
- `union_ranges(a, b)` merges overlapping/adjacent `[start, end)` intervals; `intersect_two_ranges(a, b)` returns the overlap or empty. IoU = `len(intersect) / len(union)` over `[start, end)` **character** ranges. ([code: `evaluation_framework/base_evaluation.py`])
- **Off-by-one discipline:** use half-open `[start, end)` intervals consistently; a boundary "touch" (adjacent, not overlapping) should be treated as union-merge (Chroma merges adjacent ranges), so a chunk that abuts a gold span scores partial credit rather than zero.

**Token-interval IoU (this repo's recommended reframing, §1):**
- Tokenize the whole doc once → `ids`. Build a cumulative **byte-offset** table from `doc.encode("utf-8")`; map any char span $[s,e)$ to a byte span, then to a token interval $[t_s, t_e)$ by binary search over token byte boundaries. (Verified: `tiktoken` has no `char_to_token`; decode/byte-search is the only correct path — local experiment on `cl100k_base` confirmed `decode(ids[:k])` recovers the exact substring.)
- IoU over **token-index intervals** $= |[t_s^g,t_e^g) \cap [t_s^c,t_e^c)| / |[t_s^g,t_e^g) \cup [t_s^c,t_e^c)|$.
- **Tie/edge rule (Chroma):** if a chunk edge sits exactly on a gold boundary, the overlap is exact; with half-open intervals this is automatic.

**Boundary computation (SegEval-style, §3):**
- Convert gold spans and predicted chunks to **boundary position lists** (the set of indices where a segment ends). Feed to `segeval.boundary_similarity` / `segeval.pk` / `segeval.windowdiff` with a $k$-window. Boundary Similarity $B \in [0,1]$, higher better; Pk/WindowDiff lower better. ([segeval.readthedocs.org](https://segeval.readthedocs.org))

**Structural computation (§4):** per structural node $[s,e)$, fracture = any predicted boundary in $(s,e)$. (See §4 for parser basis.)

**Implementation recommendation:** Implement one `Span` dataclass (`start`, `end`, `unit: {"char"|"token"|"byte"}`), plus `iou(a, b)`, `boundary_positions(spans)`, and `structural_fracture(struct_nodes, chunk_bounds)`. Unit-test the half-open interval math with adversarial cases (adjacent, nested, empty, identical).

**Verdict (7):** Mature *math* (interval IoU + SegEval are well-defined), **Low–Medium effort**, **Low risk** (logic is small and unit-testable; main hazard is off-by-one/unit mismatch — pin `[start, end)` everywhere).

---

## 8. Deterministic testing (pytest fixtures, golden datasets, regression gates)

**What it is:** Lock chunker behavior so refactors can't silently regress boundary/structure fidelity. Three layers, all primary-backed by existing tooling:

**(a) Invariant tests (proven pattern from Chonkie — primary code):**
Chonkie's test suite asserts, for every chunker: `sample_text == "".join(chunk.text for chunk in chunks)` (reconstruction), `chunk.text == sample_text[start_index:end_index]` (offset correctness), `start_index >= 0`, `end_index <= len(text)`. ([github.com/chonkie-inc/chonkie `tests/chunkers/test_recursive_chunker.py`](https://github.com/chonkie-inc/chonkie)) **Adopt these as the baseline CI contract** for every chunker in `tests/unit/` (this repo already has `test_*.py` for hierarchical/sentence-preserving/structured-data/spark chunkers — extend each with these invariants).

**(b) Golden-dataset regression (Chroma + HiCBench gold — primary):**
Freeze a gold set (§6) as a pytest fixture; assert metric scores stay within a pinned band (`assert iou >= 0.90`, `assert structural_fracture_rate <= 0.05`). On intentional change, update the band via **snapshot review**, not silent edit.

**(c) Snapshot / golden-file plugin — `syrupy` (primary, PyPI):**
`syrupy` is a pytest snapshot-testing plugin that serializes outputs (e.g., the full chunk-boundary list for a fixture doc) to an `.ambr` golden file; CI fails on diff, forcing human review of boundary changes. ([pypi.org/project/syrupy](https://pypi.org/project/syrupy)) Pair with `pytest-xdist` (already in `pyproject.toml` dev deps, `-n 6`) — syrupy is xdist-safe.

**Repo integration notes (from AGENTS.md / pyproject):**
- Use `make_settings()` (Ollama-only, no env files) for any test needing settings; provider-routing tests pass `_test_allow_non_ollama=True`. No API keys in golden tests — **all chunking metrics here are offline** (tiktoken + SegEval + local parsers), so they belong in `make test-quick` / `make test-unit`, not the infra-gated `make test-real`.
- Mark slow generators (if you generate gold at test time) with `@pytest.mark.slow` so `make test-quick` skips them.

**Implementation recommendation:** Add `tests/unit/test_chunking_metrics.py` with (1) invariant tests per chunker, (2) a `golden_chunks` fixture loading a tiny frozen gold CSV, (3) syrupy snapshots of boundary lists, (4) asserted metric bands. Wire a `make test-chunking` (or fold into `make test-unit`).

**Verdict (8):** Mature *tooling* (Chonkie invariants + syrupy + SegEval all primary/established), **Medium effort** (write fixtures + bands once), **Low risk** (purely additive CI; offline). Highest ROI of the eight — it makes §1–§7 *trustworthy* over time.

---

## Verdict Per Metric — Maturity / Effort / Risk

| # | Metric | Maturity | Effort | Risk | Notes |
|---|---|---|---|---|---|
| 1 | Token-level IoU | High (IoU standard; reframing of Chroma char-IoU) | Medium | Medium | Use whole-doc `tiktoken` tokenization + byte-offset table; `tiktoken` has **no** `char_to_token`. |
| 2 | Excerpt Precision (Precision_Ω) | **Non-standard (proposed)**; primitive = precision-morph of IoU | Low | Low–Med | Pin the exact formula; do **not** cite as an existing framework metric. |
| 3 | Span Recall / Boundary Integrity Rate | High (SegEval Boundary Similarity/Pk/WindowDiff) | Low–Med | Low | Prefer SegEval named metrics over the ad-hoc label. |
| 4 | Structural Boundary Fracture Rate | High concept (cAST + CommonMark/GFM) | Medium | Medium | Needs real parser/AST; highest-value differentiator for docs/code RAG. |
| 5 | Libraries & tools | High (Chroma, Chonkie, LangChain, SegEval, HiCBench) | Low (survey) | Low | No off-the-shelf "chunking quality" lib; compose from these. |
| 6 | Ground-truth construction | High (Chroma synthetic + HiCBench human) | Med–High | Medium | LLM gold needs human validation slice; store char offsets + structural_type. |
| 7 | Precise IoU / boundary computation | High (interval math + SegEval) | Low–Med | Low | Pin `[start, end)` half-open everywhere; unit-test edge cases. |
| 8 | Deterministic testing | High (Chonkie invariants + syrupy + SegEval) | Medium | Low | Offline → fits `make test-unit`; highest ROI. |

**Bottom line for a chunking-evaluation plan:**
1. **Stand up §8 first** (invariant + golden + syrupy regression) so every later number is trustworthy and refactor-safe.
2. **Implement §1 + §3** (token-IoU via whole-doc `tiktoken`; SegEval Boundary Similarity) as the core fitness signals — both are primary-backed and offline.
3. **Add §4** (structural fracture) if your corpus is Markdown/code-heavy — it is the differentiator and is justified by cAST + CommonMark/GFM.
4. **Define §2 (Precision_Ω) explicitly** as a project metric (precision-morph of IoU), not a cited standard.
5. **Build §6 gold data** from Chroma's synthetic generator + a small HiCBench-style human slice; store char offsets + `structural_type`.
6. **Treat §5** as a build-vs-buy decision: compose Chroma's IoU + SegEval + LangChain/Chonkie offset recovery; do **not** expect a single library to give you §1–§4 out of the box.
