# 012: Prompt augmentation hardening — salted tags, escaping, sandwiching, citation tri-state

Date: 2026-08-21 · Status: Accepted

## Context

Retrieved documentation is untrusted input. Microsoft's spotlighting research
(arXiv:2403.14720) shows data-marking drops indirect prompt-injection attack
success from >50% to <2% with minimal quality impact.

## Decision

Four prompt-builder defenses, each behind its own flag:

1. **Salted XML tags** (`prompt_salted_xml_tags=True`): context is wrapped in
   `<context_data_{per-request-hex}>` generated via `secrets.token_hex(4)` —
   cached injection payloads targeting predictable tag names fail.
2. **XML content escaping** (`prompt_xml_content_escape=True`): `&` and `<`
   escaped in chunk text so retrieved content cannot close the context tag.
3. **Instruction sandwiching** (`prompt_trailing_instructions=True`): a
   "## CRITICAL REMINDERS" block repeats key constraints after the
   question/context.
4. **Citation enforcement** (`prompt_citation_enforcement`: strict|soft|off):
   `[Doc-N]` citation rules baked into prompts; post-generation
   `verify_citations` drops citations not matching retrieved sources.
   Note: "strict" and "soft" render identically today — only "off" branches;
   soft is reserved for a future lenient variant.

## Consequences

- The `rag-answer` Langfuse prompt must stay byte-compatible with the
  offline fallback template (`prompt_builder._RAG_PROMPT_TEMPLATE`).
- Defense is layered with the chunk-level injection scanner
  (`services/input_guardrails.py`) which drops offending chunks pre-prompt.
