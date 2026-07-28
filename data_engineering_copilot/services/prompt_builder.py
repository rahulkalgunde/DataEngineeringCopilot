"""Prompt construction service for LLM synthesis.

Decouples prompt template rendering and system instructions from low-level
HTTP client infrastructure.
"""

from __future__ import annotations

import re

CODE_INTENTS = frozenset({"code_example", "api_lookup"})

# Pattern to detect code-related keywords in queries (for safety net)
_CODE_KEYWORDS = re.compile(
    r"\b(code|script|function|implement|snippet|sample|example|pyspark|scala|python)\b",
    re.IGNORECASE,
)

_CODE_INSTRUCTIONS = (
    "1. Provide a brief explanation (1-3 sentences) followed by a complete, runnable code example.\n"
    "2. Use fenced code blocks with the appropriate language tag (e.g. ```scala, ```python).\n"
    "3. Match the language requested by the user (Scala, Python, SQL, etc.).\n"
    "4. Include concise inline comments for non-obvious logic.\n"
    "5. Cite the source documentation for API signatures used.\n"
    "6. If context lacks sufficient API details, state the limitation explicitly."
)

_DOCUMENTATION_INSTRUCTIONS = (
    "1. For factual questions: State facts from the docs clearly.\n"
    "2. For comparative questions: Show differences between the documented options.\n"
    "3. For procedural questions: Outline steps from the documentation.\n"
    "4. For open-ended questions: Provide a thoughtful synthesis of available info.\n"
    "5. When uncertain: Explicitly say 'The documentation does not clearly address this'."
)

# Safety net: allow code blocks in documentation answers when query contains code keywords
_DOCUMENTATION_INSTRUCTIONS_WITH_CODE = (
    "1. For factual questions: State facts from the docs clearly.\n"
    "2. For comparative questions: Show differences between the documented options.\n"
    "3. For procedural questions: Outline steps from the documentation.\n"
    "4. For open-ended questions: Provide a thoughtful synthesis of available info.\n"
    "5. When uncertain: Explicitly say 'The documentation does not clearly address this'.\n"
    "6. If the user asks for code or the query contains code-related keywords, include a complete, runnable code example in a fenced code block."
)

# Code-intent output format — allows fenced code blocks in the answer
_CODE_OUTPUT_FORMAT = (
    "Return your answer as:\n"
    "1. A brief explanation (1-3 sentences)\n"
    "2. A fenced code block with the implementation (use the language the user asked for)\n"
    "3. Source citations\n\n"
    "Example:\n"
    "Brief explanation of the approach.\n\n"
    "```scala\n"
    "// implementation code here\n"
    "```\n\n"
    "Sources: [list of source names]"
)

# Documentation output format — structured JSON
_DOC_OUTPUT_FORMAT = (
    'Return ONLY valid JSON with this exact structure (no markdown, no code fences):\n'
    '{\n'
    '  "answer": "Your detailed answer here, 2-4 sentences.",\n'
    '  "citations": [\n'
    '    {"source": "Source Name", "snippet": "Direct quote from documentation"}\n'
    '  ]\n'
    '}\n'
    'If no sources are directly referenced, return "citations": [].'
)


class PromptBuilder:
    """Builds structured prompts for RAG context synthesis."""

    def __init__(self, system_role: str | None = None) -> None:
        self.system_role = system_role or "You are DataEngineeringCopilot, an expert data engineering assistant."

    def build_rag_prompt(self, context: str, question: str, intent: str = "factual") -> str:
        """Construct a structured system prompt combining context and question.

        Parameters
        ----------
        intent:
            Query intent from ``QueryRewriter.classify_intent()``.
            ``code_example`` and ``api_lookup`` get code-focused instructions;
            all others get the default documentation-focused instructions.
        question:
            Original user question, used for code keyword detection (safety net).
        """
        is_code = intent in CODE_INTENTS
        has_code_keywords = bool(_CODE_KEYWORDS.search(question))

        # Safety net: allow code blocks even for non-code intents if query contains code keywords
        if is_code:
            instructions = _CODE_INSTRUCTIONS
            output_format = _CODE_OUTPUT_FORMAT
        elif has_code_keywords:
            instructions = _DOCUMENTATION_INSTRUCTIONS_WITH_CODE
            output_format = _CODE_OUTPUT_FORMAT  # Allow code blocks in output
        else:
            instructions = _DOCUMENTATION_INSTRUCTIONS
            output_format = _DOC_OUTPUT_FORMAT

        return "\n".join(
            [
                "## SYSTEM",
                self.system_role,
                "Your role is to answer questions using ONLY the provided documentation context.",
                "",
                "## CONSTRAINTS",
                "1. Base your answer strictly on the provided context.",
                "2. Do NOT invent, assume, or use external knowledge.",
                "3. If information is missing or unclear, explicitly state the limitation.",
                "4. Cite specific documentation sources when possible.",
                "5. Use precise technical terminology from the context.",
                "",
                "## OUTPUT FORMAT",
                output_format,
                "",
                "## INSTRUCTIONS",
                instructions,
                "",
                "## USER QUESTION AND CONTEXT",
                f"Context:\n{context}\n\nQuestion: {question}",
                "",
                "## YOUR ANSWER",
            ]
        )
