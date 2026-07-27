"""Prompt construction service for LLM synthesis.

Decouples prompt template rendering and system instructions from low-level
HTTP client infrastructure.
"""

from __future__ import annotations

CODE_INTENTS = frozenset({"code_example", "api_lookup"})

_CODE_INSTRUCTIONS = (
    "1. Output clean, runnable Python/PySpark code with type hints.\n"
    "2. Include concise inline comments for non-obvious logic.\n"
    "3. Do NOT include long introductory or concluding text outside the code.\n"
    "4. Cite the source documentation for API signatures used.\n"
    "5. If context lacks sufficient API details, state the limitation explicitly."
)

_DOCUMENTATION_INSTRUCTIONS = (
    "1. For factual questions: State facts from the docs clearly.\n"
    "2. For comparative questions: Show differences between the documented options.\n"
    "3. For procedural questions: Outline steps from the documentation.\n"
    "4. For open-ended questions: Provide a thoughtful synthesis of available info.\n"
    "5. When uncertain: Explicitly say 'The documentation does not clearly address this'."
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
        """
        instructions = _CODE_INSTRUCTIONS if intent in CODE_INTENTS else _DOCUMENTATION_INSTRUCTIONS
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
                "Return ONLY valid JSON with this exact structure (no markdown, no code fences):",
                "{",
                '  "answer": "Your detailed answer here, 2-4 sentences.",',
                '  "citations": [',
                '    {"source": "Source Name", "snippet": "Direct quote from documentation"}',
                "  ]",
                "}",
                'If no sources are directly referenced, return "citations": [].',
                "",
                "## INSTRUCTIONS",
                instructions,
                "",
                "## USER QUESTION AND CONTEXT",
                f"Context:\n{context}\n\nQuestion: {question}",
                "",
                "## YOUR STRUCTURED ANSWER",
            ]
        )
