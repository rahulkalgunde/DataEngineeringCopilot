"""Prompt construction service for LLM synthesis.

Decouples prompt template rendering and system instructions from low-level
HTTP client infrastructure.
"""

from __future__ import annotations


class PromptBuilder:
    """Builds structured prompts for RAG context synthesis."""

    def __init__(self, system_role: str | None = None) -> None:
        self.system_role = system_role or "You are DataEngineeringCopilot, an expert data engineering assistant."

    def build_rag_prompt(self, context: str, question: str) -> str:
        """Construct a structured system prompt combining context and question."""
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
                "1. For factual questions: State facts from the docs clearly.",
                "2. For comparative questions: Show differences between the documented options.",
                "3. For procedural questions: Outline steps from the documentation.",
                "4. For open-ended questions: Provide a thoughtful synthesis of available info.",
                "5. When uncertain: Explicitly say 'The documentation does not clearly address this'.",
                "",
                "## USER QUESTION AND CONTEXT",
                f"Context:\n{context}\n\nQuestion: {question}",
                "",
                "## YOUR STRUCTURED ANSWER",
            ]
        )
