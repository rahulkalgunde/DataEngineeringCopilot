from __future__ import annotations

import json
import logging
import re
from urllib.error import URLError
from urllib.request import Request, urlopen

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    """Raised when local Ollama cannot return an answer."""


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: int,
        num_ctx: int,
        num_predict: int,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.num_ctx = num_ctx
        self.num_predict = num_predict

    def generate(self, prompt: str, num_predict: int | None = None, num_ctx: int | None = None) -> str:
        logger.info(
            "Ollama generation started model=%s prompt_chars=%s num_ctx=%s num_predict=%s",
            self.model,
            len(prompt),
            self.num_ctx,
            self.num_predict,
        )
        if num_predict is None:
            num_predict = self.num_predict
        if num_ctx is None:
            num_ctx = self.num_ctx

        payload = {
            "model": self.model,
            "prompt": self._format_raw_chat_prompt(prompt),
            "raw": True,
            "stream": False,
            "options": {
                "temperature": 0.05,
                "top_p": 0.8,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
            },
        }
        try:
            body = self._http_post(payload)
        except TimeoutError as exc:
            logger.exception("Ollama generation timed out timeout_seconds=%s", self.timeout_seconds)
            raise OllamaError(
                f"Ollama timed out after {self.timeout_seconds} seconds. "
                "Try again after Ollama finishes loading the model, or reduce the configured context/output limits."
            ) from exc
        except URLError as exc:
            logger.exception("Ollama connection failed base_url=%s", self.base_url)
            raise OllamaError("Could not reach Ollama. Start Ollama and run: ollama pull %s", self.model) from exc

        response = self._extract_final_response(str(body.get("response", "")))
        done_reason = body.get("done_reason", "unknown")
        logger.info(
            "Ollama generation completed done_reason=%s raw_response_chars=%s final_response_chars=%s",
            done_reason,
            len(str(body.get("response", ""))),
            len(response),
        )
        if not response:
            logger.warning("Ollama returned no final answer done_reason=%s body_keys=%s", done_reason, sorted(body))
            raise OllamaError(
                "Ollama returned no final answer. "
                f"Generation stopped with reason `{done_reason}`. "
                "The model likely spent its output budget on reasoning. "
                "Try again, or increase `ollama_num_predict` in settings.py."
            )
        return response

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((TimeoutError, ConnectionError, OSError)),
        reraise=True,
    )
    def _http_post(self, payload: dict) -> dict:
        request = Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def _extract_final_response(self, response: str) -> str:
        response = response.strip()
        if not response:
            return ""

        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE).strip()
        if response.lower().startswith("<think>"):
            return ""
        return response

    def _format_raw_chat_prompt(self, user_prompt: str) -> str:
        """Format the user prompt with structured instructions for the LLM.

        Uses a multi-part template that:
        1. Establishes role and constraints
        2. Provides explicit instructions
        3. Specifies output format
        4. Handles uncertainty
        """
        return "\n".join(
            [
                "## SYSTEM",
                "You are DataEngineeringCopilot, an expert data engineering assistant.",
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
                "Provide a clear, concise answer with these components:",
                "- Answer: [Your direct answer, 2-4 sentences]",
                "- Key Points: [2-3 bullet points if applicable]",
                "- Important Notes: [Caveats or limitations, if any]",
                "- Not Covered: [What the docs don't address, if relevant]",
                "",
                "## INSTRUCTIONS",
                "1. For factual questions: State facts from the docs clearly.",
                "2. For comparative questions: Show differences between the documented options.",
                "3. For procedural questions: Outline steps from the documentation.",
                "4. For open-ended questions: Provide a thoughtful synthesis of available info.",
                "5. When uncertain: Explicitly say 'The documentation does not clearly address this'.",
                "",
                "## USER QUESTION AND CONTEXT",
                user_prompt,
                "",
                "## YOUR STRUCTURED ANSWER",
            ]
        )
