from __future__ import annotations

import json
import logging
import re
import socket
from urllib.error import URLError
from urllib.request import Request, urlopen


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
        request = Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except TimeoutError as exc:
            logger.exception("Ollama generation timed out timeout_seconds=%s", self.timeout_seconds)
            raise OllamaError(
                f"Ollama timed out after {self.timeout_seconds} seconds. "
                "Try again after Ollama finishes loading the model, or reduce the configured context/output limits."
            ) from exc
        except socket.timeout as exc:
            logger.exception("Ollama generation socket timeout timeout_seconds=%s", self.timeout_seconds)
            raise OllamaError(
                f"Ollama timed out after {self.timeout_seconds} seconds. "
                "Try again after Ollama finishes loading the model, or reduce the configured context/output limits."
            ) from exc
        except URLError as exc:
            logger.exception("Ollama connection failed base_url=%s", self.base_url)
            raise OllamaError(
                "Could not reach Ollama. Start Ollama and run: ollama pull %s", self.model
            ) from exc

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

    def _extract_final_response(self, response: str) -> str:
        response = response.strip()
        if not response:
            return ""

        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE).strip()
        if response.lower().startswith("<think>"):
            return ""
        return response

    def _format_raw_chat_prompt(self, user_prompt: str) -> str:
        return "\n".join(
            [
                "You are DataEngineeringCopilot. Answer concisely using only provided context.",
                "- No invented information.",
                "- Brief and practical.",
                "",
                user_prompt,
                "",
                "Answer:",
            ]
        )
