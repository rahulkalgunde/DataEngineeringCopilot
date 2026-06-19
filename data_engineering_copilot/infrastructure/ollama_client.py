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

    def generate(self, prompt: str) -> str:
        logger.info(
            "Ollama generation started model=%s prompt_chars=%s num_ctx=%s num_predict=%s",
            self.model,
            len(prompt),
            self.num_ctx,
            self.num_predict,
        )
        payload = {
            "model": self.model,
            "prompt": self._format_raw_chat_prompt(prompt),
            "raw": True,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
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
                "Could not reach Ollama. Start Ollama and run: ollama pull qwen3:4b"
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
                "<|im_start|>system",
                "You are DataEngineeringCopilot. Answer directly and concisely without reasoning.",
                "Use ONLY the repository context provided by the user.",
                "Respond immediately with facts from the context. No analysis, no thinking, no <think> tags.",
                "Keep responses brief and practical.",
                "<|im_end|>",
                "<|im_start|>user",
                "/no_think",
                user_prompt,
                "<|im_end|>",
                "<|im_start|>assistant",
                "",
            ]
        )
