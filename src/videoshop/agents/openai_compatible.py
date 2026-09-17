from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Sequence


class LLMHTTPError(RuntimeError):
    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"LLM HTTP {status_code}: {body}")


class OpenAICompatibleToolClient:
    """Minimal OpenAI-compatible chat/completions client for tool calling.

    Works with SGLang/OpenAI-compatible endpoints that accept /chat/completions
    and return choices[0].message.tool_calls.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        timeout: float = 60.0,
        max_tokens: int | None = 1024,
    ) -> None:
        self.base_url = (base_url or os.environ.get("VIDEOSHOP_LLM_BASE_URL") or "").rstrip("/")
        self.api_key = api_key or os.environ.get("VIDEOSHOP_LLM_API_KEY") or ""
        self.model = model or os.environ.get("VIDEOSHOP_LLM_MODEL") or "Qwen3.8-27B"
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.last_response_metadata: dict[str, Any] = {}
        if not self.base_url:
            raise ValueError("base_url is required. Set VIDEOSHOP_LLM_BASE_URL or pass base_url=...")

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Sequence[Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        started = time.perf_counter()
        response = self._post_json("/chat/completions", payload)
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        choices = response.get("choices", [])
        if not choices:
            raise RuntimeError(f"LLM response has no choices: {response}")
        self.last_response_metadata = {
            "response_id": response.get("id"),
            "model": response.get("model", self.model),
            "created": response.get("created"),
            "finish_reason": choices[0].get("finish_reason"),
            "usage": response.get("usage", {}),
            "latency_ms": latency_ms,
        }
        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            return tool_calls

        # Some OpenAI-compatible deployments return plain JSON content instead of tool_calls.
        content = message.get("content")
        if content:
            return _tool_calls_from_plain_content(content)
        return []

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise LLMHTTPError(exc.code, body) from exc


def _tool_calls_from_plain_content(content: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return []

    if isinstance(parsed, dict) and "tool_calls" in parsed:
        return parsed["tool_calls"]
    if isinstance(parsed, dict) and "tool_requests" in parsed:
        calls = [
            {"name": item["tool"], "arguments": item.get("args", {})}
            for item in parsed.get("tool_requests", [])
        ]
        if parsed.get("final_action"):
            calls.append({"name": "submit_final_action", "arguments": parsed["final_action"]})
        return calls
    if isinstance(parsed, list):
        return parsed
    return []
