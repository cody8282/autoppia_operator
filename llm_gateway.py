from __future__ import annotations

from typing import Any, Dict, List, Optional

import os

import httpx


class LLMGateway:
    """
    Minimal gateway-style OpenAI client.

    Key requirement for Subnet/IWA: every LLM request must include the task id:
      IWA-Task-ID: <task_id>

    The validator's sandbox injects OPENAI_BASE_URL so requests go through the
    local gateway proxy (e.g. http://sandbox-gateway:9000/openai/v1).

    Notes:
    - We resolve OPENAI_API_KEY at request time by default, to avoid stale keys
      in long-lived processes.
    """

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key  # If None, read from env per request.
        self.timeout_seconds = float(timeout_seconds)

    def chat_completions(self, *, task_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        api_key = self.api_key if self.api_key is not None else os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "IWA-Task-ID": str(task_id),
        }

        with httpx.Client(timeout=self.timeout_seconds) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                json=body,
                headers=headers,
            )
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                # Avoid logging full provider error bodies (they may echo key suffixes).
                detail = ""
                try:
                    j = e.response.json()
                    err = j.get("error", {}) if isinstance(j, dict) else {}
                    detail = f"type={err.get('type')} code={err.get('code')}"
                except Exception:
                    detail = (e.response.text or "")[:200]
                raise RuntimeError(f"OpenAI error ({e.response.status_code}): {detail}") from e

            return resp.json()


_default_gateway = LLMGateway()


def openai_chat_completions(
    *,
    task_id: str,
    messages: List[Dict[str, Any]],
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 300,
) -> Dict[str, Any]:
    """Convenience wrapper used by the agent."""
    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        # Prefer JSON-mode when available, but retry without it for providers/models
        # that reject `response_format` on /chat/completions.
        "response_format": {"type": "json_object"},
    }
    try:
        return _default_gateway.chat_completions(task_id=task_id, body=body)
    except RuntimeError as e:
        msg = str(e)
        if 'unsupported_parameter' in msg or 'response_format' in msg:
            body2 = dict(body)
            body2.pop('response_format', None)
            return _default_gateway.chat_completions(task_id=task_id, body=body2)
        raise
