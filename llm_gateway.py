from __future__ import annotations

from typing import Any, Dict, List, Optional

import os

import httpx


def is_sandbox_gateway_base_url(base_url: str) -> bool:
    """
    Detect whether requests are routed through the validator's local sandbox gateway.

    In the subnet runtime, the validator injects:
      OPENAI_BASE_URL=http://sandbox-gateway:9000/openai/v1
    and the gateway uses its own upstream provider keys. Agent code must not
    require (nor receive) real provider API keys in this mode.
    """
    if os.getenv("SANDBOX_GATEWAY_URL"):
        return True
    try:
        url = httpx.URL((base_url or "").strip())
        host = (url.host or "").lower()
        # Allow common local dev hosts too; the gateway doesn't require auth.
        return host in {"sandbox-gateway", "localhost", "127.0.0.1"}
    except Exception:
        return False


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
        if not api_key and not is_sandbox_gateway_base_url(self.base_url):
            raise RuntimeError("OPENAI_API_KEY not set")

        headers = {
            "Content-Type": "application/json",
            "IWA-Task-ID": str(task_id),
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

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
    """Convenience wrapper used by the agent.

    Notes:
    - Some models (e.g. gpt-5) reject `temperature` and `max_tokens` on `/chat/completions`.
      For those we use `max_completion_tokens` and omit `temperature`.
    - We prefer JSON-mode when available, but retry without it if rejected.
    """
    m = str(model)

    # Base payload all models accept.
    body: Dict[str, Any] = {
        "model": m,
        "messages": messages,
    }

    # Model-specific parameterization.
    if m.startswith('gpt-5'):
        # gpt-5: use max_completion_tokens; do not send temperature; avoid response_format.
        body["max_completion_tokens"] = int(max_tokens)
    else:
        body.update({
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "response_format": {"type": "json_object"},
        })

    def _post(b: Dict[str, Any]) -> Dict[str, Any]:
        return _default_gateway.chat_completions(task_id=task_id, body=b)

    try:
        return _post(body)
    except RuntimeError as e:
        msg = str(e)
        # Retry without JSON-mode.
        if 'unsupported_parameter' in msg or 'response_format' in msg:
            b2 = dict(body)
            b2.pop('response_format', None)
            return _post(b2)

        # gpt-5 (or future models): if max_tokens is rejected, try max_completion_tokens.
        if 'unsupported_parameter' in msg and 'max_tokens' in body and 'max_completion_tokens' not in body:
            b2 = dict(body)
            b2.pop('max_tokens', None)
            b2.pop('temperature', None)
            b2["max_completion_tokens"] = int(max_tokens)
            b2.pop('response_format', None)
            return _post(b2)

        # If temperature is rejected, drop it.
        if 'unsupported_value' in msg and 'temperature' in body:
            b2 = dict(body)
            b2.pop('temperature', None)
            return _post(b2)

        raise
