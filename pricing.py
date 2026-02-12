"""pricing.py

Token pricing helpers for local eval.

Source of truth: OpenAI model pages / API pricing.
- gpt-4o: $2.50 / 1M input tokens, $10.00 / 1M output tokens
- gpt-4o-mini: $0.15 / 1M input tokens, $0.60 / 1M output tokens

This is only an estimate: it ignores cached-input discounts and any gateway-side
adjustments, and assumes Chat Completions token accounting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class ModelPrice:
    input_per_1m: float
    output_per_1m: float


# Keep this small and explicit; extend as needed.
_PRICES: dict[str, ModelPrice] = {
    "gpt-4o": ModelPrice(input_per_1m=2.50, output_per_1m=10.00),
    "gpt-4o-mini": ModelPrice(input_per_1m=0.15, output_per_1m=0.60),
}


def _normalize_model(model: str) -> str:
    m = (model or "").strip()
    # Strip snapshot suffix if present, keep the base alias.
    for base in sorted(_PRICES.keys(), key=len, reverse=True):
        if m == base or m.startswith(base + "-"):
            return base
    return m


def price_for_model(model: str) -> Optional[ModelPrice]:
    return _PRICES.get(_normalize_model(model))


def estimate_cost_usd(model: str, usage: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """Estimate USD cost from an OpenAI-like usage dict.

    usage should have prompt_tokens and completion_tokens (ints).
    """
    p = price_for_model(model)
    pt = int(usage.get("prompt_tokens") or 0)
    ct = int(usage.get("completion_tokens") or 0)

    if not p:
        return 0.0, {"error": "unknown_model", "model": model, "prompt_tokens": pt, "completion_tokens": ct}

    cost = (pt / 1_000_000.0) * p.input_per_1m + (ct / 1_000_000.0) * p.output_per_1m
    return float(cost), {
        "model": _normalize_model(model),
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "input_per_1m": p.input_per_1m,
        "output_per_1m": p.output_per_1m,
    }
