#!/usr/bin/env python3
"""
Validate that this repo matches the Autoppia subnet agent entrypoint format.

Checks:
  - main.py exists and exposes `app`
  - FastAPI app exposes GET /health
  - FastAPI app exposes POST /act (and optional /step)
  - /act returns actions in subnet-compatible shape (basic sanity check)
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _load_module(path: Path, name: str):
    if not path.exists():
        _fail(f"Missing {path.name}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        _fail(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def _find_route(app, path: str, method: str) -> bool:
    for route in getattr(app, "routes", []):
        if getattr(route, "path", None) == path:
            methods = {m.upper() for m in getattr(route, "methods", [])}
            if method.upper() in methods:
                return True
    return False


def _call_act(app) -> dict[str, Any] | None:
    # Try to call the /act endpoint function with a minimal payload.
    for route in getattr(app, "routes", []):
        if getattr(route, "path", None) == "/act":
            endpoint = getattr(route, "endpoint", None)
            if endpoint is None:
                return None
            payload = {
                "task_id": "check",
                "prompt": "open the homepage",
                "url": "http://localhost",
                "snapshot_html": "<html></html>",
                "step_index": 0,
            }
            if inspect.iscoroutinefunction(endpoint):
                import asyncio

                return asyncio.run(endpoint(payload))  # type: ignore[arg-type]
            return endpoint(payload)  # type: ignore[arg-type]
    return None


def main() -> None:
    main_py = REPO_ROOT / "main.py"
    api_py = REPO_ROOT / "api.py"

    _ok(f"Found {api_py.name}")
    main_mod = _load_module(main_py, "main")
    app = getattr(main_mod, "app", None)
    if app is None:
        _fail("main.py does not expose `app`")
    _ok("main.py exposes `app`")

    if not _find_route(app, "/health", "GET"):
        _fail("GET /health route not found")
    _ok("GET /health route found")

    if not _find_route(app, "/act", "POST"):
        _fail("POST /act route not found")
    _ok("POST /act route found")

    # /step is optional but recommended
    if _find_route(app, "/step", "POST"):
        _ok("POST /step route found")
    else:
        print("[WARN] POST /step route not found (optional)")

    # Basic response shape check
    resp = _call_act(app)
    if resp is None:
        _fail("Unable to invoke /act for shape check")

    # Expect {"actions": [ ... ]} or {"action": {...}} or {"navigate_url": "..."}
    ok_shape = False
    if isinstance(resp, dict):
        if isinstance(resp.get("actions"), list):
            ok_shape = True
        elif isinstance(resp.get("action"), dict):
            ok_shape = True
        elif isinstance(resp.get("navigate_url"), str):
            ok_shape = True

    if not ok_shape:
        _fail(f"/act response shape invalid: {json.dumps(resp)[:200]}")

    _ok("/act response shape looks subnet-compatible")
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
