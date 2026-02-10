from typing import Any, Dict

import os

from fastapi import Body, FastAPI


FIXED_AUTBOOKS_URL = os.getenv(
    "FIXED_AUTBOOKS_URL",
    "http://84.247.180.192:8001/books/book-original-002?seed=36",
)

app = FastAPI(title="Autoppia Web Agent API")


@app.get("/health", summary="Health check")
async def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.post("/act", summary="Decide next agent actions")
async def act(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Minimal CUA endpoint.

    Ignores the incoming observation/state and always returns a single
    navigate action to the known Autobooks BOOK_DETAIL page.

    This matches the behavior of FixedAutobooksAgent and is ideal as a
    simple test agent for the subnet + sandbox pipeline.
    """
    return {
        "actions": [
            {
                "type": "NavigateAction",
                "url": FIXED_AUTBOOKS_URL,
            }
        ]
    }


@app.post("/step", summary="Alias for /act")
async def step(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return await act(payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
