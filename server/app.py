"""OpenEnv server entry point.

This module re-exports the FastAPI application so that openenv deployment
modes (``openenv serve``, ``python -m``) can locate the ASGI app at the
conventional ``server.app:app`` path, alongside the primary entry point at
``forge_arena.main:app``.

Both paths point to the same application object — no duplication of logic.
"""

import uvicorn

from forge_arena.main import app  # noqa: F401  re-exported for openenv discovery

__all__ = ["app", "main"]


def main() -> None:
    """Run the Forge Arena server with uvicorn."""
    uvicorn.run("forge_arena.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
