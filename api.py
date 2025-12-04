"""Compatibility entrypoint for Uvicorn.

Allows running the app with `uvicorn api:app` from the repo root by forwarding
to the actual FastAPI instance in `backend/api.py`.
"""

from backend.api import app

