from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.core.config import get_settings
from app.dashboard.router import router as dashboard_router

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Multilingual AI safety middleware — Phase 1 MVP (POST /analyze)",
    version="0.1.0",
)

app.include_router(api_router)
app.include_router(dashboard_router)

dashboard_static_dir = (
    Path(__file__).resolve().parent
    / "dashboard"
    / "static"
)

app.mount(
    "/dashboard/static",
    StaticFiles(directory=dashboard_static_dir),
    name="dashboard-static",
)


@app.get("/")
def root() -> dict:
    return {
        "service": settings.app_name,
        "phase": 1,
        "status": "running",
    }