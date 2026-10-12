from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import get_settings
from app.dashboard.audit_reader import (
    AuditReader,
    DashboardDatabaseError,
)
from app.dashboard.schemas import (
    DashboardEvent,
    DashboardSummary,
    DashboardTrendPoint,
)


router = APIRouter(tags=["dashboard"])

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _ensure_enabled():
    settings = get_settings()

    if not settings.dashboard_enabled:
        raise HTTPException(
            status_code=404,
            detail="Dashboard is disabled.",
        )

    return settings


def _reader() -> AuditReader:
    settings = get_settings()
    return AuditReader(settings.security_audit_db_path)


# Serve dashboard CSS and JavaScript files.
router.mount(
    "/dashboard/static",
    StaticFiles(directory=STATIC_DIR),
    name="dashboard-static",
)


@router.get("/dashboard")
def dashboard_page():
    _ensure_enabled()

    dashboard_file = STATIC_DIR / "index.html"

    if not dashboard_file.exists():
        raise HTTPException(
            status_code=503,
            detail="Dashboard UI is unavailable.",
        )

    return FileResponse(
        dashboard_file,
        media_type="text/html",
    )


@router.get(
    "/dashboard/api/summary",
    response_model=DashboardSummary,
)
def dashboard_summary(
    hours: int = Query(
        default=24,
        ge=1,
        le=168,
    ),
):
    settings = _ensure_enabled()

    hours = min(
        hours,
        settings.dashboard_max_hours,
    )

    try:
        return _reader().summary(hours)
    except DashboardDatabaseError:
        raise HTTPException(
            status_code=503,
            detail="Dashboard data is unavailable.",
        )


@router.get(
    "/dashboard/api/events",
    response_model=list[DashboardEvent],
)
def dashboard_events(
    limit: int = Query(
        default=50,
        ge=1,
        le=100,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
    outcome: str | None = None,
    language: str | None = None,
):
    settings = _ensure_enabled()

    limit = min(
        limit,
        settings.dashboard_max_events,
    )

    try:
        return _reader().events(
            limit=limit,
            offset=offset,
            outcome=outcome,
            language=language,
        )
    except DashboardDatabaseError:
        raise HTTPException(
            status_code=503,
            detail="Dashboard data is unavailable.",
        )


@router.get(
    "/dashboard/api/trends",
    response_model=list[DashboardTrendPoint],
)
def dashboard_trends(
    hours: int = Query(
        default=24,
        ge=1,
        le=168,
    ),
    bucket: str = Query(
        default="hour",
    ),
):
    settings = _ensure_enabled()

    if bucket != "hour":
        raise HTTPException(
            status_code=400,
            detail="Only hourly buckets are supported.",
        )

    hours = min(
        hours,
        settings.dashboard_max_hours,
    )

    try:
        return _reader().trends(hours)
    except DashboardDatabaseError:
        raise HTTPException(
            status_code=503,
            detail="Dashboard data is unavailable.",
        )