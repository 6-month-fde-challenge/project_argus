"""ARGUS platform API - control plane and backend-for-frontend for the ARGUS UI.

    uvicorn app.main:app --port 8090     (OpenAPI docs at /docs)

Every feature lives in app/features/<name>/ and registers its routers in app/features/__init__.py.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import get_settings
from .core.container import Container
from .core.errors import install_error_handlers
from .features import ROUTERS
from .features.jobs.service import scheduler_loop

API_PREFIX = "/api/v1"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app(container: Container | None = None, run_scheduler: bool = True) -> FastAPI:
    container = container or Container(get_settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        stop = asyncio.Event()
        task = asyncio.create_task(scheduler_loop(container, stop)) if run_scheduler else None
        yield
        stop.set()
        if task:
            await task

    app = FastAPI(title="ARGUS Platform API", version="1.0.0", lifespan=lifespan,
                  description="Training runs, live progress, jobs, model registry, scoring and admin for Project ARGUS.")
    app.state.container = container
    app.add_middleware(CORSMiddleware, allow_origins=container.settings.cors_origins, allow_methods=["*"],
                       allow_headers=["*"])
    install_error_handlers(app)
    for router in ROUTERS:
        app.include_router(router, prefix=API_PREFIX)
    return app


app = create_app()
