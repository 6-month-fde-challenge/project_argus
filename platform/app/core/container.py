"""Composition root: every adapter is built once here and handed to routers through `Depends(get_container)`.

Tests construct a Container with fakes (e.g. a fake job runner, temp folders) instead of patching modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

from fastapi import Request

from ..infra.docs_repo import DocsRepository
from ..infra.gpu import GpuMonitor
from ..infra.http_clients import ExplainerClient, MlflowClient, ScoringClient, StreamClient
from ..infra.job_runner import DockerJobRunner, JobRunner
from ..infra.job_store import JobStore
from ..infra.profiles_repo import ProfileRepository
from ..infra.runs_repo import RunRepository
from .config import Settings


@dataclass
class Container:
    settings: Settings
    runner_override: JobRunner | None = None  # tests inject a fake; production builds Docker lazily
    _runner: JobRunner | None = field(default=None, init=False)

    @cached_property
    def runs(self) -> RunRepository:
        return RunRepository(self.settings.outputs_dir)

    @cached_property
    def profiles(self) -> ProfileRepository:
        return ProfileRepository(self.settings.configs_dir)

    @cached_property
    def jobs(self) -> JobStore:
        return JobStore(self.settings.platform_dir / "jobs.json")

    @property
    def runner(self) -> JobRunner:
        """Docker is optional for read-only use: it is only connected when a job or service action needs it."""
        if self.runner_override is not None:
            return self.runner_override
        if self._runner is None:
            self._runner = DockerJobRunner(self.settings)
        return self._runner

    @cached_property
    def scoring(self) -> ScoringClient:
        return ScoringClient(self.settings)

    @cached_property
    def mlflow(self) -> MlflowClient:
        return MlflowClient(self.settings)

    @cached_property
    def stream(self) -> StreamClient:
        return StreamClient(self.settings)

    @cached_property
    def explainer(self) -> ExplainerClient:
        return ExplainerClient(self.settings)

    @cached_property
    def gpu(self) -> GpuMonitor:
        return GpuMonitor()

    @cached_property
    def docs(self) -> DocsRepository:
        return DocsRepository(self.settings.docs_dir, self.settings.diagrams_dir)

    @cached_property
    def live(self):
        from ..features.runs.live import LiveStateCache
        return LiveStateCache(self.runs)


def get_container(request: Request) -> Container:
    return request.app.state.container
