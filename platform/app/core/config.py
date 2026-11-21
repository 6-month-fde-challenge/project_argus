"""Platform settings: one typed object, read from environment variables (prefix ARGUS_)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ARGUS_", extra="ignore")

    # paths inside the platform container (bind-mounted by docker-compose)
    outputs_dir: Path = Path("/workspace/outputs")
    configs_dir: Path = Path("/workspace/configs")
    data_dir: Path = Path("/workspace/data")
    docs_dir: Path = Path("/workspace/client_docs")
    diagrams_dir: Path = Path("/workspace/diagrams")

    # neighbouring services (compose service names)
    scoring_url: str = "http://scoring:8080"
    mlflow_url: str = "http://mlflow:5000"
    mlflow_public_url: str = "http://localhost:5000"   # what the browser opens / embeds
    explainer_url: str = "http://explainer:8000"
    scoring_token: str = "demo-service-token"
    scoring_admin_token: str = "demo-admin-token"

    # training jobs (launched as sibling containers through the Docker socket)
    training_image: str = "argus-training:latest"
    job_container_prefix: str = "argus-job-"
    job_shm_size: str = "8g"
    job_poll_seconds: float = 2.0
    gpu_service_names: tuple[str, ...] = ("scoring", "explainer")  # compose services that hold the GPU
    passthrough_env: tuple[str, ...] = ("SUPABASE_URL", "SUPABASE_KEY", "HF_TOKEN")

    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    @property
    def platform_dir(self) -> Path:
        """Platform-owned state (job registry) lives next to the runs, outside any run folder."""
        return self.outputs_dir / "_platform"


@lru_cache
def get_settings() -> Settings:
    return Settings()
