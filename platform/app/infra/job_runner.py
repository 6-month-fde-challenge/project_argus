"""Launches training runs as sibling containers of the `argus-training` image through the Docker socket.

Bind mounts of a sibling container need *host* paths, which a container cannot know. The runner therefore inspects
its own container and reuses the host source of each of its mounts (outputs, data, HF cache, configs, code), so it
works on Windows and Linux without any path configuration.
"""
from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from ..core.config import Settings
from ..core.errors import UpstreamError

# where the training image expects each folder; the platform mounts the same folders at the same paths
JOB_MOUNTS = ("/workspace/outputs", "/workspace/data", "/workspace/.hf_cache", "/workspace/configs",
              "/workspace/argus_ft")


@dataclass
class ContainerState:
    running: bool
    exit_code: int | None


class JobRunner(Protocol):
    def launch(self, job_id: str, run_name: str, command: list[str]) -> str: ...
    def state(self, container_id: str) -> ContainerState | None: ...
    def stop(self, container_id: str) -> None: ...
    def logs(self, container_id: str, follow: bool, tail: int) -> Iterator[bytes]: ...
    def gpu_holders(self) -> list[str]: ...
    def service_status(self) -> dict[str, str]: ...
    def set_service(self, service: str, running: bool) -> None: ...


class DockerJobRunner:
    def __init__(self, settings: Settings):
        import docker  # imported lazily: tests and docker-less environments never need it
        self.s = settings
        try:
            self.client = docker.from_env()
            self.client.ping()
        except Exception as e:  # noqa: BLE001 - any docker failure means "not available"
            raise UpstreamError(f"Docker is not reachable from the platform (is the socket mounted?): {e}") from e
        self._docker = docker

    # ------------------------------------------------------------ jobs

    def _host_mounts(self) -> dict[str, dict]:
        me = self.client.containers.get(socket.gethostname())
        sources = {m["Destination"]: m["Source"] for m in me.attrs.get("Mounts", [])}
        missing = [d for d in JOB_MOUNTS if d not in sources]
        if missing:
            raise UpstreamError(f"platform container is missing mounts needed by training jobs: {missing}")
        return {sources[d]: {"bind": d, "mode": "rw"} for d in JOB_MOUNTS}

    def launch(self, job_id: str, run_name: str, command: list[str]) -> str:
        env = {k: os.environ[k] for k in self.s.passthrough_env if os.environ.get(k)}
        container = self.client.containers.run(
            self.s.training_image, command, detach=True, name=f"{self.s.job_container_prefix}{job_id}",
            labels={"argus.job": job_id, "argus.run": run_name}, volumes=self._host_mounts(), environment=env,
            working_dir="/workspace", shm_size=self.s.job_shm_size,
            device_requests=[self._docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],
        )
        return container.id

    def _get(self, container_id: str):
        try:
            return self.client.containers.get(container_id)
        except self._docker.errors.NotFound:
            return None

    def state(self, container_id: str) -> ContainerState | None:
        c = self._get(container_id)
        if c is None:
            return None
        st = c.attrs["State"]
        return ContainerState(running=st.get("Running", False), exit_code=None if st.get("Running") else st.get("ExitCode"))

    def stop(self, container_id: str) -> None:
        c = self._get(container_id)
        if c is not None:
            c.stop(timeout=10)

    def logs(self, container_id: str, follow: bool, tail: int) -> Iterator[bytes]:
        c = self._get(container_id)
        if c is None:
            return iter(())
        return c.logs(stream=True, follow=follow, tail=tail)

    # ------------------------------------------------------------ serving services (compose-managed)

    def _service(self, name: str):
        found = self.client.containers.list(all=True, filters={"label": f"com.docker.compose.service={name}"})
        return found[0] if found else None

    def service_status(self) -> dict[str, str]:
        out = {}
        for name in self.s.gpu_service_names:
            c = self._service(name)
            out[name] = "not created" if c is None else c.status
        return out

    def gpu_holders(self) -> list[str]:
        return [n for n, st in self.service_status().items() if st == "running"]

    def set_service(self, service: str, running: bool) -> None:
        c = self._service(service)
        if c is None:
            raise UpstreamError(f"service '{service}' was never created: run `docker compose --profile serve up -d` once")
        c.start() if running else c.stop(timeout=10)
