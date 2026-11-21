"""GPU telemetry degrades per metric: one NVML call failing (common under WSL) must not fail the snapshot."""
from __future__ import annotations

from types import SimpleNamespace

from app.infra.gpu import GpuMonitor


class FakeNvml:
    NVML_TEMPERATURE_GPU = 0

    class NVMLError(Exception):
        pass

    def __init__(self, broken=()):
        self.broken = set(broken)

    def _call(self, name, value):
        if name in self.broken:
            raise self.NVMLError("Unknown Error")
        return value

    def nvmlDeviceGetCount(self): return self._call("count", 1)
    def nvmlDeviceGetHandleByIndex(self, i): return self._call("handle", object())
    def nvmlDeviceGetMemoryInfo(self, h): return self._call("memory", SimpleNamespace(used=1024 * 2**20, total=4096 * 2**20))
    def nvmlDeviceGetName(self, h): return self._call("name", b"Fake GPU")
    def nvmlDeviceGetUtilizationRates(self, h): return self._call("util", SimpleNamespace(gpu=42))
    def nvmlDeviceGetPowerUsage(self, h): return self._call("power", 35500)
    def nvmlDeviceGetTemperature(self, h, kind): return self._call("temp", 61)


def monitor(nvml):
    m = GpuMonitor.__new__(GpuMonitor)
    m.nvml, m.available = nvml, True
    return m


def test_full_snapshot():
    g = monitor(FakeNvml()).snapshot()["gpus"][0]
    assert g == {"index": 0, "name": "Fake GPU", "memory_used_mb": 1024, "memory_total_mb": 4096,
                 "utilization_pct": 42, "temperature_c": 61, "power_w": 35.5}


def test_one_failing_metric_is_null_not_an_error():
    g = monitor(FakeNvml(broken={"util", "power"})).snapshot()["gpus"][0]
    assert g["utilization_pct"] is None and g["power_w"] is None and g["memory_used_mb"] == 1024


def test_device_count_failure_reports_unavailable():
    snap = monitor(FakeNvml(broken={"count"})).snapshot()
    assert snap["available"] is False and "Unknown" in snap["error"]


def test_no_nvml_at_all():
    m = GpuMonitor.__new__(GpuMonitor)
    m.nvml, m.available = None, False
    assert m.snapshot() == {"available": False, "gpus": []}
