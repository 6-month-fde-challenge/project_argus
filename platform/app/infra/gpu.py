"""Live GPU telemetry through NVML (read-only: no CUDA context, so it never takes GPU memory)."""
from __future__ import annotations


class GpuMonitor:
    def __init__(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml, self.available = pynvml, True
        except Exception:  # noqa: BLE001 - no driver / no GPU passed to the container
            self.nvml, self.available = None, False

    def snapshot(self) -> dict:
        if not self.available:
            return {"available": False, "gpus": []}
        n = self.nvml

        def soft(fn, default=None):
            """One metric failing (NVML 'Unknown' is common under WSL) must not hide the others."""
            try:
                return fn()
            except n.NVMLError:
                return default

        try:
            count = n.nvmlDeviceGetCount()
        except n.NVMLError as e:
            return {"available": False, "gpus": [], "error": str(e)}
        gpus = []
        for i in range(count):
            h = soft(lambda: n.nvmlDeviceGetHandleByIndex(i))
            if h is None:
                continue
            mem, name = soft(lambda: n.nvmlDeviceGetMemoryInfo(h)), soft(lambda: n.nvmlDeviceGetName(h), "GPU")
            util = soft(lambda: n.nvmlDeviceGetUtilizationRates(h))
            power = soft(lambda: n.nvmlDeviceGetPowerUsage(h))
            gpus.append({
                "index": i, "name": name.decode() if isinstance(name, bytes) else name,
                "memory_used_mb": round(mem.used / 2**20) if mem else None,
                "memory_total_mb": round(mem.total / 2**20) if mem else None,
                "utilization_pct": util.gpu if util else None,
                "temperature_c": soft(lambda: n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU)),
                "power_w": round(power / 1000, 1) if power is not None else None,
            })
        return {"available": True, "gpus": gpus}
