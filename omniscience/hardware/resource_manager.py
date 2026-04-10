"""Hardware Resource Manager — "Total Saturation" mode.

Monitors and manages CPU, GPU, and RAM utilization to maximize
throughput during cryptanalysis. Provides real-time metrics for
the GUI monitoring dashboard.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import psutil

from omniscience.core.config import HardwareConfig

log = logging.getLogger(__name__)


@dataclass
class ResourceSnapshot:
    """Point-in-time resource utilization snapshot."""

    timestamp: float = 0.0
    cpu_percent: float = 0.0
    cpu_per_core: list[float] = field(default_factory=list)
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    ram_percent: float = 0.0
    gpu_available: bool = False
    gpu_name: str = ""
    gpu_utilization: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_temperature: float = 0.0


class ResourceManager:
    """Centralized hardware resource manager.

    Provides:
      - Real-time monitoring (poll-based snapshots)
      - CPU worker pool sizing
      - GPU availability detection and memory management
      - Callback system for GUI gauges
    """

    def __init__(self, config: HardwareConfig | None = None):
        self.config = config or HardwareConfig()
        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._latest_snapshot = ResourceSnapshot()
        self._callbacks: list[Callable[[ResourceSnapshot], None]] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    @property
    def cpu_count(self) -> int:
        if self.config.max_cpu_workers > 0:
            return self.config.max_cpu_workers
        return multiprocessing.cpu_count() or 1

    @property
    def gpu_available(self) -> bool:
        if not self.config.use_gpu:
            return False
        return self._check_gpu()

    def snapshot(self) -> ResourceSnapshot:
        """Take an immediate resource snapshot."""
        snap = ResourceSnapshot(timestamp=time.time())

        # CPU
        snap.cpu_percent = psutil.cpu_percent(interval=0.1)
        snap.cpu_per_core = psutil.cpu_percent(percpu=True)

        # RAM
        mem = psutil.virtual_memory()
        snap.ram_total_gb = mem.total / (1024**3)
        snap.ram_used_gb = mem.used / (1024**3)
        snap.ram_percent = mem.percent

        # GPU (nvidia-smi or pynvml)
        if self.config.use_gpu:
            self._fill_gpu_stats(snap)

        with self._lock:
            self._latest_snapshot = snap
        return snap

    @property
    def latest(self) -> ResourceSnapshot:
        with self._lock:
            return self._latest_snapshot

    def start_monitoring(self, interval: float = 1.0) -> None:
        """Start background monitoring thread."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(interval,), daemon=True
        )
        self._monitor_thread.start()
        log.info("Resource monitor started (interval=%.1fs)", interval)

    def stop_monitoring(self) -> None:
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)

    def on_update(self, callback: Callable[[ResourceSnapshot], None]) -> None:
        """Register a callback invoked on each monitoring tick."""
        self._callbacks.append(callback)

    def configure_gpu(self) -> None:
        """Set GPU memory fraction and device."""
        if not self.gpu_available:
            return
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(
                    self.config.gpu_memory_fraction, self.config.gpu_device_id
                )
                log.info(
                    "GPU memory fraction set to %.0f%% on device %d",
                    self.config.gpu_memory_fraction * 100,
                    self.config.gpu_device_id,
                )
        except Exception as exc:
            log.warning("Failed to configure GPU memory: %s", exc)

    def summary(self) -> str:
        snap = self.snapshot()
        lines = [
            f"CPU: {snap.cpu_percent:.1f}% ({len(snap.cpu_per_core)} cores)",
            f"RAM: {snap.ram_used_gb:.1f}/{snap.ram_total_gb:.1f} GB ({snap.ram_percent:.1f}%)",
        ]
        if snap.gpu_available:
            lines.append(
                f"GPU: {snap.gpu_name} — {snap.gpu_utilization:.0f}% util, "
                f"{snap.gpu_memory_used_mb:.0f}/{snap.gpu_memory_total_mb:.0f} MB, "
                f"{snap.gpu_temperature:.0f}°C"
            )
        else:
            lines.append("GPU: not available")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Internals                                                          #
    # ------------------------------------------------------------------ #

    def _monitor_loop(self, interval: float) -> None:
        while not self._stop_event.is_set():
            snap = self.snapshot()
            for cb in self._callbacks:
                try:
                    cb(snap)
                except Exception:
                    log.exception("Monitor callback error")
            self._stop_event.wait(interval)

    @staticmethod
    def _check_gpu() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            pass
        # Fallback: check for nvidia-smi
        return os.path.exists("/usr/bin/nvidia-smi") or os.path.exists("/usr/local/bin/nvidia-smi")

    @staticmethod
    def _fill_gpu_stats(snap: ResourceSnapshot) -> None:
        try:
            import torch

            if not torch.cuda.is_available():
                return
            snap.gpu_available = True
            snap.gpu_name = torch.cuda.get_device_name(0)
            mem = torch.cuda.mem_get_info(0)
            snap.gpu_memory_total_mb = mem[1] / (1024**2)
            snap.gpu_memory_used_mb = (mem[1] - mem[0]) / (1024**2)
        except Exception:
            pass

        # Try pynvml for utilization and temperature
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            snap.gpu_utilization = util.gpu
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            snap.gpu_temperature = temp
            pynvml.nvmlShutdown()
        except Exception:
            pass
