"""
ZeroLink - Unified Zero-Copy Runtime for CPU/GPU Computing
Copyright (C) 2025 Slava Maltsev <nanotec@live.ru>
SPDX-License-Identifier: MIT

zerolink/runtime/unified.py

ZeroLinkRuntime: Высокоуровневый оркестратор для ZeroLink v2.0.
Объединяет Main Server (GPU Memory Management) и CPU Pool.
"""

import os
import signal
import sys
import json
import logging
import torch
import multiprocessing as mp
import threading
import time
from typing import Callable, Any, List

from ..monitoring.telemetry import (
    telemetry,
    monitor_pool_allocation,
    update_pool_usage,
    update_active_leases,
    observe_runtime_latency,
    record_alloc_failure,
)

# Импорты внутренних компонентов
from ..core.gpu.vmm_pool import DeviceMemoryPoolV2
from ..core.gpu.multi_gpu_pool import MultiDevicePool, PoolType
from ..core.gpu.pinned_region import PinnedMemoryManager
from ..server.main_server import MainIPCLeaseManager2P, MainServer

# ============================================================================
# Критическое исправление: CUDA Contexts и Fork
# ============================================================================

# CUDA контекст не переживает fork. Используем spawn.
# Это гарантия безопасности.
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass # Уже установлено или spawn недоступен (Windows)


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("zerolink.runtime")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger



class ZeroLinkRuntime:
    """
    ZeroLink Runtime для одновременной работы CPU и GPU.
    В режиме этого процесса запускается Main Server для управления памятью.
    """
    def __init__(
        self,
        socket_path: str = "/tmp/zerolink.sock",
        gpu_device_id: int = 0,
        gpu_pool_size_gb: int = 4,
        cpu_pool_name: str = "zerolink_cpu_pool",
        cpu_pool_size_mb: int = 512,
        num_cpu_workers: int = 4,
        enable_integrity_hash: bool = True
    ):
        self.socket_path = socket_path
        self._running = False
        self.logger = _build_logger()

        # 1. Инициализация GPU Subsystem (Main Process)
        self.logger.info(json.dumps({"event": "runtime_gpu_init_start", "device_id": gpu_device_id}))
        try:
            # Инициализация GPU пула
            self.gpu_pool = DeviceMemoryPoolV2(
                device_id=gpu_device_id,
                total_size_gb=gpu_pool_size_gb,
                pool_id=f"gpu_pool_{gpu_device_id}_{id(self)}"
            )
            self.logger.info(json.dumps({"event": "runtime_gpu_init_success", "device_id": gpu_device_id}))
        except Exception as e:
            record_alloc_failure("runtime", "gpu_pool_init")
            self.logger.exception(json.dumps({"event": "runtime_gpu_init_failed", "device_id": gpu_device_id, "error": str(e)}))
            # В реальном коде здесь была бы полная инициализация
            self.gpu_pool = None

        # Менеджер IPC аренд (управляет соединениями с воркерами)
        self.ipc_manager = MainIPCLeaseManager2P(
            pool=self.gpu_pool,
            enable_integrity_hash=enable_integrity_hash,
        ) if self.gpu_pool is not None else None
        self.main_server = MainServer(self.socket_path, self.ipc_manager) if self.ipc_manager else None

        # 2. Инициализация CPU Subsystem
        self.logger.info(json.dumps({"event": "runtime_cpu_init_start", "workers": num_cpu_workers}))
        self.cpu_pool_name = cpu_pool_name
        self.cpu_pool_size_mb = cpu_pool_size_mb
        self.num_cpu_workers = num_cpu_workers
        self.cpu_executor = mp.Pool(processes=num_cpu_workers)
        self.cpu_pool_manager = None # Заглушка для примера

        self.logger.info(json.dumps({"event": "runtime_ready", "socket_path": self.socket_path}))

    def start(self, block: bool = False):
        """
        Запускает сервер в отдельном потоке.
        
        Args:
            block: Если True, блокирует выполнение текущего потока.
                    Обычно False для интеграции с другими лупами.
        """
        if self._running:
            return

        self._running = True
        self._server_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._server_thread.start()
        self.logger.info(json.dumps({"event": "runtime_server_started"}))

        if block:
            self._server_thread.join()

    def stop(self):
        """Остановка всех подсистем."""
        stop_started = time.time()
        self.logger.info(json.dumps({"event": "runtime_shutdown_start"}))
        self._running = False

        # Остановка CPU воркеров
        self.cpu_executor.close()
        self.cpu_executor.join()

        # Остановка GPU пула и сервера
        if self.main_server:
            self.main_server.stop()

        # Остановка потока сервера (если он был запущен)
        if hasattr(self, '_server_thread') and self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=5.0)

        observe_runtime_latency("runtime_shutdown", time.time() - stop_started)
        self.logger.info(json.dumps({"event": "runtime_shutdown_complete"}))

    def _run_event_loop(self):
        """Внутренний метод для потока сервера."""
        if not self.main_server:
            self.logger.warning(json.dumps({"event": "runtime_server_skipped", "reason": "gpu_pool_unavailable"}))
            while self._running:
                time.sleep(0.1)
            return

        try:
            self.main_server.start()
        except Exception as e:
            record_alloc_failure("runtime", "main_server_start")
            self.logger.exception(json.dumps({"event": "runtime_server_error", "error": str(e)}))
            self._running = False

    @monitor_pool_allocation
    def execute_gpu(self, func, *args, **kwargs):
        """
        Выполняет функцию на GPU.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        # В реальном проекте `func` выполнялась бы с тензорами,
        # аллоцированными через TorchAwareDeviceMemoryPoolV2.
        # Здесь мы просто эмулируем выполнение на нужном устройстве.
        device = torch.device(f"cuda:0") # Заглушка device_id

        with torch.cuda.device(device):
            # Поддержка CUDA Graphs для оптимизации
            if hasattr(torch.cuda, 'graph') and kwargs.get('use_cuda_graph', False):
                # Создаем CUDA graph для повторяющихся вычислений
                stream = torch.cuda.Stream()
                with torch.cuda.graph(stream=stream):
                    result = func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Обновляем метрики использования пула
            if self.gpu_pool:
                used_bytes = sum(a.size for a in self.gpu_pool.allocations.values())
                update_pool_usage(used_bytes)

            return result

    def execute_cpu(self, func: Callable, *args, **kwargs):
        """
        Выполняет функцию в одном из CPU воркеров (multiprocessing).
        """
        return self.cpu_executor.apply(func, args, kwargs)

    def map_cpu(self, func: Callable, iterable: List[Any]) -> List[Any]:
        """
        Применяет функцию к итерируемому объекту параллельно на CPU.
        """
        return self.cpu_executor.map(func, iterable)

    # -------------------------------------------------------------------------
    # Helper методы для advanced usage
    # -------------------------------------------------------------------------

    def get_pool_stats(self) -> dict:
        """Возвращает статистику GPU и CPU пулов."""
        gpu_stats = {
            "pool_id": None,
            "used_gb": 0.0,
            "allocations": 0,
            "total_gb": 0.0,
        }

        if self.gpu_pool is not None:
            used_bytes = sum(a.size for a in self.gpu_pool.allocations.values())
            gpu_stats = {
                "pool_id": self.gpu_pool.pool_id,
                "used_gb": used_bytes / (1024**3),
                "allocations": len(self.gpu_pool.allocations),
                "total_gb": self.gpu_pool.total_size / (1024**3),
            }
            update_pool_usage(int(used_bytes))

        stats = {
            "gpu": gpu_stats,
            "cpu": {
                "pool_name": self.cpu_pool_name,
                "total_mb": self.cpu_pool_size_mb,
                "workers": self.cpu_executor._processes if hasattr(self.cpu_executor, '_processes') else self.num_cpu_workers,
            },
        }
        return stats

    def get_ipc_lease_status(self) -> dict:
        """Возвращает статус IPC аренд."""
        if not self.ipc_manager:
            return {"pending": 0, "active": 0, "workers_connected": 0}

        with self.ipc_manager.lock:
            pending = sum(1 for l in self.ipc_manager.leases.values() if l.status == "PENDING")
            active = sum(1 for l in self.ipc_manager.leases.values() if l.status == "ACTIVE")
            workers_connected = len(self.ipc_manager.workers)

        update_active_leases(active)
        return {
            "pending": pending,
            "active": active,
            "workers_connected": workers_connected,
        }

    # -------------------------------------------------------------------------
    # Signal Handlers (Graceful Shutdown)
    # -------------------------------------------------------------------------

    def _signal_handler(self, sig, frame):
        """Обрабатывает SIGINT/SIGTERM для корректного завершения."""
        self.logger.warning(json.dumps({"event": "runtime_signal", "signal": int(sig)}))
        self.stop()
        sys.exit(0)

    def __enter__(self):
        """Support for `with UnifiedRuntime() as rt: ...`"""
        self.start(block=False) # Не блокируем старт, если мы используем этот класс как контекст
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Гарантирует stop при выходе из контекста."""
        self.stop()

# Создаем псевдоним для обратной совместимости
UnifiedRuntime = ZeroLinkRuntime