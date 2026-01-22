"""
zerolink/runtime/unified.py

UnifiedRuntime: Высокоуровневый оркестратор для ZeroLink v2.0.
Объединяет Main Server (GPU Memory Management) и CPU Pool.
"""

import os
import signal
import sys
import torch
import multiprocessing as mp
import threading
import time
from typing import Callable, Any, List

from ..monitoring.telemetry import telemetry, monitor_pool_allocation, update_pool_usage, update_active_leases

# Импорты внутренних компонентов
from ..core.gpu.vmm_pool import DeviceMemoryPoolV2
from ..core.gpu.multi_gpu_pool import MultiDevicePool, PoolType
from ..core.gpu.pinned_region import PinnedMemoryManager
from ..server.main_server import MainIPCLeaseManager2P

# ============================================================================
# Критическое исправление: CUDA Contexts и Fork
# ============================================================================

# CUDA контекст не переживает fork. Используем spawn.
# Это гарантия безопасности.
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass # Уже установлено или spawn недоступен (Windows)

class UnifiedRuntime:
    """
    Unified Runtime для одновременной работы CPU и GPU.
    В режиме этого процесса запускается Main Server для управления памятью.
    """
    def __init__(
        self,
        socket_path: str = "/tmp/pynexus.sock",
        gpu_device_id: int = 0,
        gpu_pool_size_gb: int = 4,
        cpu_pool_name: str = "pynexus_cpu_pool",
        cpu_pool_size_mb: int = 512,
        num_cpu_workers: int = 4,
        enable_integrity_hash: bool = True
    ):
        self.socket_path = socket_path
        self._running = False

        # 1. Инициализация GPU Subsystem (Main Process)
        print(f"[Runtime] Initializing GPU Subsystem on device {gpu_device_id}...")
        try:
            # Инициализация GPU пула
            self.gpu_pool = DeviceMemoryPoolV2(
                device_id=gpu_device_id,
                total_size_gb=gpu_pool_size_gb,
                pool_id=f"gpu_pool_{gpu_device_id}_{id(self)}"
            )
            print(f"[Runtime] GPU Pool initialized on device {gpu_device_id}")
        except Exception as e:
            print(f"[Runtime] CRITICAL: Failed to init GPU Pool: {e}")
            # В реальном коде здесь была бы полная инициализация
            self.gpu_pool = None

        # Менеджер IPC аренд (управляет соединениями с воркерами)
        self.ipc_manager = None # Будет инициализирован в `start()`

        # 2. Инициализация CPU Subsystem
        print(f"[Runtime] Initializing CPU Subsystem ({num_cpu_workers} workers)...")
        self.cpu_pool_name = cpu_pool_name
        self.cpu_pool_size_mb = cpu_pool_size_mb
        self.num_cpu_workers = num_cpu_workers
        self.cpu_executor = mp.Pool(processes=num_cpu_workers)
        self.cpu_pool_manager = None # Заглушка для примера

        print(f"✅ UnifiedRuntime Ready")

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
        print("[Runtime] Server started (Thread)")

        if block:
            self._server_thread.join()

    def stop(self):
        """Остановка всех подсистем."""
        print("[Runtime] Shutting down...")
        self._running = False

        # Остановка CPU воркеров
        self.cpu_executor.close()
        self.cpu_executor.join()

        # Остановка GPU пула и сервера
        if self.ipc_manager:
            # В реальном коде тут был бы вызов shutdown
            pass

        # Остановка потока сервера (если он был запущен)
        if hasattr(self, '_server_thread') and self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=5.0)

        print("[Runtime] Shutdown complete.")

    def _run_event_loop(self):
        """Внутренний метод для потока сервера."""
        while self._running:
            # Здесь была бы логика сокета и вызов менеджера аренд
            time.sleep(1.0)
            # ... socket.accept() ...
            # ... self.ipc_manager.handle_worker() ...

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
                # В реальном коде тут будет получение статистики из пула
                # telemetry.update_pool_usage(used_bytes)
                pass

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
        # В реальном коде здесь были бы реальные метрики
        stats = {
            "gpu": {"pool_id": "mock_gpu_pool", "used_gb": 2.0},
            "cpu": {
                "pool_name": self.cpu_pool_name,
                "total_mb": self.cpu_pool_size_mb,
                "workers": self.cpu_executor._processes if hasattr(self.cpu_executor, '_processes') else self.num_cpu_workers
            }
        }

        # Обновляем метрики телеметрии
        if 'gpu' in stats and 'used_gb' in stats['gpu']:
            used_bytes = stats['gpu']['used_gb'] * 1024**3  # Переводим в байты
            update_pool_usage(int(used_bytes))

        return stats

    def get_ipc_lease_status(self) -> dict:
        """Возвращает статус IPC аренд."""
        # В реальном коде делегируется менеджеру аренд
        return {
            "pending": 0,
            "active": 0,
            "workers_connected": 0
        }

    # -------------------------------------------------------------------------
    # Signal Handlers (Graceful Shutdown)
    # -------------------------------------------------------------------------

    def _signal_handler(self, sig, frame):
        """Обрабатывает SIGINT/SIGTERM для корректного завершения."""
        print(f"\n[Runtime] Caught signal {sig}, cleaning up...")
        self.stop()
        sys.exit(0)

    def __enter__(self):
        """Support for `with UnifiedRuntime() as rt: ...`"""
        self.start(block=False) # Не блокируем старт, если мы используем этот класс как контекст
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Гарантирует stop при выходе из контекста."""
        self.stop()