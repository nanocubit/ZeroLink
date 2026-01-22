"""
zerolink/ray_integration.py

Интеграция ZeroLink v2.0 с Ray для распределенных вычислений.
"""

import ray
import torch
import time
from typing import Dict, List, Optional
from dataclasses import dataclass

from .core.gpu.multi_gpu_pool import MultiDevicePool, PoolConfig
from .runtime.unified import UnifiedRuntime


@dataclass
class RayWorkerConfig:
    """Конфигурация для Ray Worker актора."""
    device_id: int
    pool_size_gb: int = 4
    num_gpus: int = 1  # Сколько GPU запрашивать у Ray


@ray.remote(num_gpus=1)
class RayGPUWorker:
    """
    Ray Actor для выполнения GPU задач с использованием ZeroLink пулов.
    """
    def __init__(self, config: RayWorkerConfig, main_pool_ref):
        self.config = config
        self.main_pool_ref = main_pool_ref  # Ссылка на глобальный пул от Main процесса
        
        # Инициализируем локальный GPU контекст
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{config.device_id}")
            torch.cuda.set_device(self.device)
        
        print(f"[RayGPUWorker] Initialized on GPU {config.device_id}")

    def execute_tensor_operation(self, operation: str, *args, **kwargs):
        """
        Выполняет GPU операцию с использованием пула памяти.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available in Ray worker")
        
        # В реальной интеграции здесь мы бы запросили память из пула
        # и создали тензоры, используя zero-copy аллокацию
        
        if operation == "matrix_multiply":
            size = args[0]
            # Создаем тензоры в GPU памяти
            a = torch.randn(size, size, device=self.device)
            b = torch.randn(size, size, device=self.device)
            result = torch.mm(a, b)
            return result.mean().item()
        
        elif operation == "convolution":
            batch_size, channels, height, width = args
            conv = torch.nn.Conv2d(channels, channels, 3, padding=1).to(self.device)
            x = torch.randn(batch_size, channels, height, width, device=self.device)
            result = conv(x)
            return result.mean().item()
        
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def get_gpu_memory_info(self):
        """Возвращает информацию о GPU памяти."""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(self.device)
            memory_reserved = torch.cuda.memory_reserved(self.device)
            return {
                "device_id": self.config.device_id,
                "allocated": memory_allocated,
                "reserved": memory_reserved,
                "utilization": torch.cuda.utilization(self.device) if hasattr(torch.cuda, 'utilization') else 0
            }
        return {"error": "CUDA not available"}


class RayClusterManager:
    """
    Менеджер для управления Ray кластером с ZeroLink интеграцией.
    """
    def __init__(self, device_configs: List[RayWorkerConfig]):
        self.device_configs = device_configs
        self.workers: Dict[int, RayGPUWorker] = {}
        self.global_pool = None
        
    def initialize_cluster(self):
        """Инициализирует Ray кластер и создает GPU workers."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # Создаем worker акторов для каждого GPU
        for config in self.device_configs:
            worker = RayGPUWorker.remote(config, None)  # Пока без ссылки на глобальный пул
            self.workers[config.device_id] = worker
            print(f"[RayClusterManager] Created worker for GPU {config.device_id}")
    
    def execute_distributed_task(self, operation: str, args_per_device: Dict[int, tuple]):
        """
        Выполняет распределенную задачу на нескольких GPU.
        """
        futures = []
        
        for device_id, args in args_per_device.items():
            if device_id in self.workers:
                future = self.workers[device_id].execute_tensor_operation.remote(operation, *args)
                futures.append(future)
        
        # Собираем результаты
        results = ray.get(futures)
        
        return results
    
    def get_cluster_status(self):
        """Возвращает статус всех GPU workers."""
        futures = []
        for device_id, worker in self.workers.items():
            future = worker.get_gpu_memory_info.remote()
            futures.append(future)
        
        worker_infos = ray.get(futures)
        
        return {
            "num_workers": len(self.workers),
            "worker_info": worker_infos,
            "ray_dashboard_url": ray.get_webui_url() if ray.is_initialized() else "Not initialized"
        }
    
    def shutdown(self):
        """Останавливает Ray кластер."""
        for worker in self.workers.values():
            ray.kill(worker)
        
        if ray.is_initialized():
            ray.shutdown()


def create_ray_unified_runtime(gpu_device_ids: List[int], pool_size_gb: int = 4):
    """
    Создает UnifiedRuntime с интеграцией Ray.
    """
    # Подготовим конфигурации для Multi-GPU пула
    pool_configs = [
        {
            'device_id': dev_id,
            'total_size_gb': pool_size_gb,
            'min_block_size': 2**21  # 2MB
        }
        for dev_id in gpu_device_ids
    ]
    
    # Создаем основной runtime
    runtime = UnifiedRuntime(
        gpu_configs=pool_configs,
        cpu_pool_size_mb=512,
        num_cpu_workers=4
    )
    
    # Создаем конфигурации для Ray workers
    ray_configs = [
        RayWorkerConfig(device_id=dev_id, pool_size_gb=pool_size_gb)
        for dev_id in gpu_device_ids
    ]
    
    # Создаем менеджер кластера
    cluster_manager = RayClusterManager(ray_configs)
    
    return runtime, cluster_manager


# Пример использования
def example_ray_integration():
    """
    Пример интеграции ZeroLink с Ray.
    """
    print("=== ZeroLink + Ray Integration Example ===")
    
    # Предположим, у нас есть 2 GPU
    gpu_ids = [0, 1] if torch.cuda.device_count() >= 2 else [0]
    pool_size = 2  # GB на GPU
    
    # Создаем интегрированную систему
    runtime, cluster_manager = create_ray_unified_runtime(gpu_ids, pool_size)
    
    try:
        # Инициализируем Ray кластер
        cluster_manager.initialize_cluster()
        
        # Выполняем распределенную задачу
        print("\nExecuting distributed matrix multiplication...")
        args_per_device = {
            gpu_id: (1024,)  # Размер матрицы 1024x1024
            for gpu_id in gpu_ids
        }
        
        start_time = time.time()
        results = cluster_manager.execute_distributed_task("matrix_multiply", args_per_device)
        end_time = time.time()
        
        print(f"Results: {results}")
        print(f"Execution time: {end_time - start_time:.2f}s")
        
        # Проверяем статус кластера
        status = cluster_manager.get_cluster_status()
        print(f"\nCluster status: {status}")
        
    finally:
        # Очищаем ресурсы
        cluster_manager.shutdown()
        runtime.stop()


if __name__ == "__main__":
    example_ray_integration()