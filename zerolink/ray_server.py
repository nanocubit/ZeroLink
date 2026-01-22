"""
ZeroLink - Unified Zero-Copy Runtime for CPU/GPU Computing
Copyright (C) 2025 Slava Maltsev <nanotec@live.ru>
SPDX-License-Identifier: MIT

zerolink/ray_server.py

Ray Server: Главный процесс, который держит MultiDevicePool и управляет арендами.
Использует Ray Namespace для глобального состояния пула.
"""

import ray
import torch
import time
from typing import List, Dict
from dataclasses import dataclass

from .core.gpu.multi_gpu_pool import MultiDevicePool, PoolConfig
from .core.gpu.vmm_pool import DeviceAllocation


@dataclass
class GlobalPoolConfig:
    """Глобальная конфигурация пула для Ray кластера."""
    device_ids: List[int]
    pool_size_gb: int = 4
    enable_integrity_checks: bool = True


@ray.remote
class GlobalMemoryPoolActor:
    """
    Ray Actor, который управляет глобальным MultiDevicePool.
    Доступен для всех worker процессов в Ray кластере.
    """
    def __init__(self, config: GlobalPoolConfig):
        self.config = config
        self.multi_pool = None
        self.active_allocations: Dict[str, DeviceAllocation] = {}
        
        # Создаем MultiDevicePool
        pool_configs = [
            PoolConfig(
                device_id=dev_id,
                total_size_gb=config.pool_size_gb,
                min_block_size=2**21  # 2MB
            )
            for dev_id in config.device_ids
        ]
        
        self.multi_pool = MultiDevicePool(pool_configs)
        print(f"[GlobalMemoryPoolActor] Initialized pool for devices: {config.device_ids}")
    
    def allocate_memory(self, device_id: int, size_bytes: int) -> str:
        """
        Выделяет GPU память на указанном устройстве.
        Возвращает allocation_id для дальнейшего использования.
        """
        if device_id not in self.config.device_ids:
            raise ValueError(f"Device {device_id} not managed by this pool")
        
        allocation = self.multi_pool.allocate(size_bytes, device_id)
        if allocation is None:
            raise RuntimeError(f"Failed to allocate {size_bytes} bytes on device {device_id}")
        
        # Генерируем уникальный ID для аллокации
        alloc_id = f"alloc_{device_id}_{int(time.time()*1000)}_{size_bytes}"
        self.active_allocations[alloc_id] = allocation
        
        print(f"[GlobalMemoryPoolActor] Allocated {size_bytes} bytes on GPU {device_id}, ID: {alloc_id}")
        return alloc_id
    
    def get_allocation_info(self, alloc_id: str) -> Dict:
        """
        Возвращает информацию об аллокации.
        """
        if alloc_id not in self.active_allocations:
            raise KeyError(f"Allocation {alloc_id} not found")
        
        allocation = self.active_allocations[alloc_id]
        return {
            "alloc_id": alloc_id,
            "device_id": allocation.device_id,
            "size": allocation.size,
            "va_addr": allocation.va_addr
        }
    
    def release_memory(self, alloc_id: str):
        """
        Освобождает GPU память.
        """
        if alloc_id not in self.active_allocations:
            raise KeyError(f"Allocation {alloc_id} not found")
        
        allocation = self.active_allocations[alloc_id]
        # Освобождаем память через пул
        self.multi_pool.pools[allocation.device_id].free(allocation)
        
        del self.active_allocations[alloc_id]
        print(f"[GlobalMemoryPoolActor] Released allocation {alloc_id}")
    
    def get_pool_stats(self) -> Dict:
        """
        Возвращает статистику по всем пулам.
        """
        return self.multi_pool.get_stats()


@ray.remote(num_gpus=1)
class RayWorkerWithPoolAccess:
    """
    Ray Worker Actor, который может запрашивать память из глобального пула.
    """
    def __init__(self, device_id: int, global_pool_actor):
        self.device_id = device_id
        self.global_pool_actor = global_pool_actor
        
        # Устанавливаем CUDA устройство
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)
        
        print(f"[RayWorkerWithPoolAccess] Initialized on GPU {device_id}")
    
    def execute_with_allocated_memory(self, size_mb: int, operation: str) -> Dict:
        """
        Выполняет операцию с использованием выделенной GPU памяти.
        """
        size_bytes = size_mb * 1024 * 1024
        
        # Запрашиваем память из глобального пула
        alloc_id = ray.get(self.global_pool_actor.allocate_memory.remote(self.device_id, size_bytes))
        
        try:
            # Получаем информацию об аллокации
            alloc_info = ray.get(self.global_pool_actor.get_allocation_info.remote(alloc_id))
            
            # Выполняем операцию с использованием выделенной памяти
            if operation == "matrix_multiply":
                # Создаем тензоры размером с выделенную память (или часть ее)
                tensor_size = int((size_bytes // 4) ** 0.5)  # Приблизительный размер квадратной матрицы
                if tensor_size > 2048:  # Ограничиваем размер для теста
                    tensor_size = 2048
                
                # Создаем тензоры на GPU
                a = torch.randn(tensor_size, tensor_size, device=f"cuda:{self.device_id}")
                b = torch.randn(tensor_size, tensor_size, device=f"cuda:{self.device_id}")
                
                # Выполняем операцию
                result = torch.mm(a, b)
                
                operation_result = result.mean().item()
                
            elif operation == "memory_fill":
                # Просто заполняем память значениями
                tensor_size = size_bytes // 4  # Количество float32 элементов
                if tensor_size > 1024*1024:  # Ограничиваем для теста
                    tensor_size = 1024*1024
                
                data = torch.full((tensor_size,), 42.0, device=f"cuda:{self.device_id}")
                operation_result = data.sum().item()
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            return {
                "success": True,
                "operation": operation,
                "device_id": self.device_id,
                "allocation_id": alloc_id,
                "result": operation_result,
                "allocation_info": alloc_info
            }
            
        finally:
            # Всегда освобождаем память
            self.global_pool_actor.release_memory.remote(alloc_id)


class RayServer:
    """
    Основной Ray Server, который управляет глобальным пулом памяти и worker акторами.
    """
    def __init__(self, config: GlobalPoolConfig):
        self.config = config
        self.global_pool_actor = None
        self.workers: List[RayWorkerWithPoolAccess] = []
        
    def start(self):
        """Запускает Ray Server и инициализирует глобальный пул."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # Создаем глобальный пул памяти как Ray Actor
        self.global_pool_actor = GlobalMemoryPoolActor.remote(self.config)
        print(f"[RayServer] Started with global pool for devices: {self.config.device_ids}")
        
    def create_workers(self, num_workers_per_gpu: int = 1):
        """Создает worker акторов для каждого GPU."""
        for device_id in self.config.device_ids:
            for i in range(num_workers_per_gpu):
                worker = RayWorkerWithPoolAccess.remote(device_id, self.global_pool_actor)
                self.workers.append(worker)
                print(f"[RayServer] Created worker {i+1} for GPU {device_id}")
    
    def execute_distributed_operations(self, operations: List[Dict]) -> List[Dict]:
        """
        Выполняет распределенные операции на разных GPU.
        
        Args:
            operations: Список операций в формате:
                       [{"device_id": int, "size_mb": int, "operation": str}, ...]
        """
        futures = []
        
        for op in operations:
            device_id = op["device_id"]
            size_mb = op["size_mb"]
            operation = op["operation"]
            
            # Находим первого доступного worker'а для этого GPU
            worker = None
            for w in self.workers:
                # В реальности нужно более точно сопоставлять worker и GPU
                # Для простоты берем первого
                worker = w
                break
            
            if worker:
                future = worker.execute_with_allocated_memory.remote(size_mb, operation)
                futures.append(future)
        
        # Собираем результаты
        results = ray.get(futures)
        return results
    
    def get_cluster_stats(self) -> Dict:
        """Получает статистику кластера."""
        if self.global_pool_actor:
            return ray.get(self.global_pool_actor.get_pool_stats.remote())
        return {}
    
    def shutdown(self):
        """Останавливает Ray Server."""
        for worker in self.workers:
            ray.kill(worker)
        
        if self.global_pool_actor:
            ray.kill(self.global_pool_actor)
        
        if ray.is_initialized():
            ray.shutdown()


def run_ray_server_example():
    """
    Пример запуска Ray Server с ZeroLink интеграцией.
    """
    print("=== ZeroLink Ray Server Example ===")
    
    # Определяем доступные GPU
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs found, using CPU-only mode for demonstration")
        device_ids = [0]  # Используем условный ID
    else:
        device_ids = list(range(min(num_gpus, 2)))  # Используем до 2 GPU для примера
    
    # Создаем конфигурацию сервера
    server_config = GlobalPoolConfig(
        device_ids=device_ids,
        pool_size_gb=2,  # 2GB на GPU для примера
        enable_integrity_checks=True
    )
    
    # Создаем и запускаем сервер
    server = RayServer(server_config)
    
    try:
        server.start()
        server.create_workers(num_workers_per_gpu=1)
        
        print(f"\nServer stats: {server.get_cluster_stats()}")
        
        # Определяем операции для выполнения
        operations = [
            {"device_id": device_ids[0], "size_mb": 100, "operation": "matrix_multiply"},
            {"device_id": device_ids[0], "size_mb": 50, "operation": "memory_fill"},
        ]
        
        if len(device_ids) > 1:
            operations.append({"device_id": device_ids[1], "size_mb": 80, "operation": "matrix_multiply"})
        
        # Выполняем операции
        print(f"\nExecuting operations: {operations}")
        start_time = time.time()
        results = server.execute_distributed_operations(operations)
        end_time = time.time()
        
        print(f"\nResults: {results}")
        print(f"Total execution time: {end_time - start_time:.2f}s")
        
        print(f"\nFinal server stats: {server.get_cluster_stats()}")
        
    finally:
        server.shutdown()
        print("\nRay Server shut down.")


if __name__ == "__main__":
    run_ray_server_example()