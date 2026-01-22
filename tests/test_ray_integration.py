"""
tests/test_ray_integration.py

Тесты для интеграции ZeroLink с Ray.
"""

import pytest
import torch
import time

# Убедимся, что ZeroLink доступен
from zerolink.ray_server import RayServer, GlobalPoolConfig
from zerolink.ray_integration import RayClusterManager, RayWorkerConfig


def test_ray_server_initialization():
    """Тест инициализации Ray Server."""
    # Пропускаем тест, если CUDA недоступна
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping Ray integration test")
    
    # Ограничиваем количество GPU для теста
    num_gpus = min(torch.cuda.device_count(), 1)  # Используем только 1 GPU для теста
    if num_gpus == 0:
        pytest.skip("No GPUs available for testing")
    
    device_ids = list(range(num_gpus))
    
    # Создаем конфигурацию сервера
    config = GlobalPoolConfig(
        device_ids=device_ids,
        pool_size_gb=1,  # Маленький пул для теста
        enable_integrity_checks=True
    )
    
    # Создаем сервер
    server = RayServer(config)
    
    try:
        # Инициализируем сервер
        server.start()
        
        # Проверяем, что сервер запустился
        stats = server.get_cluster_stats()
        assert isinstance(stats, dict)
        
        print(f"Ray Server stats: {stats}")
        
    finally:
        server.shutdown()


def test_ray_cluster_manager():
    """Тест Ray Cluster Manager."""
    # Пропускаем тест, если CUDA недоступна
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping Ray integration test")
    
    num_gpus = min(torch.cuda.device_count(), 1)  # Используем только 1 GPU для теста
    if num_gpus == 0:
        pytest.skip("No GPUs available for testing")
    
    # Создаем конфигурации для воркеров
    ray_configs = [
        RayWorkerConfig(device_id=i, pool_size_gb=1)
        for i in range(num_gpus)
    ]
    
    # Создаем менеджер кластера
    cluster_manager = RayClusterManager(ray_configs)
    
    try:
        # Инициализируем кластер
        cluster_manager.initialize_cluster()
        
        # Проверяем статус
        status = cluster_manager.get_cluster_status()
        assert "num_workers" in status
        assert status["num_workers"] == num_gpus
        
        print(f"Ray Cluster status: {status}")
        
    finally:
        cluster_manager.shutdown()


def test_ray_distributed_operations():
    """Тест распределенных операций через Ray."""
    # Пропускаем тест, если CUDA недоступна
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping Ray integration test")
    
    num_gpus = min(torch.cuda.device_count(), 1)  # Используем только 1 GPU для теста
    if num_gpus == 0:
        pytest.skip("No GPUs available for testing")
    
    device_ids = list(range(num_gpus))
    
    # Создаем конфигурацию сервера
    config = GlobalPoolConfig(
        device_ids=device_ids,
        pool_size_gb=1,
        enable_integrity_checks=True
    )
    
    # Создаем сервер
    server = RayServer(config)
    
    try:
        # Инициализируем сервер
        server.start()
        server.create_workers(num_workers_per_gpu=1)
        
        # Определяем операции для выполнения
        operations = [
            {"device_id": device_ids[0], "size_mb": 50, "operation": "matrix_multiply"},
            {"device_id": device_ids[0], "size_mb": 30, "operation": "memory_fill"},
        ]
        
        # Выполняем операции
        start_time = time.time()
        results = server.execute_distributed_operations(operations)
        end_time = time.time()
        
        # Проверяем результаты
        assert len(results) == len(operations)
        for result in results:
            assert result["success"] is True
            assert "result" in result
        
        print(f"Distributed operations took {end_time - start_time:.2f}s")
        print(f"Results: {results}")
        
    finally:
        server.shutdown()


if __name__ == "__main__":
    # Запуск тестов вручную
    test_ray_server_initialization()
    test_ray_cluster_manager()
    test_ray_distributed_operations()
    print("All Ray integration tests passed!")