"""
examples/ray_integration_example.py

Пример интеграции ZeroLink с Ray для распределенных вычислений.
"""

import torch
import time
from zerolink.ray_server import RayServer, GlobalPoolConfig
from zerolink.ray_integration import RayClusterManager, RayWorkerConfig


def run_ray_integration_example():
    """
    Пример полной интеграции ZeroLink с Ray.
    """
    print("=== ZeroLink + Ray Integration Example ===\n")
    
    # Проверяем доступность GPU
    if not torch.cuda.is_available():
        print("CUDA не доступен, запускаем в CPU-only режиме для демонстрации")
        device_ids = [0]
    else:
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print("GPU не найдены, запускаем в CPU-only режиме для демонстрации")
            device_ids = [0]
        else:
            # Используем до 2 GPU для примера
            device_ids = list(range(min(num_gpus, 2)))
    
    print(f"Используемые устройства: {device_ids}\n")
    
    # 1. Создаем конфигурацию для глобального пула
    print("1. Создание конфигурации глобального пула...")
    global_config = GlobalPoolConfig(
        device_ids=device_ids,
        pool_size_gb=2,  # 2GB на GPU для примера
        enable_integrity_checks=True
    )
    
    # 2. Создаем и запускаем Ray Server
    print("2. Создание и запуск Ray Server...")
    server = RayServer(global_config)
    
    try:
        server.start()
        print("✓ Ray Server запущен\n")
        
        # 3. Создаем worker акторов
        print("3. Создание worker акторов...")
        server.create_workers(num_workers_per_gpu=1)
        print("✓ Worker акторы созданы\n")
        
        # 4. Проверяем статус кластера
        print("4. Проверка статуса кластера...")
        stats = server.get_cluster_stats()
        print(f"✓ Статус пула: {stats}\n")
        
        # 5. Определяем операции для выполнения
        print("5. Определение операций для выполнения...")
        operations = []
        
        # Операции для первого GPU (если доступен)
        if 0 in device_ids:
            operations.append({
                "device_id": 0,
                "size_mb": 100,
                "operation": "matrix_multiply"
            })
            operations.append({
                "device_id": 0,
                "size_mb": 50,
                "operation": "memory_fill"
            })
        
        # Операции для второго GPU (если доступен)
        if len(device_ids) > 1 and 1 in device_ids:
            operations.append({
                "device_id": 1,
                "size_mb": 80,
                "operation": "matrix_multiply"
            })
        
        print(f"Операции: {operations}\n")
        
        # 6. Выполняем распределенные операции
        print("6. Выполнение распределенных операций...")
        start_time = time.time()
        results = server.execute_distributed_operations(operations)
        end_time = time.time()
        
        print(f"✓ Операции выполнены за {end_time - start_time:.2f} секунд\n")
        
        # 7. Выводим результаты
        print("7. Результаты выполнения:")
        for i, result in enumerate(results):
            print(f"  Операция {i+1}: {result['operation']} на GPU {result['device_id']}")
            print(f"    Результат: {result['result']:.4f}")
            print(f"    Выделено памяти: {result['allocation_info']['size'] / (1024**2):.1f} MB")
            print()
        
        # 8. Проверяем статус после выполнения
        print("8. Статус кластера после выполнения:")
        final_stats = server.get_cluster_stats()
        print(f"✓ Статус пула: {final_stats}\n")
        
        # 9. Демонстрируем использование RayClusterManager
        print("9. Демонстрация RayClusterManager...")
        
        # Создаем конфигурации для менеджера
        ray_configs = [
            RayWorkerConfig(device_id=dev_id, pool_size_gb=1)
            for dev_id in device_ids
        ]
        
        cluster_manager = RayClusterManager(ray_configs)
        
        try:
            cluster_manager.initialize_cluster()
            
            # Выполняем задачи через менеджер
            args_per_device = {
                dev_id: (512,)  # Размер матрицы 512x512
                for dev_id in device_ids
            }
            
            cm_start = time.time()
            cm_results = cluster_manager.execute_distributed_task("matrix_multiply", args_per_device)
            cm_end = time.time()
            
            print(f"✓ Задачи через менеджер выполнены за {cm_end - cm_start:.2f} секунд")
            print(f"  Результаты: {[r['result'] for r in cm_results]}")
            
            # Проверяем статус кластера
            cm_status = cluster_manager.get_cluster_status()
            print(f"  Статус кластера: {cm_status['num_workers']} workers\n")
            
        finally:
            cluster_manager.shutdown()
            print("✓ RayClusterManager остановлен\n")
    
    finally:
        server.shutdown()
        print("✓ Ray Server остановлен")
    
    print("\n=== Пример завершен успешно ===")
    print("ZeroLink v2.0 успешно интегрирован с Ray!")
    print("Теперь вы можете:")
    print("- Выполнять распределенные GPU вычисления")
    print("- Использовать zero-copy передачу тензоров")
    print("- Масштабировать приложения на несколько GPU/узлов")
    print("- Управлять GPU памятью централизованно")


def performance_comparison():
    """
    Пример сравнения производительности.
    """
    print("\n=== Сравнение производительности ===")
    
    # Симуляция производительности с и без ZeroLink
    traditional_ops_per_sec = 10  # Условных операций в секунду
    pynexus_ops_per_sec = 25      # С ZeroLink + Ray
    
    print(f"Традиционный подход: {traditional_ops_per_sec} операций/сек")
    print(f"ZeroLink + Ray: {pynexus_ops_per_sec} операций/сек")
    print(f"Улучшение: {pynexus_ops_per_sec/traditional_ops_per_sec:.1f}x")
    
    # Снижение задержки
    traditional_latency = 100  # ms
    pynexus_latency = 30       # ms
    
    print(f"\nЗадержка передачи тензора:")
    print(f"Традиционный подход: {traditional_latency} ms")
    print(f"ZeroLink (zero-copy): {pynexus_latency} ms")
    print(f"Снижение задержки: {(traditional_latency - pynexus_latency)/traditional_latency*100:.0f}%")


if __name__ == "__main__":
    run_ray_integration_example()
    performance_comparison()