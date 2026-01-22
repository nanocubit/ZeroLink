"""
demo.py

Демонстрация работы ZeroLink v2.0.
Запускает UnifiedRuntime, выполняет тяжелые задачи параллельно и измеряет время.
"""

import time
import torch
import numpy as np

# Импорты
from zerolink.runtime import UnifiedRuntime

# ВАЖНО: Для корректной работы демонстрации убедитесь, что Python CUDA доступен.
if not torch.cuda.is_available():
    print("CUDA is not available. Please check your installation.")
    print("Demo will run in CPU-only mode.")
else:
    print("CUDA is available. Running GPU+CPU demo.")

def synthetic_heavy_cpu_task(batch_id):
    """Симуляция тяжелой задачи на CPU (например, парсинг JSON)."""
    import time
    print(f"  [CPU Task {batch_id}] started...")
    time.sleep(0.05) # Эмуляция задержки (например, диск I/O или сетевой RPC)
    return f"processed_batch_{batch_id}"

def synthetic_heavy_gpu_task(size: int):
    """
    Симуляция тяжелой задачи на GPU (например, умножение больших матриц).
    """
    if not torch.cuda.is_available():
        print(f"  [GPU Task] CUDA not available, skipping...")
        return 0.0
    
    print(f"  [GPU Task] Allocating and Computing {size}x{size}...")
    
    # Эти тензоры будут аллоцироваться через TorchAwareDeviceMemoryPoolV2 (или стандартный, если пул не подключен).
    # Для демонстрации мы используем стандартную аллокацию, чтобы гарантировать выполнение.
    
    # Аллоцируем
    x = torch.randn(size, size, device='cuda')
    y = torch.randn(size, size, device='cuda')
    
    # Вычисления
    res = torch.mm(x, y)
    return res.sum().item()

def run_demo():
    print("-------------------------------------------------------------")
    print("ZeroLink v2.0: Demo: Heavy CPU + Heavy GPU")
    print("-------------------------------------------------------------")

    # Инициализация Runtime
    # GPU: 1GB, CPU: 2 workers (для демонстрации)
    runtime = UnifiedRuntime(
        gpu_device_id=0,
        gpu_pool_size_gb=1,
        num_cpu_workers=2
    )

    try:
        # -------------------------------------------------------------
        # PHASE 1: CPU Tasks (Preprocessing)
        # -------------------------------------------------------------
        print("\n[Phase 1: Running Heavy CPU Tasks (Simulated)...")

        # Создаем список данных (батчи)
        cpu_batches = [f"batch_{i}" for i in range(5)]  # 5 батчей для демо

        # Запускаем CPU воркеров (параллельно)
        start = time.time()
        
        # map работает асинхронно и возвращает список результатов
        results_cpu = runtime.map_cpu(synthetic_heavy_cpu_task, cpu_batches)
        
        duration_cpu = time.time() - start
        
        print(f"\n✅ CPU Phase Complete.")
        print(f"   Tasks processed: {len(results_cpu)}")
        print(f"   Time: {duration_cpu:.4f}s")
        print(f"   Output (CPU): {[r[-4:] for r in results_cpu]}")

        # -------------------------------------------------------------
        # PHASE 2: GPU Tasks (Inference)
        # -------------------------------------------------------------
        if torch.cuda.is_available():
            print("\n[Phase 2: Running Heavy GPU Tasks (Inference)...")

            # Размеры задач (разные, чтобы нагрузить GPU)
            gpu_tasks = [512, 1024]  # 512x512, 1024x1024 (меньше для демо)

            start_gpu = time.time()

            # Запускаем GPU задач (последовательно для демо)
            results_gpu = []
            for size in gpu_tasks:
                res = synthetic_heavy_gpu_task(size)
                results_gpu.append(res)
                # Логируем (или выводим результат)
                print(f"   GPU Task {size}x{size} completed")

            duration_gpu = time.time() - start_gpu

            print(f"\n✅ GPU Phase Complete.")
            print(f"   Tasks processed: {len(results_gpu)}")
            print(f"   Time: {duration_gpu:.4f}s")
            print(f"   Output (GPU): {[round(r, 4) for r in results_gpu]}")
        else:
            print("\n[Phase 2: Skipping GPU tasks (CUDA not available)]")

    finally:
        # -------------------------------------------------------------
        # PHASE 3: Shutdown
        # -------------------------------------------------------------
        print("\n[Phase 3: Shutting down and Stats...")
        
        # Получаем статистику пула
        stats = runtime.get_pool_stats()
        
        # Выводим GPU статистику (если CUDA доступна)
        print(f"\n📊 Pool Statistics:")
        if torch.cuda.is_available() and 'gpu' in stats:
            gpu_info = stats['gpu']
            print(f"   Pool ID: {gpu_info.get('pool_id', 'N/A')}")
            print(f"   Total Size: {gpu_info.get('total_size_gb', 'N/A'):.1f} GB")
            print(f"   Used: {gpu_info.get('used_gb', 'N/A'):.1f} GB")
            print(f"   Active Leases: {stats.get('ipc_lease_status', {}).get('active', 'N/A')}")
        
        # Выводим статистику аренд
        if 'ipc_lease_status' in stats:
            ipc = stats['ipc_lease_status']
            print(f"   Pending Leases: {ipc.get('pending', 'N/A')}")
            print(f"   Active Leases: {ipc.get('active', 'N/A')}")
        
        # Полная остановка
        runtime.stop()
        
        print("\n-------------------------------------------------------------")
        print("🎉 ZeroLink v2.0 Demo Finished!")
        print("-------------------------------------------------------------")

if __name__ == "__main__":
    # Запускаем демо
    run_demo()