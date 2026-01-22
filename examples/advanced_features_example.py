"""
examples/advanced_features_example.py

Пример использования продвинутых функций ZeroLink v2.0.
"""

import torch
import time
from zerolink.runtime.unified import UnifiedRuntime
from zerolink.monitoring import telemetry, prometheus_exporter
from zerolink.profiling import profiler
from zerolink.core.gpu import PinnedMemoryManager, MultiDevicePool, PoolType


def example_pinned_memory():
    """Пример использования pinned memory."""
    print("=== Пример: Pinned Memory ===")
    
    # Создаем менеджер pinned memory
    pinned_mgr = PinnedMemoryManager()
    
    # Выделяем pinned tensor
    pinned_region = pinned_mgr.allocate_pinned_tensor((1024, 1024), dtype=torch.float32)
    print(f"✓ Выделен pinned tensor размером {pinned_region.tensor.shape}")
    
    # Заполняем данными
    pinned_region.tensor.fill_(3.14)
    
    # Копируем на GPU (быстрее чем с обычной памяти)
    if torch.cuda.is_available():
        gpu_tensor = pinned_region.copy_to_device(torch.device('cuda:0'))
        print(f"✓ Скопировано на GPU: {gpu_tensor.shape}, max value: {gpu_tensor.max().item():.3f}")
    
    # Статистика
    stats = pinned_mgr.get_stats()
    print(f"✓ Статистика pinned memory: {stats}")
    
    print()


def example_monitoring():
    """Пример использования мониторинга."""
    print("=== Пример: Мониторинг и телеметрия ===")
    
    # Запускаем Prometheus exporter
    prometheus_exporter.start()
    print("✓ Prometheus exporter запущен на http://localhost:8000/metrics")
    
    # Выполняем несколько операций для генерации метрик
    for i in range(5):
        telemetry.increment_counter("my_app_requests_total", 1, {"endpoint": "/api/data"})
        telemetry.set_gauge("my_app_active_users", i * 10)
        time.sleep(0.1)  # Небольшая задержка
    
    # Показываем текущие метрики
    print("Текущие метрики:")
    print(telemetry.get_prometheus_format())
    
    print()


@profiler.profile_function("gpu_matrix_op")
def gpu_matrix_operation(size):
    """Профилируемая GPU операция."""
    if torch.cuda.is_available():
        a = torch.randn(size, size, device='cuda')
        b = torch.randn(size, size, device='cuda')
        c = torch.mm(a, b)
        return c.sum().item()
    else:
        # Заглушка для CPU
        a = torch.randn(size, size)
        b = torch.randn(size, size)
        c = torch.mm(a, b)
        return c.sum().item()


def example_profiling():
    """Пример использования профилирования."""
    print("=== Пример: Профилирование ===")
    
    # Выполняем несколько профилируемых операций
    for size in [512, 1024, 2048]:
        result = gpu_matrix_operation(size)
        print(f"✓ GPU операция {size}x{size}: результат = {result:.3f}")
    
    # Показываем отчет профилирования
    print("\nОтчет профилирования:")
    print(profiler.get_profile_report())
    
    print()


def example_multigpu_nvlink():
    """Пример Multi-GPU с NVLink/P2P."""
    print("=== Пример: Multi-GPU с NVLink/P2P ===")
    
    # Проверяем доступные GPU
    if not torch.cuda.is_available():
        print("CUDA недоступна, пропускаем Multi-GPU пример")
        return
    
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print(f"Требуется минимум 2 GPU (доступно: {num_gpus}), пропускаем пример")
        return
    
    # Используем первые 2 GPU
    device_ids = [0, 1] if num_gpus >= 2 else [0]
    
    # Создаем MultiDevicePool с поддержкой P2P
    try:
        multi_pool = MultiDevicePool(
            device_ids=device_ids,
            pool_size_gb=1,
            pool_type=PoolType.P2P  # Используем P2P соединение
        )
        print(f"✓ Создан MultiDevicePool для GPU: {device_ids}")
        
        # Выполняем кросс-девайсную аллокацию (если поддерживается)
        if len(device_ids) > 1:
            cross_allocs = multi_pool.allocate_cross_device(1024*1024, device_ids)  # 1MB на каждый GPU
            print(f"✓ Выполнена кросс-девайсная аллокация на {len(cross_allocs)} GPU")
        
    except Exception as e:
        print(f"⚠ Multi-GPU setup failed (expected in some environments): {e}")
    
    print()


def main():
    """Основная функция примера."""
    print("ZeroLink v2.0 - Продвинутые функции")
    print("=" * 50)
    
    example_pinned_memory()
    example_monitoring()
    example_profiling()
    example_multigpu_nvlink()
    
    print("Все примеры выполнены!")
    print("\nДополнительные возможности ZeroLink v2.0:")
    print("- Мониторинг с экспортом в Prometheus")
    print("- Профилирование GPU/CPU операций")
    print("- Поддержка pinned memory для быстрого data loading")
    print("- Multi-GPU масштабирование с NVLink/P2P")
    print("- Интеграция с CUDA Graphs для оптимизации")


if __name__ == "__main__":
    main()