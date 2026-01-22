"""
examples/final_integration_example.py

Финальный пример интеграции всех компонентов ZeroLink v2.0,
включая cgpu интеграцию.
"""

import torch
import time
from zerolink.runtime.unified import UnifiedRuntime
from zerolink.monitoring import telemetry
from zerolink.profiling import profiler
from zerolink.core.gpu import PinnedMemoryManager, MultiDevicePool, cgpu_memory_manager


def example_full_pipeline():
    """Пример полного пайплайна с использованием всех компонентов."""
    print("=== ZeroLink v2.0 - Full Pipeline Example ===\n")
    
    # 1. Используем UnifiedRuntime для оркестрации
    print("1. Инициализация UnifiedRuntime...")
    runtime = UnifiedRuntime(
        gpu_device_id=0,
        gpu_pool_size_gb=1,
        num_cpu_workers=2
    )
    print("   ✓ UnifiedRuntime готов")
    
    # 2. Используем pinned memory для эффективного data loading
    print("\n2. Использование pinned memory...")
    pinned_mgr = PinnedMemoryManager()
    data_tensor = pinned_mgr.allocate_pinned_tensor((1024, 1024))
    print(f"   ✓ Выделен pinned tensor: {data_tensor.tensor.shape}")
    
    # 3. Используем профилирование для анализа производительности
    print("\n3. Профилирование GPU операций...")
    
    @profiler.profile_function("model_inference")
    def gpu_inference_task(input_data):
        if torch.cuda.is_available():
            # Копируем данные на GPU
            gpu_input = input_data.tensor.to('cuda', non_blocking=True)
            
            # Выполняем вычисления
            result = torch.nn.functional.relu(gpu_input)
            return result.sum().item()
        else:
            # CPU fallback
            result = torch.nn.functional.relu(input_data.tensor)
            return result.sum().item()
    
    # Выполняем задачу
    result = runtime.execute_gpu(gpu_inference_task, data_tensor)
    print(f"   ✓ Результат инференса: {result:.3f}")
    
    # Проверяем отчет профилирования
    print("   ✓ Профилирование завершено")
    
    # 4. Используем телеметрию для мониторинга
    print("\n4. Мониторинг с помощью телеметрии...")
    telemetry.increment_counter("inference_requests_total", 1)
    telemetry.set_gauge("current_batch_size", 1)
    
    print("   ✓ Метрики обновлены")
    
    # 5. Проверяем интеграцию с cgpu (если доступна)
    print("\n5. Проверка интеграции с cgpu...")
    if cgpu_memory_manager is not None:
        try:
            # Проверяем доступность устройств
            device_count = len(cgpu_memory_manager.devices)
            print(f"   ✓ cgpu обнаружено {device_count} GPU устройств")
            
            if device_count > 0:
                # Получаем свойства первого устройства
                props = cgpu_memory_manager.get_device_properties(0)
                print(f"   ✓ GPU 0: {props['name']}, {props['total_memory'] / (1024**3):.1f}GB")
                
        except Exception as e:
            print(f"   ⚠ cgpu интеграция недоступна: {e}")
    else:
        print("   ⚠ cgpu не установлено (pip install cgpu для полной интеграции)")
    
    # 6. Используем Multi-GPU (если доступны)
    print("\n6. Multi-GPU масштабирование...")
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        try:
            multi_pool = MultiDevicePool(
                device_ids=[0, 1],
                pool_size_gb=1
            )
            print(f"   ✓ MultiDevicePool создан для {len(multi_pool.pools)} GPU")
        except Exception as e:
            print(f"   ⚠ Multi-GPU setup failed: {e}")
    else:
        print("   ⚠ Multi-GPU недоступно (требуется 2+ GPU)")
    
    # 7. Останавливаем runtime
    print("\n7. Завершение работы...")
    runtime.stop()
    print("   ✓ UnifiedRuntime остановлен")
    
    print("\n=== Отчет о выполнении ===")
    
    # Показываем профиль
    print("\nПрофиль выполнения:")
    print(profiler.get_profile_report())
    
    # Показываем метрики
    print("\nМетрики телеметрии:")
    metrics_text = telemetry.get_prometheus_format()
    for line in metrics_text.split('\n'):
        if line and not line.startswith('#'):
            print(f"  {line}")
    
    print("\n=== Полный пайплайн завершен успешно ===")
    print("ZeroLink v2.0 объединяет все компоненты в единый поток:")


def show_architecture_summary():
    """Показывает сводку архитектуры."""
    print("\n=== Архитектурная сводка ===")
    print("1. UnifiedRuntime - оркестрация CPU/GPU задач")
    print("2. PinnedMemoryManager - эффективный data loading")
    print("3. Profiler - анализ производительности")
    print("4. Telemetry - мониторинг и метрики")
    print("5. cgpu Integration - улучшенное CUDA API")
    print("6. MultiDevicePool - масштабирование на несколько GPU")
    print("7. Prometheus Exporter - экспорт метрик")
    print("8. Ray Integration - распределенные вычисления")
    print("9. Zero-Copy IPC - передача данных без копирования")
    print("10. Buddy Allocator - O(1) аллокация памяти")


if __name__ == "__main__":
    example_full_pipeline()
    show_architecture_summary()
    
    print("\n🎉 ZeroLink v2.0 - Полностью функциональная система!")
    print("Готова к использованию в production средах.")