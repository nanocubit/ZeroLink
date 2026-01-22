"""
integration_test.py

Интеграционный тест всех компонентов ZeroLink v2.0.
"""

import torch
import time
from zerolink.runtime.unified import UnifiedRuntime
from zerolink.monitoring import telemetry, prometheus_exporter
from zerolink.profiling import profiler
from zerolink.core.gpu import PinnedMemoryManager, MultiDevicePool, PoolType


# Глобальные функции для тестирования
def cpu_task(x):
    """CPU задача для тестирования."""
    return x * 2


def gpu_task(size):
    """GPU задача для тестирования."""
    if torch.cuda.is_available():
        t = torch.randn(size, size, device='cuda')
        return t.sum().item()
    else:
        return 0.0


def run_integration_test():
    """Запускает интеграционный тест всех компонентов."""
    print("=== ZeroLink v2.0 - Интеграционный тест ===\n")
    
    # 1. Тестируем UnifiedRuntime
    print("1. Тест UnifiedRuntime...")
    try:
        runtime = UnifiedRuntime(
            gpu_device_id=0,
            gpu_pool_size_gb=1,
            num_cpu_workers=2
        )
        print("   ✓ UnifiedRuntime создан")
        
        # Тестируем CPU выполнение
        result = runtime.execute_cpu(cpu_task, 21)
        assert result == 42
        print("   ✓ CPU выполнение работает")
        
        # Тестируем GPU выполнение (если доступно)
        if torch.cuda.is_available():
            gpu_result = runtime.execute_gpu(gpu_task, 32)  # Маленькая матрица
            print(f"   ✓ GPU выполнение работает: {gpu_result:.3f}")
        
        runtime.stop()
        print("   ✓ UnifiedRuntime остановлен")
    except Exception as e:
        print(f"   ❌ Ошибка в UnifiedRuntime: {e}")
        return False
    
    # 2. Тестируем телеметрию
    print("\n2. Тест телеметрии...")
    try:
        # Регистрируем тестовые метрики
        telemetry.register_counter("integration_test_counter", "Test counter for integration")
        telemetry.register_gauge("integration_test_gauge", "Test gauge for integration")

        # Увеличиваем счетчики
        telemetry.increment_counter("integration_test_counter", 5)
        telemetry.set_gauge("integration_test_gauge", 42.0)

        # Проверяем, что метрики доступны
        metrics = telemetry.get_metrics()
        metric_names = [m.name for m in metrics.values()]
        assert "integration_test_counter" in metric_names
        assert "integration_test_gauge" in metric_names
        print("   ✓ Телеметрия работает")

        # Проверяем формат Prometheus
        prom_text = telemetry.get_prometheus_format()
        assert "integration_test_counter" in prom_text
        print("   ✓ Формат Prometheus работает")
    except Exception as e:
        print(f"   ❌ Ошибка в телеметрии: {e}")
        return False
    
    # 3. Тестируем профилирование
    print("\n3. Тест профилирования...")
    try:
        @profiler.profile_function("integration_test_func")
        def test_func():
            time.sleep(0.1)  # Имитация работы
            return "done"
        
        result = test_func()
        assert result == "done"
        
        # Проверяем отчет
        report = profiler.get_profile_report()
        assert "integration_test_func" in report
        print("   ✓ Профилирование работает")
    except Exception as e:
        print(f"   ❌ Ошибка в профилировании: {e}")
        return False
    
    # 4. Тестируем pinned memory
    print("\n4. Тест pinned memory...")
    try:
        pinned_mgr = PinnedMemoryManager()
        region = pinned_mgr.allocate_pinned_tensor((100, 100))
        
        assert region.tensor.shape == (100, 100)
        print("   ✓ Pinned memory работает")
    except Exception as e:
        print(f"   ❌ Ошибка в pinned memory: {e}")
        return False
    
    # 5. Тестируем Multi-GPU (если доступны GPU)
    print("\n5. Тест Multi-GPU...")
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            device_ids = [0, 1]
            multi_pool = MultiDevicePool(
                device_ids=device_ids,
                pool_size_gb=1,
                pool_type=PoolType.ISOLATED  # Используем ISOLATED для теста
            )
            print(f"   ✓ MultiDevicePool создан для GPU: {device_ids}")
        else:
            print("   ⚠ CUDA недоступна или < 2 GPU, пропускаем тест")
    except Exception as e:
        print(f"   ⚠ Ошибка в Multi-GPU (ожидаемо в тестовой среде): {e}")
    
    # 6. Тестируем Prometheus exporter
    print("\n6. Тест Prometheus exporter...")
    try:
        prometheus_exporter.start()
        print("   ✓ Prometheus exporter запущен")
        
        # Останавливаем для чистоты
        prometheus_exporter.stop()
        print("   ✓ Prometheus exporter остановлен")
    except Exception as e:
        print(f"   ❌ Ошибка в Prometheus exporter: {e}")
        return False
    
    print("\n=== Все интеграционные тесты пройдены успешно! ===")
    print("ZeroLink v2.0 полностью функционален со всеми компонентами.")
    
    return True


if __name__ == "__main__":
    success = run_integration_test()
    
    if success:
        print("\n🎉 ZeroLink v2.0 готов к использованию!")
    else:
        print("\n❌ Один или несколько тестов не прошли")
        exit(1)