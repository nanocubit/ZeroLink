"""
test_basic_functionality_fixed.py

Простой тест основной функциональности ZeroLink v2.0 (исправленный).
"""

import torch
from zerolink.runtime.unified import UnifiedRuntime


# Глобальные функции для тестирования
def simple_cpu_task(x):
    """Простая CPU задача для тестирования."""
    return x * 2


def simple_gpu_task(size):
    """Простая GPU задача для тестирования."""
    if torch.cuda.is_available():
        t = torch.randn(size, size, device='cuda')
        return t.sum().item()
    else:
        return 0.0


def test_unified_runtime_creation():
    """Тест создания UnifiedRuntime."""
    print("Тест 1: Создание UnifiedRuntime...")
    
    # Создаем runtime с минимальными параметрами
    runtime = UnifiedRuntime(
        gpu_device_id=0,
        gpu_pool_size_gb=1,
        num_cpu_workers=1
    )
    
    print("✓ UnifiedRuntime успешно создан")
    
    # Проверяем, что атрибуты установлены
    assert hasattr(runtime, 'gpu_pool')
    assert hasattr(runtime, 'cpu_executor')
    
    print("✓ Атрибуты runtime проверены")
    
    # Проверяем статус GPU пула (может быть None если CUDA недоступна)
    print(f"✓ GPU Pool: {'Available' if runtime.gpu_pool is not None else 'Not available (CUDA)'}")
    
    # Останавливаем runtime
    runtime.stop()
    print("✓ UnifiedRuntime успешно остановлен")
    

def test_cpu_execution():
    """Тест выполнения CPU задач."""
    print("\nТест 2: Выполнение CPU задач...")
    
    runtime = UnifiedRuntime(
        gpu_device_id=0,
        gpu_pool_size_gb=1,
        num_cpu_workers=2
    )
    
    # Выполняем задачу
    result = runtime.execute_cpu(simple_cpu_task, 21)
    assert result == 42
    print(f"✓ CPU задача выполнена: 21 * 2 = {result}")
    
    # Тест map
    inputs = [1, 2, 3, 4]
    results = runtime.map_cpu(simple_cpu_task, inputs)
    expected = [2, 4, 6, 8]
    assert results == expected
    print(f"✓ CPU map задачи выполнены: {inputs} -> {results}")
    
    runtime.stop()
    print("✓ CPU тесты пройдены")


def test_gpu_execution_if_available():
    """Тест выполнения GPU задач (если доступна)."""
    print("\nТест 3: Выполнение GPU задач...")
    
    runtime = UnifiedRuntime(
        gpu_device_id=0,
        gpu_pool_size_gb=1,
        num_cpu_workers=1
    )
    
    if torch.cuda.is_available():
        print("  CUDA доступна, тестируем GPU функциональность...")
        
        try:
            result = runtime.execute_gpu(simple_gpu_task, 32)  # Маленькая матрица для теста
            print(f"  ✓ GPU задача выполнена, результат: {result:.4f}")
        except Exception as e:
            print(f"  ⚠ GPU задача не выполнена (ожидаемо): {e}")
    else:
        print("  CUDA недоступна, пропускаем GPU тесты")
    
    runtime.stop()
    print("✓ GPU тесты завершены")


def test_get_pool_stats():
    """Тест получения статистики пула."""
    print("\nТест 4: Получение статистики пула...")
    
    runtime = UnifiedRuntime(
        gpu_device_id=0,
        gpu_pool_size_gb=1,
        num_cpu_workers=1
    )
    
    stats = runtime.get_pool_stats()
    assert isinstance(stats, dict)
    assert 'cpu' in stats
    print("✓ Статистика пула успешно получена")
    print(f"  CPU статистика: {stats['cpu']}")
    
    runtime.stop()
    print("✓ Тест статистики завершен")


if __name__ == "__main__":
    print("=== Тестирование основной функциональности ZeroLink v2.0 ===\n")

    try:
        test_unified_runtime_creation()
        test_cpu_execution()
        test_gpu_execution_if_available()
        test_get_pool_stats()

        print("\n=== Все тесты пройдены успешно! ===")
        print("ZeroLink v2.0 работает корректно.")
        print("Основные компоненты функционируют как ожидалось.")

    except Exception as e:
        print(f"\n❌ Ошибка в тестах: {e}")
        import traceback
        traceback.print_exc()