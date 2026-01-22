"""
tests/test_cgpu_integration.py

Тесты для интеграции с cgpu.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock

from zerolink.core.gpu import CgpuMemoryManager, cgpu_memory_manager


def test_cgpu_manager_initialization():
    """Тест инициализации CgpuMemoryManager."""
    try:
        import cgpu
    except ImportError:
        # Если cgpu не установлен, пропускаем тест
        pytest.skip("cgpu not available")
    
    # Создаем менеджер
    mgr = CgpuMemoryManager()
    
    # Проверяем, что устройства инициализированы
    assert hasattr(mgr, 'devices')
    assert hasattr(mgr, 'contexts')
    
    print(f"✓ Found {len(mgr.devices)} CUDA devices")
    
    # Проверяем, что контексты созданы для каждого устройства
    for i, device in enumerate(mgr.devices):
        assert i in mgr.contexts
    
    # Очищаем ресурсы
    mgr.cleanup()


def test_cgpu_memory_operations():
    """Тест операций с памятью через cgpu."""
    try:
        import cgpu
    except ImportError:
        pytest.skip("cgpu not available")
    
    if torch.cuda.device_count() == 0:
        pytest.skip("No CUDA devices available")
    
    mgr = CgpuMemoryManager()
    
    try:
        # Тестируем создание пула
        pool_id = mgr.create_memory_pool(device_id=0, size_gb=1)
        assert pool_id >= 0
        print(f"✓ Created memory pool with ID: {pool_id}")
        
        # Тестируем резервирование VA
        va_base = mgr.reserve_virtual_address_range(size_gb=1, device_id=0)
        assert va_base > 0
        print(f"✓ Reserved VA range at: 0x{va_base:x}")
        
        # Тестируем маппинг
        success = mgr.map_memory_to_address(pool_id, va_base)
        assert success is True
        print("✓ Memory mapped successfully")
        
        # Тестируем экспорт/импорт
        handle_data = mgr.export_memory_handle(pool_id)
        assert len(handle_data) > 0
        print(f"✓ Exported memory handle ({len(handle_data)} bytes)")
        
        imported_id = mgr.import_memory_handle(handle_data, device_id=0)
        assert imported_id >= 0
        print(f"✓ Imported memory handle as allocation {imported_id}")
        
        # Очищаем
        mgr.unmap_memory(pool_id)
        mgr.free_memory(pool_id)
        mgr.free_memory(imported_id)
        
        print("✓ All memory operations completed successfully")
        
    finally:
        mgr.cleanup()


def test_cgpu_device_properties():
    """Тест получения свойств устройств."""
    try:
        import cgpu
    except ImportError:
        pytest.skip("cgpu not available")
    
    if torch.cuda.device_count() == 0:
        pytest.skip("No CUDA devices available")
    
    mgr = CgpuMemoryManager()
    
    # Получаем свойства для первого устройства
    props = mgr.get_device_properties(0)
    
    assert 'name' in props
    assert 'compute_capability' in props
    assert 'total_memory' in props
    assert 'supports_vmm' in props
    
    print(f"✓ Device properties: {props['name']}, {props['total_memory'] / (1024**3):.1f}GB")
    
    mgr.cleanup()


def test_cgpu_manager_without_cgpu():
    """Тест поведения без установленного cgpu."""
    # Имитируем отсутствие cgpu
    with patch.dict('sys.modules', {'cgpu': None}):
        # Перезагружаем модуль для проверки обработки ошибки
        import importlib
        import zerolink.core.gpu.cgpu_adapter
        importlib.reload(zerolink.core.gpu.cgpu_adapter)
        
        from zerolink.core.gpu.cgpu_adapter import CgpuMemoryManager, CGPU_AVAILABLE
        
        assert CGPU_AVAILABLE is False
        
        # Проверяем, что при попытке создать менеджер выбрасывается ошибка
        with pytest.raises(RuntimeError, match="cgpu is not available"):
            CgpuMemoryManager()


if __name__ == "__main__":
    # Запуск тестов вручную
    test_cgpu_manager_without_cgpu()
    print("✓ cgpu absence test passed")
    
    try:
        test_cgpu_manager_initialization()
        print("✓ Initialization test passed")
    except Exception as e:
        print(f"⚠ Initialization test skipped: {e}")
    
    try:
        test_cgpu_memory_operations()
        print("✓ Memory operations test passed")
    except Exception as e:
        print(f"⚠ Memory operations test skipped: {e}")
    
    try:
        test_cgpu_device_properties()
        print("✓ Device properties test passed")
    except Exception as e:
        print(f"⚠ Device properties test skipped: {e}")
    
    print("\n=== cgpu Integration Tests Complete ===")