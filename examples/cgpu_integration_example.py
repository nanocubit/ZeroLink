"""
examples/cgpu_integration_example.py

Пример интеграции ZeroLink v2.0 с cgpu (https://github.com/augustsletto/cgpu).
"""

import torch
from zerolink.core.gpu import CgpuMemoryManager


def example_cgpu_integration():
    """Пример использования cgpu адаптера."""
    print("=== ZeroLink + cgpu Integration Example ===\n")
    
    try:
        # Проверяем доступность cgpu
        import cgpu
        print(f"✓ cgpu version: {cgpu.__version__ if hasattr(cgpu, '__version__') else 'unknown'}")
    except ImportError:
        print("⚠ cgpu not available. This example shows the integration pattern.")
        print("  To use cgpu, install it: pip install cgpu")
        return
    
    # Создаем менеджер памяти на основе cgpu
    try:
        memory_mgr = CgpuMemoryManager()
        print("✓ CgpuMemoryManager initialized")
    except RuntimeError as e:
        print(f"⚠ Could not initialize CgpuMemoryManager: {e}")
        print("  This is expected if CUDA is not available in the environment")
        return
    
    # Получаем информацию об устройствах
    for i in range(len(memory_mgr.devices)):
        props = memory_mgr.get_device_properties(i)
        print(f"  GPU {i}: {props['name']}, {props['total_memory'] / (1024**3):.1f}GB")
    
    if len(memory_mgr.devices) == 0:
        print("⚠ No CUDA devices found, skipping memory operations")
        return
    
    # Пример работы с памятью
    print(f"\n2. Creating memory pool on GPU 0...")
    try:
        pool_id = memory_mgr.create_memory_pool(device_id=0, size_gb=1)
        print(f"   ✓ Created memory pool (ID: {pool_id}) of 1GB on GPU 0")
    except Exception as e:
        print(f"   ⚠ Could not create memory pool: {e}")
        return
    
    # Резервируем виртуальный адрес
    print(f"\n3. Reserving virtual address range...")
    try:
        va_base = memory_mgr.reserve_virtual_address_range(size_gb=1, device_id=0)
        print(f"   ✓ Reserved VA range starting at: 0x{va_base:x}")
    except Exception as e:
        print(f"   ⚠ Could not reserve VA range: {e}")
        return
    
    # Маппим память в адресное пространство
    print(f"\n4. Mapping memory to virtual address...")
    try:
        success = memory_mgr.map_memory_to_address(pool_id, va_base)
        if success:
            print(f"   ✓ Memory mapped to VA: 0x{va_base:x}")
        else:
            print(f"   ⚠ Memory mapping failed")
    except Exception as e:
        print(f"   ⚠ Could not map memory: {e}")
    
    # Экспорт дескриптора для IPC (симуляция)
    print(f"\n5. Exporting memory handle for IPC...")
    try:
        handle_data = memory_mgr.export_memory_handle(pool_id)
        print(f"   ✓ Exported memory handle ({len(handle_data)} bytes)")
        
        # Импорт в другом процессе (симуляция)
        imported_id = memory_mgr.import_memory_handle(handle_data, device_id=0)
        print(f"   ✓ Imported memory handle as new allocation (ID: {imported_id})")
    except Exception as e:
        print(f"   ⚠ Could not export/import memory handle: {e}")
    
    # Очистка
    print(f"\n6. Cleaning up...")
    try:
        memory_mgr.unmap_memory(pool_id)
        memory_mgr.free_memory(pool_id)
        print("   ✓ Original allocation freed")
        
        if 'imported_id' in locals():
            memory_mgr.free_memory(imported_id)
            print("   ✓ Imported allocation freed")
    except Exception as e:
        print(f"   ⚠ Error during cleanup: {e}")
    
    print(f"\n=== cgpu Integration Example Complete ===")
    print("This demonstrates how cgpu can enhance ZeroLink's VMM capabilities.")
    print("Features like:")
    print("- More robust CUDA Driver API bindings")
    print("- Better multi-GPU support")
    print("- Improved IPC mechanisms")
    print("- Enhanced memory management")


def comparison_with_current_approach():
    """Сравнение с текущим подходом."""
    print("\n=== Comparison: Current vs cgpu-enhanced ===")
    
    print("\nCurrent ZeroLink VMM:")
    print("  - Custom CUDA Driver API wrappers in C++")
    print("  - Manual VMM implementation")
    print("  - Custom IPC protocol")
    
    print("\nWith cgpu Integration:")
    print("  - Standardized CUDA Driver API bindings")
    print("  - Proven VMM implementation")
    print("  - Built-in IPC support")
    print("  - Better error handling")
    print("  - Cross-platform compatibility")
    
    print("\nThe cgpu adapter provides a bridge between")
    print("ZeroLink's orchestration layer and low-level")
    print("CUDA operations, potentially improving stability")
    print("and maintainability.")


if __name__ == "__main__":
    example_cgpu_integration()
    comparison_with_current_approach()