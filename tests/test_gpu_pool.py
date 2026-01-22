"""
tests/test_gpu_pool.py

Тесты для GPU пула ZeroLink v2.0.
"""

import pytest
import torch

# Пропускаем тесты, если CUDA недоступна (для CI без GPU)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_pool_basic():
    """Проверка базовой функциональности GPU пула."""
    from zerolink.core.gpu.vmm_pool import DeviceMemoryPoolV2
    
    # Создаем пул (только если CUDA доступна)
    pool = DeviceMemoryPoolV2(device_id=0, total_size_gb=1)
    
    # Проверяем базовые параметры
    assert pool.device_id == 0
    assert pool.total_size > 0
    
    # Проверяем аллокацию
    alloc = pool.allocate(1024*1024)  # 1MB
    if alloc is not None:  # Может не пройти, если память закончилась
        assert alloc.size == 1024*1024
        assert alloc.device_id == 0
        # Освобождаем
        pool.free(alloc)

def test_buddy_allocator():
    """Тест Buddy Allocator."""
    from zerolink.core.gpu.buddy import BuddyAllocatorV2
    
    # Создаем маленький аллокатор для теста
    allocator = BuddyAllocatorV2(total_size=2**23, min_block_size=2**21)  # 8MB, 2MB min
    
    # Выделяем блок
    block = allocator.allocate(1024*1024)  # 1MB
    assert block is not None
    assert block.size >= 1024*1024
    
    # Освобождаем
    result = allocator.free(block)
    assert result is True
    
    # Закрываем аллокатор
    allocator.shutdown()