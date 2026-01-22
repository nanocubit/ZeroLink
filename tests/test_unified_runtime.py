"""
tests/test_unified_runtime.py

Тесты для ZeroLinkRuntime (ранее UnifiedRuntime) в ZeroLink v2.0.
"""

import pytest
import torch
from zerolink.runtime.unified import ZeroLinkRuntime, UnifiedRuntime


def test_zerolink_runtime_creation():
    """Тест создания ZeroLinkRuntime."""
    runtime = ZeroLinkRuntime(
        gpu_device_id=0,
        gpu_pool_size_gb=1,
        num_cpu_workers=1
    )
    
    assert runtime is not None
    assert hasattr(runtime, 'gpu_pool')
    assert hasattr(runtime, 'cpu_executor')
    
    runtime.stop()


def test_unified_runtime_alias():
    """Тест обратной совместимости через UnifiedRuntime."""
    # Проверяем, что UnifiedRuntime является тем же самым классом
    assert ZeroLinkRuntime is UnifiedRuntime
    
    runtime = UnifiedRuntime(
        gpu_device_id=0,
        gpu_pool_size_gb=1,
        num_cpu_workers=1
    )
    
    assert runtime is not None
    runtime.stop()


def test_runtime_with_context_manager():
    """Тест использования ZeroLinkRuntime как контекстного менеджера."""
    with ZeroLinkRuntime(
        gpu_device_id=0,
        gpu_pool_size_gb=1,
        num_cpu_workers=1
    ) as runtime:
        assert runtime is not None
        assert runtime._running is True
        
    # После выхода из контекста runtime должен быть остановлен
    assert not hasattr(runtime, '_running') or not runtime._running


if __name__ == "__main__":
    test_zerolink_runtime_creation()
    test_unified_runtime_alias()
    test_runtime_with_context_manager()
    print("Все тесты для ZeroLinkRuntime пройдены!")