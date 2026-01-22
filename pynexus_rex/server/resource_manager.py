"""
zerolink/server/resource_manager.py

Адаптер для управления пулами (GPU/CPU) на уровне сервера.
В данном случае — это просто обертка над DeviceMemoryPoolV2.
"""

from ..core.gpu.vmm_pool import DeviceMemoryPoolV2

class ResourceManager:
    """Менеджер ресурсов (в настоящее время GPU-ориентированный)."""
    def __init__(self, device_id: int, pool_size_gb: int):
        self.pool = DeviceMemoryPoolV2(
            device_id=device_id,
            total_size_gb=pool_size_gb
        )

    def get_pool(self):
        return self.pool

    def cleanup(self):
        self.pool.cleanup()