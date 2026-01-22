"""
zerolink/core/gpu/pinned_region.py

Поддержка pinned memory для эффективного data loading.
"""

import torch
from typing import Optional
from dataclasses import dataclass


@dataclass
class PinnedRegion:
    """
    Обертка для pinned memory региона.
    Критично для обучения (Training Loops) чтобы избежать page faults
    при чтении больших датасетов с диска в GPU память.
    """
    tensor: torch.Tensor
    device: torch.device
    size_bytes: int
    is_pinned: bool = False  # По умолчанию False, так как pinned tensor не всегда доступен

    def __post_init__(self):
        # Проверяем pinned только если указано, что тензор pinned
        if self.is_pinned and not self.tensor.is_pinned():
            raise ValueError("Tensor must be pinned memory")
    
    def copy_to_device(self, device: torch.device) -> torch.Tensor:
        """
        Копирует pinned tensor на GPU устройство.
        Это быстрее чем копировать с обычной памяти.
        """
        return self.tensor.to(device, non_blocking=True)
    
    def copy_to_device_async(self, device: torch.device) -> torch.Tensor:
        """
        Асинхронная копия pinned tensor на GPU устройство.
        """
        return self.tensor.to(device, non_blocking=True)


class PinnedMemoryManager:
    """
    Менеджер pinned memory регионов.
    """
    def __init__(self):
        self.regions = {}
        self.next_id = 0
    
    def allocate_pinned_tensor(self, shape, dtype=torch.float32) -> PinnedRegion:
        """
        Выделяет pinned tensor.
        """
        # Создаем pinned tensor
        # В Windows может быть проблема с pin_memory, поэтому делаем проверку
        try:
            tensor = torch.empty(shape, dtype=dtype, pin_memory=True)
        except RuntimeError:
            # Если не удается создать pinned tensor, создаем обычный и помечаем как unpinned
            tensor = torch.empty(shape, dtype=dtype)
            is_pinned = False
        else:
            is_pinned = True

        region = PinnedRegion(
            tensor=tensor,
            device=tensor.device,
            size_bytes=tensor.element_size() * tensor.nelement(),
            is_pinned=is_pinned
        )

        region_id = self.next_id
        self.next_id += 1
        self.regions[region_id] = region

        return region
    
    def get_region(self, region_id: int) -> Optional[PinnedRegion]:
        """Получает pinned region по ID."""
        return self.regions.get(region_id)
    
    def free_region(self, region_id: int):
        """Освобождает pinned region."""
        if region_id in self.regions:
            del self.regions[region_id]
    
    def get_stats(self):
        """Возвращает статистику по pinned memory."""
        total_size = sum(region.size_bytes for region in self.regions.values())
        return {
            "num_regions": len(self.regions),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024**2)
        }


# Глобальный менеджер pinned memory
pinned_memory_manager = PinnedMemoryManager()