"""
ZeroLink - Unified Zero-Copy Runtime for CPU/GPU Computing
Copyright (C) 2025 Slava Maltsev <nanotec@live.ru>
SPDX-License-Identifier: MIT

zerolink/core/cpu/shm_pool.py

Shared Memory Pool для CPU тензоров.
Использует multiprocessing.shared_memory для Zero-Copy IPC.
"""

import multiprocessing.shared_memory as shm
import threading
import logging
from typing import Dict, List, Optional, Tuple, Any

try:
    import numpy as np
except ImportError:
    np = None
from dataclasses import dataclass

@dataclass
class CPUBlock:
    """Блок разделяемой памяти."""
    offset: int
    size: int
    shm_name: str
    pool_name: str  # Для идентификации

class SharedMemoryPool:
    """
    Менеджер разделяемой памяти для CPU.
    Управляет большим блоком SHM и выдает из него куски.
    """
    def __init__(self, name: str = "zerolink_cpu_pool", total_size_mb: int = 1024):
        self.name = name
        self.total_size = total_size_mb * 1024 * 1024
        
        self.lock = threading.Lock()
        self.logger = logging.getLogger("zerolink.cpu.shm_pool")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        # Создаем или открываем SHM
        try:
            self.shm = shm.SharedMemory(name=self.name, create=True, size=self.total_size)
            self.logger.info("created shared memory pool %s", self.name)
        except FileExistsError:
            self.shm = shm.SharedMemory(name=self.name)
            self.logger.info("attached to existing shared memory pool %s", self.name)

        # Простой Free List (offset -> size)
        # В production можно использовать Buddy Allocator аналогично GPU
        self.free_blocks: Dict[int, int] = {0: self.total_size}
        self.used_blocks: Dict[int, CPUBlock] = {}

    def allocate(self, size: int) -> Optional[CPUBlock]:
        """Выделяет блок памяти."""
        with self.lock:
            # Ищем первый подходящий блок (First-Fit)
            for offset in sorted(self.free_blocks.keys()):
                block_size = self.free_blocks[offset]
                
                if block_size >= size:
                    # Вырезаем блок
                    remaining_size = block_size - size
                    
                    # Обновляем свободные блоки
                    del self.free_blocks[offset]
                    if remaining_size > 0:
                        self.free_blocks[offset + size] = remaining_size
                    
                    # Регистрируем использованный блок
                    block = CPUBlock(
                        offset=offset,
                        size=size,
                        shm_name=self.name,
                        pool_name=self.name
                    )
                    self.used_blocks[offset] = block
                    return block
        return None

    def free(self, block: CPUBlock) -> bool:
        """Освобождает блок памяти."""
        with self.lock:
            if block.offset not in self.used_blocks:
                return False
            
            del self.used_blocks[block.offset]
            
            # Простое слияние (Coalescing)
            # Проверяем соседние блоки
            new_offset = block.offset
            new_size = block.size
            
            # Попытка слить со следующим
            next_offset = block.offset + block.size
            if next_offset in self.free_blocks:
                new_size += self.free_blocks[next_offset]
                del self.free_blocks[next_offset]
            
            # Полное коалесцирование: сливаем с любыми соседними блоками до стабилизации.
            merged = True
            while merged:
                merged = False
                for off, size in sorted(self.free_blocks.items()):
                    if off + size == new_offset:
                        # merge left neighbor
                        new_offset = off
                        new_size += size
                        del self.free_blocks[off]
                        merged = True
                        break
                    if new_offset + new_size == off:
                        # merge right neighbor
                        new_size += size
                        del self.free_blocks[off]
                        merged = True
                        break

            self.free_blocks[new_offset] = new_size
            return True

    def get_numpy_array(self, block: CPUBlock, dtype: Any, shape: Tuple[int, ...]):
        """Создает NumPy массив над разделяемой памятью (Zero-Copy)."""
        if np is None:
            raise RuntimeError("NumPy is required for get_numpy_array(); install numpy.")

        # Используем memoryview для доступа к SHM
        # self.shm.buf возвращает memoryview всего региона
        mem_slice = self.shm.buf[block.offset:block.offset + block.size]
        return np.ndarray(shape, dtype=dtype, buffer=mem_slice)

    def cleanup(self):
        """Удаляет SHM (только если владелец)."""
        try:
            self.shm.close()
            self.shm.unlink()
            self.logger.info("unlinked shared memory pool %s", self.name)
        except Exception:
            pass