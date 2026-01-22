"""
zerolink/core/cpu/shm_pool.py

Shared Memory Pool для CPU тензоров.
Использует multiprocessing.shared_memory для Zero-Copy IPC.
"""

import multiprocessing.shared_memory as shm
import numpy as np
import threading
from typing import Dict, List, Optional, Tuple
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
    def __init__(self, name: str = "pynexus_cpu_pool", total_size_mb: int = 1024):
        self.name = name
        self.total_size = total_size_mb * 1024 * 1024
        
        self.lock = threading.Lock()
        
        # Создаем или открываем SHM
        try:
            self.shm = shm.SharedMemory(name=self.name, create=True, size=self.total_size)
            print(f"[CPU Pool] Created new SHM: {self.name}")
        except FileExistsError:
            self.shm = shm.SharedMemory(name=self.name)
            print(f"[CPU Pool] Attached to existing SHM: {self.name}")

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
            
            # Попытка слить с предыдущим
            # (Для упрощения ищем только точное совпадение конца, нужно итератор)
            # В упрощенном варианте просто добавляем блок в free_blocks
            self.free_blocks[new_offset] = new_size
            return True

    def get_numpy_array(self, block: CPUBlock, dtype: np.dtype, shape: Tuple[int, ...]):
        """Создает NumPy массив над разделяемой памятью (Zero-Copy)."""
        # Используем memoryview для доступа к SHM
        # self.shm.buf возвращает memoryview всего региона
        mem_slice = self.shm.buf[block.offset:block.offset + block.size]
        return np.ndarray(shape, dtype=dtype, buffer=mem_slice)

    def cleanup(self):
        """Удаляет SHM (только если владелец)."""
        try:
            self.shm.close()
            self.shm.unlink()
            print(f"[CPU Pool] Unlinked SHM: {self.name}")
        except Exception:
            pass