"""
ZeroLink - Unified Zero-Copy Runtime for CPU/GPU Computing
Copyright (C) 2025 Slava Maltsev <nanotec@live.ru>
SPDX-License-Identifier: MIT

zerolink/core/gpu/buddy.py

Buddy Allocator для управления виртуальным адресным пространством (VA).
Обеспечивает эффективное выделение памяти без фрагментации.
"""

import threading
import time
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
from collections import deque

class BlockState(Enum):
    FREE = 0
    ALLOCATED = 1
    SPLIT = 2

@dataclass
class BlockInfo:
    """Информация о блоке памяти."""
    offset: int
    size: int
    state: BlockState
    order: int
    last_access: float = field(default_factory=time.time)
    allocation_time: float = 0.0

class BuddyAllocatorV2:
    """
    Buddy Allocator для управления VA пространством.
    """
    def __init__(self, total_size: int, min_block_size: int = 2**21): # 2MB default
        assert (total_size & (total_size - 1)) == 0, "total_size must be power of 2"
        assert (min_block_size & (min_block_size - 1)) == 0, "min_block_size must be power of 2"
        
        self.total_size = total_size
        self.min_block_size = min_block_size
        self.max_order = int(math.log2(total_size // min_block_size))
        
        # Свободные списки: free_lists[order] -> deque of offsets
        self.free_lists: List[deque] = [deque() for _ in range(self.max_order + 1)]
        
        # Все блоки: offset -> BlockInfo
        self.blocks: Dict[int, BlockInfo] = {}
        
        self.lock = threading.RLock()
        self.stat_lock = threading.Lock()
        
        # Статистика
        self.stats = {
            'allocations': 0,
            'frees': 0,
            'splits': 0,
            'merges': 0,
            'fragmentation': 0.0,
            'allocated_bytes': 0
        }
        
        # Инициализация root блока
        root = BlockInfo(
            offset=0, size=total_size, state=BlockState.FREE,
            order=self.max_order, last_access=time.time()
        )
        self.blocks[0] = root
        self.free_lists[self.max_order].append(0)
        
        # Фоновая дефрагментация
        self._shutdown = False
        self._defrag_thread = threading.Thread(target=self._defrag_worker, daemon=True)
        self._defrag_thread.start()

    def _size_to_order(self, size: int) -> int:
        """Вычисляет минимальный порядок для размера."""
        if size < self.min_block_size:
            size = self.min_block_size
        
        order = 0
        while (1 << order) * self.min_block_size < size:
            order += 1
        return min(order, self.max_order)

    def allocate(self, size: int) -> Optional[BlockInfo]:
        """Выделяет блок."""
        with self.lock:
            order = self._size_to_order(size)
            block = self._find_and_split(order)
            if not block:
                self._defragment() # Попытка дефрагментации
                block = self._find_and_split(order) # Повторный поиск
            
            if block:
                block.state = BlockState.ALLOCATED
                block.allocation_time = time.time()
                with self.stat_lock:
                    self.stats['allocations'] += 1
                    self.stats['allocated_bytes'] += block.size
                    self._update_fragmentation()
                return block
        return None

    def _find_and_split(self, target_order: int) -> Optional[BlockInfo]:
        """Ищет и разбивает блок."""
        for order in range(target_order, self.max_order + 1):
            if self.free_lists[order]:
                offset = self.free_lists[order].popleft()
                block = self.blocks[offset]
                while block.order > target_order:
                    block = self._split_one_level(block)
                return block
        return None

    def _split_one_level(self, block: BlockInfo) -> BlockInfo:
        """Разделяет блок пополам."""
        new_order = block.order - 1
        block_size = (1 << new_order) * self.min_block_size
        buddy_offset = block.offset + block_size
        
        buddy = BlockInfo(
            offset=buddy_offset, size=block_size, state=BlockState.FREE,
            order=new_order, last_access=time.time()
        )
        
        block.order = new_order
        block.state = BlockState.SPLIT
        
        self.blocks[buddy_offset] = buddy
        self.free_lists[new_order].append(buddy_offset)
        
        with self.stat_lock:
            self.stats['splits'] += 1
        return block

    def free(self, block: BlockInfo) -> bool:
        """Освобождает блок."""
        with self.lock:
            if block.offset not in self.blocks:
                return False
            
            block.state = BlockState.FREE
            self.free_lists[block.order].append(block.offset)
            self._coalesce(block.offset, block.order)
            
            with self.stat_lock:
                self.stats['frees'] += 1
                self.stats['allocated_bytes'] -= block.size
                self._update_fragmentation()
            return True

    def _coalesce(self, offset: int, order: int):
        """Объединяет с buddy блоком."""
        if order >= self.max_order:
            return
        
        block_size = (1 << order) * self.min_block_size
        buddy_offset = offset ^ block_size
        
        if (buddy_offset in self.blocks and 
            self.blocks[buddy_offset].state == BlockState.FREE and
            self.blocks[buddy_offset].order == order):
            
            self.free_lists[order].remove(buddy_offset)
            del self.blocks[buddy_offset]
            
            parent_offset = min(offset, buddy_offset)
            parent_order = order + 1
            
            parent_block = BlockInfo(
                offset=parent_offset,
                size=(1 << parent_order) * self.min_block_size,
                state=BlockState.FREE,
                order=parent_order,
                last_access=time.time()
            )
            
            self.blocks[parent_offset] = parent_block
            self.free_lists[parent_order].append(parent_offset)
            
            with self.stat_lock:
                self.stats['merges'] += 1
            
            self._coalesce(parent_offset, parent_order)

    def _update_fragmentation(self):
        """Обновляет метрику фрагментации."""
        total_free = 0
        largest_contiguous = 0
        
        for order, free_list in enumerate(self.free_lists):
            block_size = (1 << order) * self.min_block_size
            total_free += len(free_list) * block_size
            if free_list:
                largest_contiguous = max(largest_contiguous, block_size)
        
        if total_free > 0:
            self.stats['fragmentation'] = 1.0 - (largest_contiguous / total_free)
        else:
            self.stats['fragmentation'] = 0.0

    def _defragment(self):
        """Пытается слить свободные блоки."""
        free_blocks = []
        for order, free_list in enumerate(self.free_lists):
            for offset in free_list:
                free_blocks.append((offset, order))
        
        free_blocks.sort(key=lambda x: x[0])
        
        i = 0
        while i < len(free_blocks) - 1:
            offset1, order1 = free_blocks[i]
            offset2, order2 = free_blocks[i + 1]
            
            if order1 == order2:
                block_size = (1 << order1) * self.min_block_size
                expected_buddy_offset = offset1 ^ block_size
                
                if offset2 == expected_buddy_offset:
                    self._coalesce(offset1, order1)
                    return self._defragment()
            i += 1

    def _defrag_worker(self):
        """Фоновый поток дефрагментации."""
        while not self._shutdown:
            time.sleep(60)
            with self.lock:
                if self.stats['fragmentation'] > 0.7:
                    self._defragment()

    def shutdown(self):
        self._shutdown = True
        if self._defrag_thread.is_alive():
            self._defrag_thread.join(timeout=2.0)

    def get_stats(self) -> Dict:
        return dict(self.stats)