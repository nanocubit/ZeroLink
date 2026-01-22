"""
ZeroLink - Unified Zero-Copy Runtime for CPU/GPU Computing
Copyright (C) 2025 Slava Maltsev <nanotec@live.ru>
SPDX-License-Identifier: MIT

zerolink/core/gpu/vmm_pool.py

DeviceMemoryPoolV2: Управление физической и виртуальной памятью GPU.
"""

import torch
import os
import struct
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

try:
    import cuda
    from cuda import cuda, cudart
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    # Заглушки для работы без CUDA
    class cuda:
        class CUmemAllocationProp:
            pass
        class CUmemAllocationType:
            CU_MEM_ALLOCATION_TYPE_PINNED = 0
        class CUmemLocationType:
            CU_MEM_LOCATION_TYPE_DEVICE = 0
        class CUmemLocation:
            pass
        class CUmemAllocationHandleType:
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 0
        class CUmemAccess_flags:
            PROT_READWRITE = 0
        class CUmemAccessDesc:
            pass
    class cudart:
        class cudaDeviceAttr:
            cudaDevAttrVirtualMemoryManagementSupported = 0
        @staticmethod
        def cudaDeviceGetAttribute(attr, device):
            return 0
        @staticmethod
        def cudaSetDevice(device):
            pass
        @staticmethod
        def cudaGetDeviceProperties(device):
            class Prop:
                name = "Mock GPU"
            return Prop()

from .buddy import BuddyAllocatorV2, BlockInfo

@dataclass
class DeviceAllocation:
    """Информация об аллокации."""
    va_addr: int
    size: int
    physical_handle: int
    device_id: int
    pool_id: str

class DeviceMemoryPoolV2:
    def __init__(self, device_id: int, total_size_gb: int = 8, pool_id: str = None):
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available. Install cuda-python.")
        
        self.device_id = device_id
        self.total_size = total_size_gb * (1024**3)
        # Гарантируем степень двойки
        self.total_size = 1 << (self.total_size.bit_length() - 1)
        self.pool_id = pool_id or f"gpu_pool_{device_id}_{id(self)}"
        
        self._check_vmm_support()
        self._init_cuda_context()
        
        self.mem_props = self._create_memory_props()
        
        # Инициализация Buddy Allocator
        self.va_allocator = BuddyAllocatorV2(
            total_size=self.total_size,
            min_block_size=2**21
        )
        
        # Выделение физической памяти (чанки по 1GB)
        self.physical_chunks: List[Dict] = []
        self._allocate_physical_memory()
        
        # Резервирование VA Range
        self.va_base = self._reserve_va_range()
        
        # Отслеживание аллокаций
        self.allocations: Dict[int, DeviceAllocation] = {}
        self.lock = None # Будет добавлен позже для thread-safety при необходимости

    def _check_vmm_support(self):
        attr = cudart.cudaDeviceAttr.cudaDevAttrVirtualMemoryManagementSupported
        supported = cudart.cudaDeviceGetAttribute(attr, self.device_id)
        if supported == 0:
            raise RuntimeError(f"GPU {self.device_id} does not support VMM")

    def _init_cuda_context(self):
        cudart.cudaSetDevice(self.device_id)
        prop = cudart.cudaGetDeviceProperties(self.device_id)
        print(f"GPU {self.device_id}: {prop.name}")

    def _create_memory_props(self):
        props = cuda.CUmemAllocationProp()
        if CUDA_AVAILABLE:
            props.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
            props.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            props.location.id = self.device_id

            # Используем POSIX FD для IPC
            props.requestedHandleTypes = cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        return props

    def _allocate_physical_memory(self):
        chunk_size = 1 * (1024**3) # 1GB
        num_chunks = self.total_size // chunk_size
        
        for i in range(num_chunks):
            try:
                physical_mem = cuda.cuMemCreate(chunk_size, self.mem_props)
                
                # Экспорт для IPC
                handle_type = cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
                ipc_handle = cuda.cuMemExportToShareableHandle(physical_mem, handle_type, 0)
                
                chunk_info = {
                    'handle': physical_mem,
                    'size': chunk_size,
                    'ipc_handle': ipc_handle,
                    'offset': i * chunk_size,
                    'is_mapped': False
                }
                self.physical_chunks.append(chunk_info)
            except Exception as e:
                raise RuntimeError(f"Failed to allocate chunk {i}: {e}")

    def _reserve_va_range(self) -> int:
        alignment = 2**21
        try:
            va_addr, _ = cuda.cuMemAddressReserve(
                self.total_size, alignment, 0, 0
            )
            return va_addr
        except Exception as e:
            raise RuntimeError(f"Failed to reserve VA range: {e}")

    def allocate(self, size: int, alignment: int = 2**21) -> Optional[DeviceAllocation]:
        """Выделяет память из пула."""
        # Выделяем VA через buddy allocator
        va_block = self.va_allocator.allocate(size)
        if not va_block:
            return None
        
        va_offset = va_block.offset
        va_addr = self.va_base + va_offset
        
        # Находим чанки
        required_chunks = self._find_chunks_for_range(va_offset, size)
        if not required_chunks:
            self.va_allocator.free(va_block)
            return None
        
        # Мапим память
        try:
            for chunk in required_chunks:
                if not chunk['is_mapped']:
                    self._map_chunk(chunk, va_addr + chunk['offset'] - va_offset)
            
            allocation = DeviceAllocation(
                va_addr=va_addr,
                size=size,
                physical_handle=required_chunks[0]['handle'],
                device_id=self.device_id,
                pool_id=self.pool_id
            )
            self.allocations[va_addr] = allocation
            return allocation
        except Exception as e:
            # Rollback
            for chunk in required_chunks:
                if chunk.get('is_mapped'):
                    self._unmap_chunk(chunk)
            self.va_allocator.free(va_block)
            return None

    def _find_chunks_for_range(self, offset: int, size: int) -> List[Dict]:
        """Находит чанки, покрывающие диапазон."""
        chunks = []
        end_offset = offset + size
        
        for chunk in self.physical_chunks:
            chunk_start = chunk['offset']
            chunk_end = chunk_start + chunk['size']
            
            # Проверка пересечения
            if not (end_offset <= chunk_start or offset >= chunk_end):
                chunks.append(chunk)
        
        chunks.sort(key=lambda c: c['offset'])
        return chunks

    def _map_chunk(self, chunk: Dict, va_addr: int):
        cuda.cuMemMap(va_addr, chunk['size'], 0, chunk['handle'])
        
        access_desc = cuda.CUmemAccessDesc(
            location=cuda.CUmemLocation(
                type=cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE,
                id=self.device_id
            ),
            flags=cuda.CUmemAccess_flags.PROT_READWRITE
        )
        
        cuda.cuMemSetAccess(va_addr, chunk['size'], [access_desc], 1)
        chunk['is_mapped'] = True
        chunk['va_addr'] = va_addr

    def _unmap_chunk(self, chunk: Dict):
        if chunk.get('is_mapped'):
            cuda.cuMemUnmap(chunk['va_addr'], chunk['size'])
            chunk['is_mapped'] = False

    def free(self, allocation: DeviceAllocation) -> bool:
        """Освобождает аллокацию."""
        if allocation.va_addr not in self.allocations:
            return False
            
        offset = allocation.va_addr - self.va_base
        chunks = self._find_chunks_for_range(offset, allocation.size)
        
        # Отмаппинг (но не освобождаем физическую память, она живет вечно в этом упрощенном примере)
        for chunk in chunks:
            if chunk.get('is_mapped'):
                self._unmap_chunk(chunk)
        
        # Освобождение VA
        fake_block = BlockInfo(
            offset=offset, size=allocation.size, state=BlockState.ALLOCATED,
            order=0, last_access=time.time()
        )
        self.va_allocator.free(fake_block)
        
        del self.allocations[allocation.va_addr]
        return True
    
    def cleanup(self):
        print(f"Cleaning up pool {self.pool_id}")
        self.va_allocator.shutdown()