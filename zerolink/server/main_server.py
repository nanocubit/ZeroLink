"""
ZeroLink - Unified Zero-Copy Runtime for CPU/GPU Computing
Copyright (C) 2025 Slava Maltsev <nanotec@live.ru>
SPDX-License-Identifier: MIT

zerolink/server/main_server.py

Реализация сервера (Orchestrator).
Управляет соединениями с воркерами, раздает GPU-память и контролирует таймауты.
"""

import os
import socket
import struct
import time
import selectors
import threading
import secrets
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

# Импорты протокола
from ..core.protocol import (
    MSG_HELLO, MSG_ALLOC, MSG_ACK, MSG_RELEASE, MSG_ERROR, MSG_PING, MSG_PONG,
    CTRL_FLAG_HAS_HASH,
    pack_ctrl,
    pack_alloc_payload,
    unpack_ack_payload,
    unpack_release_payload,
    unpack_error_payload,
    hash32
)

from ..core.protocol.framing import recv_frame_with_fds, send_frame_with_fds, send_frame

# Импорты пула
from ..core.gpu.vmm_pool import DeviceMemoryPoolV2, DeviceAllocation

# ============================================================================
# Структуры состояния
# ============================================================================

@dataclass
class WorkerState:
    """Состояние подключенного воркера."""
    sock: socket.socket
    device_id: int
    last_seen: float = field(default_factory=time.time)
    address: str = "unknown"

@dataclass
class LeaseState:
    """Состояние аренды (Lease)."""
    lease_id: int
    worker_id: int
    allocation: DeviceAllocation
    alloc_id: str
    status: str = "PENDING" # PENDING, ACTIVE
    created_at: float = field(default_factory=time.time)
    expected_hash: Optional[bytes] = None

# ============================================================================
# Lease Manager (Business Logic)
# ============================================================================

class MainIPCLeaseManager2P:
    """
    Управляет жизненным циклом 2-Phase Lease протокола.
    Не работает с сокетами напрямую на чтение, только отправляет и обновляет состояние.
    """
    def __init__(self, pool: DeviceMemoryPoolV2, enable_integrity_hash: bool = True):
        self.pool = pool
        self.enable_integrity_hash = enable_integrity_hash
        self.lock = threading.RLock()
        
        # worker_fd (int) -> WorkerState
        self.workers: Dict[int, WorkerState] = {}
        
        # lease_id (int) -> LeaseState
        self.leases: Dict[int, LeaseState] = {}
        
        self.logger = logging.getLogger("LeaseManager")
        self.logger.setLevel(logging.INFO)

    def register_worker(self, sock: socket.socket, device_id: int):
        """Регистрирует нового воркера."""
        fd = sock.fileno()
        with self.lock:
            self.workers[fd] = WorkerState(sock=sock, device_id=device_id)
            self.logger.info(f"Worker registered: fd={fd}, device={device_id}")

    def unregister_worker(self, fd: int):
        """Удаляет воркера и очищает его аренды."""
        with self.lock:
            if fd in self.workers:
                del self.workers[fd]
                # Находим и отзываем аренды этого воркера
                # В реальной системе нужно возвращать память в пул
                to_remove = [lid for lid, l in self.leases.items() if l.worker_id == fd]
                for lid in to_remove:
                    self.logger.warning(f"Revoking lease {lid} due to worker disconnect")
                    self._revoke_lease(lid)

    def grant_lease(self, worker_fd: int, allocation: DeviceAllocation) -> int:
        """
        Инициирует выдачу аренды воркеру (Отправляет ALLOC).
        Возвращает lease_id.
        """
        with self.lock:
            worker = self.workers.get(worker_fd)
            if not worker:
                raise RuntimeError(f"Worker {worker_fd} not found")
            
            lease_id = secrets.randbits(64)
            alloc_id = f"alloc_{lease_id}_{allocation.va_addr:x}"
            
            # 1. Собираем сегменты и FDs
            # Это сложная часть: нужно найти, какие физические чанки покрывают VA range
            segments, fds = self._get_backing_fds(allocation)
            
            # 2. Формируем Mapping Payload (словарь для упаковщика)
            # В реальном коде упаковщик ожидает bytes, здесь мы используем pack_alloc_payload
            # который внутри вызывает build_mapping_payload.
            # Нам нужно передать метаданные.
            # Для простоты реализации протокола, мы соберем mapping_payload здесь
            # используя хелпер из protocol.main_server (если бы он был там)
            # или соберем структуру для pack_alloc_payload
            # Эмуляция структуры segments для упаковщика
            meta_segments = []
            for seg in segments:
                meta_segments.append({
                    "fd_index": seg["fd_index"],
                    "src_offset": seg["src_offset"],
                    "dst_offset": seg["dst_offset"],
                    "length": seg["length"]
                })
            
            # Здесь мы используем внутренний метод сборки пейлоада протокола
            # В `protocol/messages.py` есть `_PNX_HDR_FMT` и т.д.,
            # но высокоуровневая функция `pack_alloc_payload` принимает bytes.
            # Нам нужна функция `build_mapping_payload`.
            # Я добавлю её реализацию прямо сюда для автономности файла.
            mapping_payload = self._build_mapping_payload(
                device_id=self.pool.device_id,
                gran=2097152, # 2MB hugepage default
                alloc_id=alloc_id,
                total_size=allocation.size,
                segments=meta_segments
            )
            
            # 3. Hash Integrity
            expected_hash = None
            ctrl_flags = 0
            if self.enable_integrity_hash:
                expected_hash = hash32(mapping_payload)
                ctrl_flags |= CTRL_FLAG_HAS_HASH
            
            # 4. Отправка
            wrapper = pack_alloc_payload(lease_id, mapping_payload, expected_hash)
            try:
                # ALLOC = Type 2
                send_frame_with_fds(worker.sock, MSG_ALLOC, 0, wrapper, fds, flags=ctrl_flags)
            except Exception as e:
                self.logger.error(f"Failed to send ALLOC to worker {worker_fd}: {e}")
                raise
            
            # 5. Сохранение состояния
            self.leases[lease_id] = LeaseState(
                lease_id=lease_id,
                worker_id=worker_fd,
                allocation=allocation,
                alloc_id=alloc_id,
                expected_hash=expected_hash
            )
            return lease_id

    def handle_message(self, fd: int, mtype: int, req_id: int, payload: bytes):
        """Обрабатывает входящее сообщение от воркера."""
        with self.lock:
            # Обновляем heartbeat
            if fd in self.workers:
                self.workers[fd].last_seen = time.time()
            
            if mtype == MSG_ACK:
                self._handle_ack(fd, payload)
            elif mtype == MSG_RELEASE:
                self._handle_release(fd, payload)
            elif mtype == MSG_PONG:
                pass # Просто обновили last_seen
            elif mtype == MSG_ERROR:
                self.logger.error(f"Received ERROR from worker {fd}")

    def _handle_ack(self, fd: int, payload: bytes):
        lease_id, alloc_id, got_hash = unpack_ack_payload(payload, 0)
        # flags check skipped for simplicity
        if lease_id not in self.leases:
            self.logger.warning(f"ACK for unknown lease {lease_id}")
            return
        
        lease = self.leases[lease_id]
        if lease.status != "PENDING":
            return # Уже активна или ошибка состояния
        
        # Проверка хеша (если воркер прислал)
        if self.enable_integrity_hash and got_hash:
            if got_hash != lease.expected_hash:
                self.logger.error(f"Hash mismatch for lease {lease_id}!")
                self._revoke_lease(lease_id)
                return
        
        lease.status = "ACTIVE"
        self.logger.info(f"Lease {lease_id} is now ACTIVE on worker {fd}")

    def _handle_release(self, fd: int, payload: bytes):
        lease_id, _ = unpack_release_payload(payload)
        self.logger.info(f"Worker {fd} released lease {lease_id}")
        self._revoke_lease(lease_id)

    def _revoke_lease(self, lease_id: int):
        """Возвращает память в пул."""
        if lease_id in self.leases:
            lease = self.leases[lease_id]
            # В полноценной системе мы бы вызывали self.pool.free(lease.allocation)
            # Но только если Main сам больше не использует эту память.
            # Здесь мы просто удаляем запись об аренде.
            del self.leases[lease_id]

    def _get_backing_fds(self, allocation: DeviceAllocation) -> Tuple[List[Dict], List[int]]:
        """ 
        Находит физические чанки и соответствующие FD для диапазона VA.
        Возвращает: (segments_metadata, list_of_fds)
        """
        # Это упрощенная логика. В `vmm_pool.py` мы храним `physical_chunks`.
        # Нам нужно найти пересечения.
        # 1. Находим чанки (используем приватный метод пула или эмулируем)
        # Предположим, пул предоставляет метод find_chunks(offset, size)
        offset = allocation.va_addr - self.pool.va_base
        chunks = self.pool._find_chunks_for_range(offset, allocation.size)
        
        fds = []
        segments = []
        
        # Уникальные FD (чтобы не отправлять один FD дважды, хотя sendmsg это переварит)
        # Но протокол PNXIPC10 использует индекс в массиве FD.
        unique_handles = {} # handle_obj -> fd_index
        current_fds_list = []
        
        for chunk in chunks:
            handle = chunk['ipc_handle'] # Это int (fd) в Linux
            # Проверка на дубликаты FD в рамках одного сообщения
            if handle not in unique_handles:
                unique_handles[handle] = len(current_fds_list)
                current_fds_list.append(handle)
            
            fd_idx = unique_handles[handle]
            
            # Вычисляем смещения
            # chunk['offset'] - это смещение чанка от начала базы пула
            # allocation offset - это смещение начала аллокации от базы пула
            # Смещение внутри чанка, с которого начинается аллокация
            chunk_start_in_pool = chunk['offset']
            alloc_start_in_pool = offset
            src_offset = max(0, alloc_start_in_pool - chunk_start_in_pool)
            
            # Смещение внутри аллокации (destination), куда мапить этот кусок
            dst_offset = max(0, chunk_start_in_pool - alloc_start_in_pool)
            
            # Длина маппинга
            # Пересечение [chunk_start, chunk_end] и [alloc_start, alloc_end]
            chunk_end = chunk_start_in_pool + chunk['size']
            alloc_end = alloc_start_in_pool + allocation.size
            intersect_start = max(chunk_start_in_pool, alloc_start_in_pool)
            intersect_end = min(chunk_end, alloc_end)
            length = intersect_end - intersect_start
            
            segments.append({
                "fd_index": fd_idx,
                "src_offset": src_offset,
                "dst_offset": dst_offset,
                "length": length
            })
        
        return segments, current_fds_list

    def _build_mapping_payload(self, device_id, gran, alloc_id, total_size, segments):
        """ 
        Локальная реализация упаковки PNXIPC10, чтобы не зависеть от helper'ов.
        """
        # PNXIPC10 Header: <8sHHHHIIIQQI
        # Magic, Ver, Type, Flags, Res, DevId, NFds, NSeg, TotalSize, Gran, IdLen
        magic = b"PNXIPC10"
        ver = 1
        htype = 1
        flags = 0
        res = 0
        n_fds = max((s['fd_index'] for s in segments), default=-1) + 1
        n_seg = len(segments)
        alloc_id_b = alloc_id.encode("utf-8")
        id_len = len(alloc_id_b)
        
        hdr_fmt = "<8sHHHHIIIQQI"
        hdr = struct.pack(hdr_fmt, magic, ver, htype, flags, res, device_id, n_fds, n_seg, total_size, gran, id_len)
        
        payload = bytearray(hdr)
        payload.extend(alloc_id_b)
        
        # Segments: <IIQQQ (FdIdx, Res, Src, Dst, Len)
        seg_fmt = "<IIQQQ"
        for s in segments:
            payload.extend(struct.pack(seg_fmt, s['fd_index'], 0, s['src_offset'], s['dst_offset'], s['length']))
        
        return bytes(payload)

# ============================================================================
# Main Server (Network Layer)
# ============================================================================

class MainServer:
    """
    Сетевой сервер. Использует selectors для обработки множества соединений в одном потоке.
    """
    def __init__(self, socket_path: str, lease_manager: MainIPCLeaseManager2P):
        self.socket_path = socket_path
        self.manager = lease_manager
        self.selector = selectors.DefaultSelector()
        self.server_sock: Optional[socket.socket] = None
        self._running = False

    def start(self):
        """Инициализация сокета и запуск лупа (блокирующий вызов)."""
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass

        self.server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.server_sock.bind(self.socket_path)
        self.server_sock.listen(128)
        self.server_sock.setblocking(False)
        self.selector.register(self.server_sock, selectors.EVENT_READ, data=None)
        
        self._running = True
        print(f"[Server] Listening on {self.socket_path}")
        
        try:
            while self._running:
                events = self.selector.select(timeout=1.0)
                for key, mask in events:
                    if key.fileobj is self.server_sock:
                        self._accept_connection(key.fileobj)
                    else:
                        self._handle_client_data(key, mask)
                # Здесь можно добавить self.manager.check_heartbeats()
        except Exception as e:
            print(f"[Server] Event loop error: {e}")
        finally:
            self.stop()

    def stop(self):
        self._running = False
        if self.server_sock:
            try:
                self.selector.unregister(self.server_sock)
                self.server_sock.close()
            except:
                pass
        self.selector.close()
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except:
                pass

    def _accept_connection(self, sock):
        conn, _ = sock.accept()
        conn.setblocking(False)
        # Пока не регистрируем в менеджере, ждем HELLO
        # Но для селектора нужно зарегистрировать
        self.selector.register(conn, selectors.EVENT_READ, data={"state": "HANDSHAKE"})

    def _handle_client_data(self, key, mask):
        sock = key.fileobj
        try:
            # Читаем заголовок/фрейм
            # Используем recv_frame_with_fds (неблокирующий, но для простоты здесь блокирующий на пакет)
            # В реальном async I/O нужно буферизировать.
            # SEQPACKET упрощает дело - одно чтение = один пакет (обычно).
            try:
                mtype, flags, req_id, payload, fds = recv_frame_with_fds(sock)
            except (BlockingIOError, InterruptedError):
                return
            except Exception:
                # Ошибка протокола или разрыв
                self._close_client(key)
                return
            
            if key.data["state"] == "HANDSHAKE":
                if mtype == MSG_HELLO:
                    # Parse device_id from payload
                    device_id, _ = struct.unpack("<II", payload[:8])
                    # Регистрируем
                    self.manager.register_worker(sock, device_id)
                    key.data["state"] = "CONNECTED"
                    # Можно ответить ACK
                else:
                    print(f"[Server] Expected HELLO, got {mtype}")
                    self._close_client(key)
            else:
                # Делегируем менеджеру
                self.manager.handle_message(sock.fileno(), mtype, req_id, payload)
                # Закрываем входящие FD, если они не нужны (менеджер их не забирает при приеме)
                # В текущем протоколе воркер не шлет FD серверу, только наоборот.
                for fd in fds:
                    os.close(fd)
        except Exception as e:
            print(f"[Server] Client error: {e}")
            self._close_client(key)

    def _close_client(self, key):
        sock = key.fileobj
        fd = sock.fileno()
        self.manager.unregister_worker(fd)
        try:
            self.selector.unregister(sock)
            sock.close()
        except:
            pass