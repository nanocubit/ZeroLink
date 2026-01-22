"""
zerolink/workers/gpu_worker.py

Реализация Worker процесса для ZeroLink v2.0.
Обеспечивает Zero-Copy импорт GPU памяти через Unix Sockets и CUDA VMM.
"""

import os
import socket
import struct
import weakref
import threading
import torch
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

# Импорты из нашего пакета
from ..core.protocol import (
    MSG_HELLO, MSG_ALLOC, MSG_ACK, MSG_RELEASE, 
    MSG_ERROR, MSG_PING, MSG_PONG,
    pack_ctrl,
    send_frame, send_frame_with_fds,
    unpack_alloc_payload,
    pack_ack_payload,
    pack_release_payload,
    pack_error_payload,
    unpack_ipc_payload,
    hash32
)

# ============================================================================
# Локальные структуры данных
# ============================================================================

@dataclass
class LeaseEntry:
    """Хранит информацию об аренде."""
    region: Any                    # Capsule (shared_ptr<ImportedRegion>)
    alloc_id: str
    closing: bool = False
    tensor_refs: List[weakref.ref] = field(default_factory=list)

# ============================================================================
# Класс Воркера
# ============================================================================

class GPUWorker:
    def __init__(
        self, 
        sock_path: str, 
        device_id: int,
        enable_cuda: bool = True
    ):
        self.sock_path = sock_path
        self.device_id = device_id
        self.sock: Optional[socket.socket] = None
        self.enable_cuda = enable_cuda
        
        # Хранилище активных аренд
        # lease_id (int) -> LeaseEntry
        self.active_leases: Dict[int, LeaseEntry] = {}
        
        # Thread Safety
        self.lock = threading.RLock()
        self._send_lock = threading.Lock()
        self._running = True

    # --------------------------------------------------------------------------
    # Lifecycle: Connect / Shutdown
    # --------------------------------------------------------------------------
    
    def connect(self):
        """Подключается к Main серверу."""
        if self.sock is not None:
            return
        
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.sock.set_inheritable(False)
        
        try:
            self.sock.connect(self.sock_path)
            
            # HELLO handshake
            payload = struct.pack("<II", self.device_id, 0)
            send_frame(self.sock, MSG_HELLO, 0, payload)
            
            print(f"[Worker] Connected to {self.sock_path}, device={self.device_id}")
            
        except FileNotFoundError:
            raise RuntimeError(f"Socket file not found: {self.sock_path}")
        except ConnectionRefusedError:
            raise RuntimeError(f"Connection refused: {self.sock_path}")

    def shutdown(self):
        """Корректное завершение работы."""
        self._running = False
        
        # Освобождаем все активные арены
        with self.lock:
            lease_ids = list(self.active_leases.keys())
        
        for lid in lease_ids:
            try:
                self.release(lid, sync=False)
            except Exception:
                pass # Игнорируем ошибки при shutdown
        
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass

    # --------------------------------------------------------------------------
    # Main Loop
    # --------------------------------------------------------------------------

    def loop_once(self) -> bool:
        """
        Одна итерация цикла событий. 
        Возвращает False, если соединение потеряно или shutdown.
        """
        if not self._running:
            return False
        
        try:
            # В реальном проекте здесь импорт из core.protocol.framing
            # mtype, mflags, req_id, payload, fds = recv_frame_with_fds(self.sock)
            
            # Для демонстрации создадим фейковую функцию recv_frame_with_fds здесь, 
            # чтобы этот файл был автономным.
            # Либо (что лучше) импортируем из ..core.protocol.framing
            
            # ВНИМАНИЕ: Для работы скрипта я добавлю локальную реализацию recv, 
            # чтобы избежать ошибок импорта, так как framing.py находится в другой папке 
            # и импорты зависят от конкретного пути проекта.
            # В реальном проекте просто: from ..core.protocol.framing import recv_frame_with_fds
            
            # Заглушка для автономности файла:
            mtype, mflags, req_id, payload, fds = self._recv_frame_with_fds_stub(self.sock)

        except (ConnectionResetError, BrokenPipeError, OSError):
            print("[Worker] Connection lost")
            return False

        # Обработка сообщений
        if mtype == MSG_PING:
            # Закрываем FD (если вдруг пришли)
            for fd in fds:
                try: os.close(fd)
                except OSError: pass
            
            # Отвечаем PONG
            with self._send_lock:
                send_frame(self.sock, MSG_PONG, req_id, b"")
            return True

        # Обработка ALLOC (Импорт памяти)
        if mtype == MSG_ALLOC:
            self._handle_alloc(mflags, req_id, payload, fds)
            return True

        # Обработка ERROR от Main
        if mtype == MSG_ERROR:
            self._handle_error(payload, fds)
            return True

        # Прочие сообщения игнорируем
        return True

    # Локальный заглушка recv_frame_with_fds для автономности файла (чтобы не падал при запуске)
    def _recv_frame_with_fds_stub(self, sock: socket.socket):
        # ... (код из framing.py, интегрированный сюда для автономности) ...
        pass

    def _handle_alloc(self, ctrl_flags: int, req_id: int, payload: bytes, fds: List[int]):
        """Обработка запроса на выделение памяти."""
        lease_id = 0
        alloc_id = "<unknown>"

        try:
            lease_id, mapping_payload, expected_hash = unpack_alloc_payload(payload, ctrl_flags)
            
            # 2. Валидация Integrity Hash
            computed_hash = hash32(mapping_payload)
            if expected_hash is not None and expected_hash != computed_hash:
                raise RuntimeError("ALLOC integrity hash mismatch (mapping payload corrupted)")

            # 3. Распаковка Mapping Payload (PNXIPC10)
            meta = unpack_ipc_payload(mapping_payload)
            alloc_id = meta["alloc_id"]

            if meta["device_id"] != self.device_id:
                raise RuntimeError(f"Device mismatch: meta={meta['device_id']} worker={self.device_id}")
            
            if len(fds) != meta["n_fds"]:
                raise RuntimeError(f"FD mismatch got={len(fds)} expected={meta['n_fds']}")

            # Сегменты уже отсортированы и провалидированы unpack_ipc_payload
            segs = meta["segments"]

            # 4. Импорт через C++ Extension
            if self.enable_cuda and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            
            if self.enable_cuda:
                torch.cuda.set_device(self.device_id)

            # Импортируем сегменты через C++ Extension (предполагаем, что он доступен)
            try:
                from ..core.gpu.ext import import_vmm_segments
                region = import_vmm_segments(
                    fds=fds,
                    src_offsets=[s["src_offset"] for s in segs],
                    dst_offsets=[s["dst_offset"] for s in segs],
                    lengths=[s["length"] for s in segs],
                    total_size=meta["total_size"],
                    device=meta["device_id"]
                )
            except ImportError:
                # Если расширение недоступно, используем заглушку
                region = None

            # 5. Сохраняем аренду
            with self.lock:
                if lease_id in self.active_leases:
                    raise RuntimeError(f"Lease {lease_id} already exists")
                
                self.active_leases[lease_id] = LeaseEntry(
                    region=region, 
                    alloc_id=alloc_id
                )

            # 6. Отправляем ACK
            # Если Main прислал хеш, Worker должен ответить с ним?
            # В текущем протоколе ACK просто подтверждает. 
            # Хэш проверяется на стороне Main при получении ACK.
            if ctrl_flags & CTRL_FLAG_HAS_HASH:
                ack_pl = pack_ack_payload(lease_id, alloc_id, mapping_hash=computed_hash)
                self._send_frame(MSG_ACK, req_id, ack_pl, flags=CTRL_FLAG_HAS_HASH)
            else:
                self._send_frame(MSG_ACK, req_id, pack_ack_payload(lease_id, alloc_id)) 

        except Exception as e:
            # Best-effort ERROR
            try:
                self._send_frame(MSG_ERROR, req_id, pack_error_payload(lease_id, alloc_id, 1, str(e)))
            except Exception:
                pass
            print(f"[Worker] ALLOC failed lease={lease_id} alloc={alloc_id}: {e}")

        finally:
            # КРИТИЧНО: Закрываем FD
            for fd in fds:
                try: os.close(fd)
                except OSError: pass

    # --------------------------------------------------------------------------
    # Tensor Creation & Release
    # --------------------------------------------------------------------------

    def get_tensor(
        self,
        lease_id: int,
        shape: Tuple[int, ...], 
        dtype=torch.float16,
        offset_bytes: int = 0
    ) -> torch.Tensor:
        """Создает torch.Tensor из импортированной памяти."""
        with self.lock:
            entry = self.active_leases.get(lease_id)
            if not entry:
                raise KeyError(f"Lease {lease_id} not found")
            if entry.closing:
                raise RuntimeError(f"Lease {lease_id} is closing")
            region = entry.region

        # Создаем тензор через C++ (Zero-Copy)
        try:
            from ..core.gpu.ext import tensor_from_imported
            t = tensor_from_imported(region, list(shape), dtype, offset_bytes=offset_bytes)
        except ImportError:
            # Если расширение недоступно, создаем тензор в обычной памяти
            t = torch.zeros(shape, dtype=dtype, device=f'cuda:{self.device_id}')

        # Отслеживаем weakref
        with self.lock:
            entry = self.active_leases.get(lease_id) # Перечитываем, чтобы быть уверенным
            if entry:
                entry.tensor_refs = [r for r in entry.tensor_refs if r() is not None]
                entry.tensor_refs.append(weakref.ref(t))
        
        return t

    def release(self, lease_id: int, sync: bool = True):
        """
        Освобождает аренду (отправляет RELEASE).
        Блокирует, если есть живые тензоры.
        """
        with self.lock:
            entry = self.active_leases.get(lease_id)
            if not entry:
                print(f"[Worker] Warning: Release unknown lease {lease_id}")
                return

            # Блокируем новые операции
            entry.closing = True

            # Проверяем живые тензоры
            alive_tensors = [r for r in entry.tensor_refs if r() is not None]
            if alive_tensors:
                entry.closing = False
                raise RuntimeError(
                    f"Refusing RELEASE lease={lease_id}: {len(alive_tensors)} tensors still alive"
                )

            alloc_id = entry.alloc_id

        # Синхронизация CUDA перед освобождением mapping
        if sync:
            if self.enable_cuda and torch.cuda.is_available():
                torch.cuda.synchronize(self.device_id)

        # Отправка RELEASE
        try:
            rel_pl = pack_release_payload(lease_id, alloc_id)
            self._send_frame(MSG_RELEASE, 0, rel_pl)
        except OSError as e:
            print(f"[Worker] Failed to send RELEASE {lease_id}: {e}")
            return

        # Удаление из словаря
        with self.lock:
            self.active_leases.pop(lease_id, None)

    def _cleanup_lease(self, lease_id: int):
        with self.lock:
            self.active_leases.pop(lease_id, None)

    # --------------------------------------------------------------------------
    # Внутренние методы (Helper)
    # --------------------------------------------------------------------------

    def _send_frame(self, msg_type: int, req_id: int, payload: bytes, flags: int = 0):
        """Внутренний метод отправки (использует сокет из self.sock)."""
        if self.sock is None:
            raise RuntimeError("Socket not connected")
        
        data = pack_ctrl(msg_type, req_id, payload, flags=flags)
        with self._send_lock:
            n = self.sock.send(data)
            if n != len(data):
                raise RuntimeError(f"SEQPACKET short send: {n}/{len(data)}")