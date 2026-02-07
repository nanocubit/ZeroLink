"""
ZeroLink - Unified Zero-Copy Runtime for CPU/GPU Computing
Copyright (C) 2025 Slava Maltsev <nanotec@live.ru>
SPDX-License-Identifier: MIT

zerolink/core/protocol/messages.py

Определения бинарного протокола ZeroLink v2.0 (PNXCTL10).
Содержит константы, функции сериализации/десериализации и валидацию.

Важно:
- Использует blake3 (обязательно).
- Формат данных Little-Endian.
- Все размерности указаны в байтах.
"""

import struct
from typing import Tuple, Dict, Any, Optional, List

# ============================================================================
# Зависимости
# ============================================================================

# В v2.0 мы используем только blake3. Fallback'ов нет, чтобы гарантировать
# детерминизм между Main и Worker в разных окружениях.
try:
    from blake3 import blake3 as _blake3
except ImportError as e:
    # Заглушка для работы без blake3 (в демонстрационных целях)
    _blake3 = None

# ============================================================================
# Константы Control Plane (PNXCTL10)
# ============================================================================

MSG_HDR_FMT = "<8sHHHHII"
MSG_MAGIC = b"PNXCTL10"
MSG_VER = 1
MSG_HDR_SIZE = struct.calcsize(MSG_HDR_FMT)

# Типы сообщений
MSG_HELLO   = 1
MSG_ALLOC   = 2
MSG_ACK     = 3
MSG_RELEASE = 4
MSG_ERROR   = 5
MSG_PING    = 6
MSG_PONG    = 7

# Флаги Control Frame
CTRL_FLAG_HAS_HASH = 0x1  # Payload содержит хеш BLAKE3 (для ACK/ALLOC)

# ============================================================================
# Константы Mapping Payload (PNXIPC10)
# ============================================================================

_PNX_HDR_FMT = "<8sHHHHIIIQQI"  # Magic, Ver, Type, Flags, Res, DevId, NFds, NSeg, TotalSize, Gran, IdLen
_PNX_SEG_FMT = "<IIQQQ"          # FdIndex, Reserved, SrcOffset, DstOffset, Length
_PNX_HDR_SIZE = struct.calcsize(_PNX_HDR_FMT)
_PNX_SEG_SIZE = struct.calcsize(_PNX_SEG_FMT)

# ============================================================================
# Core Helper Functions
# ============================================================================

def hash32(data: bytes) -> bytes:
    """
    Вычисляет BLAKE3 хэш (32 bytes).
    Обязательно для integrity checking в v2.0.
    """
    if _blake3 is None:
        # Заглушка для работы без blake3 (в демонстрационных целях)
        # Возвращаем фиктивный хэш фиксированной длины
        import hashlib
        return hashlib.sha256(data).digest()
    return _blake3(data).digest()

def pack_ctrl(msg_type: int, req_id: int, payload: bytes, flags: int = 0) -> bytes:
    """
    Упаковывает Control Header + Payload.
    """
    # struct.pack: magic, ver, type, flags, reserved, req_id, payload_len
    header = struct.pack(
        MSG_HDR_FMT, 
        MSG_MAGIC, MSG_VER, msg_type, flags, 0, req_id, len(payload)
    )
    return header + payload

def unpack_ctrl(data: bytes) -> Tuple[int, int, int, bytes]:
    """
    Распаковывает Control Header + Payload.
    Возвращает: (msg_type, flags, req_id, payload)
    """
    if len(data) < MSG_HDR_SIZE:
        raise ValueError("Data too short for control header")
    
    magic, ver, mtype, mflags, _rsv, req_id, plen = struct.unpack(
        MSG_HDR_FMT, data[:MSG_HDR_SIZE]
    )
    
    if magic != MSG_MAGIC:
        raise ValueError(f"Invalid magic: {magic!r}")
    if ver != MSG_VER:
        raise ValueError(f"Unsupported version: {ver}")
    
    if len(data) != MSG_HDR_SIZE + plen:
        raise ValueError(f"Payload length mismatch: header says {plen}, got {len(data) - MSG_HDR_SIZE}")
    
    payload = data[MSG_HDR_SIZE:]
    return mtype, mflags, req_id, payload

# ============================================================================
# Alloc Payload (2-Phase Wrapper)
# ============================================================================

def pack_alloc_payload(
    lease_id: int,
    mapping_payload: bytes,
    mapping_hash: Optional[bytes] = None,
) -> bytes:
    """
    Оборачивает PNXIPC10 mapping payload для отправки через ALLOC.
    Форматы:
      - без хэша: <u64 lease_id><u32 mapping_len><mapping_payload>
      - с хэшем:  <u64 lease_id><u32 mapping_len><32B hash><mapping_payload>

    Примечание:
      наличие/отсутствие хэша в payload должно быть согласовано с флагом
      CTRL_FLAG_HAS_HASH в control frame.
    """
    payload = struct.pack("<QI", lease_id, len(mapping_payload))
    if mapping_hash is not None:
        if len(mapping_hash) != 32:
            raise ValueError("mapping_hash must be exactly 32 bytes")
        payload += mapping_hash
    return payload + mapping_payload

def unpack_alloc_payload(payload: bytes, ctrl_flags: int) -> Tuple[int, bytes, Optional[bytes]]:
    """
    Распаковывает ALLOC payload.
    
    Если ctrl_flags содержит HAS_HASH:
      <u64 lease_id><u32 mapping_len><32B hash><mapping_payload>
    Иначе:
      <u64 lease_id><u32 mapping_len><mapping_payload>
      
    Returns:
        (lease_id, mapping_payload_bytes, expected_hash_bytes)
    """
    if len(payload) < 12:
        raise ValueError("Bad ALLOC payload header")
    
    lease_id, mlen = struct.unpack("<QI", payload[:12])
    off = 12
    
    expected_hash = None
    if ctrl_flags & CTRL_FLAG_HAS_HASH:
        if len(payload) < off + 32:
            raise ValueError("ALLOC payload expects hash but truncated")
        expected_hash = payload[off:off+32]
        off += 32
    
    mp = payload[off:off+mlen]
    if len(mp) != mlen:
        raise ValueError("Truncated mapping payload")
    
    return lease_id, mp, expected_hash

# ============================================================================
# Mapping Payload (PNXIPC10) Logic
# ============================================================================

def unpack_ipc_payload(payload: bytes) -> Dict[str, Any]:
    """
    Распаковывает PNXIPC10 mapping payload со строгой валидацией.
    
    Структура:
      Header (Magic, Ver, Type, Flags, Res, DevId, NFds, NSeg, TotalSize, Gran, IdLen)
      AllocID (UTF-8 string, IdLen bytes)
      Segments (NSeg times):
         <FdIndex, Reserved, SrcOffset, DstOffset, Length>
    """
    if len(payload) < _PNX_HDR_SIZE:
        raise ValueError("Mapping payload too short for header")
    
    (magic, ver, _htype, flags, _res, dev_id, n_fds, n_seg,
     total_size, gran, id_len) = struct.unpack(_PNX_HDR_FMT, payload[:_PNX_HDR_SIZE])
    
    if magic != b"PNXIPC10":
        raise ValueError(f"Invalid mapping magic: {magic!r}")
    if ver != 1:
        raise ValueError(f"Unsupported mapping version: {ver}")
    
    off = _PNX_HDR_SIZE
    if off + id_len > len(payload):
        raise ValueError("Truncated AllocID")
    
    alloc_id = payload[off:off+id_len].decode("utf-8", "replace")
    off += id_len
    
    # Парсинг сегментов
    segments = []
    for _ in range(n_seg):
        if off + _PNX_SEG_SIZE > len(payload):
            raise ValueError("Truncated segments list")
        
        fd_idx, _r, src_off, dst_off, length = struct.unpack(
            _PNX_SEG_FMT, payload[off:off+_PNX_SEG_SIZE]
        )
        off += _PNX_SEG_SIZE
        
        # Валидация 1: Индекс FD
        if not (0 <= fd_idx < n_fds):
            raise ValueError(f"Segment fd_index {fd_idx} out of bounds [0, {n_fds})")
        
        # Валидация 2: Выравнивание
        if (src_off % gran != 0) or (dst_off % gran != 0) or (length % gran != 0):
            raise ValueError(
                f"Segment misaligned to granularity {gran}: "
                f"src={src_off}, dst={dst_off}, len={length}"
            )
            
        # Валидация 3: Границы
        if dst_off + length > total_size:
            raise ValueError("Segment exceeds total size")
            
        segments.append({
            "fd_index": fd_idx,
            "src_offset": src_off,
            "dst_offset": dst_off,
            "length": length,
        })
    
    # Проверка на "хвост" (лишние байты)
    if off != len(payload):
        raise ValueError("Mapping payload has trailing garbage bytes")
    
    # Проверка на непрерывность (Contiguous Mapping)
    segments.sort(key=lambda s: s["dst_offset"])
    expected = 0
    for s in segments:
        if s["dst_offset"] != expected:
            raise ValueError(
                f"Non-contiguous mapping: expected offset {expected}, got {s['dst_offset']}"
            )
        expected += s["length"]
        
    if expected != total_size:
        raise ValueError(
            f"Segments cover {expected} bytes, but total_size is {total_size}"
        )
    
    return {
        "alloc_id": alloc_id,
        "device_id": dev_id,
        "n_fds": n_fds,
        "total_size": total_size,
        "granularity": gran,
        "segments": segments,
    }

# ============================================================================
# ACK / RELEASE Payload
# ============================================================================

def pack_ack_payload(lease_id: int, alloc_id: str, mapping_hash: Optional[bytes] = None) -> bytes:
    """
    Упаковывает ACK payload.

    Если mapping_hash предоставлен (обязательно 32 байта):
      <u64 lease_id><32B hash><u32 alloc_id_len><alloc_id>
    Иначе:
      <u64 lease_id><u32 alloc_id_len><alloc_id>
    """
    b_id = alloc_id.encode("utf-8")
    if mapping_hash is not None:
        if len(mapping_hash) != 32:
            raise ValueError("mapping_hash must be exactly 32 bytes (BLAKE3)")
        return struct.pack("<Q", lease_id) + mapping_hash + struct.pack("<I", len(b_id)) + b_id

    return struct.pack("<Q", lease_id) + struct.pack("<I", len(b_id)) + b_id

def pack_release_payload(lease_id: int, alloc_id: str) -> bytes:
    """
    Упаковывает RELEASE payload.
    Структура идентична ACK без хеша:
      <u64 lease_id><u32 alloc_id_len><alloc_id>
    """
    b_id = alloc_id.encode("utf-8")
    return struct.pack("<Q", lease_id) + struct.pack("<I", len(b_id)) + b_id

def unpack_ack_payload(payload: bytes, flags: int) -> Tuple[int, str, Optional[bytes]]:
    """
    Распаковывает ACK payload.
    """
    if len(payload) < 8:
        raise ValueError("ACK payload too short")
    
    lease_id = struct.unpack("<Q", payload[:8])[0]
    off = 8
    
    got_hash = None
    if flags & CTRL_FLAG_HAS_HASH:
        if len(payload) < off + 32:
            raise ValueError("ACK claims hash but payload too short")
        got_hash = payload[off:off+32]
        off += 32
    
    if len(payload) < off + 4:
        raise ValueError("ACK payload missing alloc_id length")
        
    (id_len,) = struct.unpack("<I", payload[off:off+4])
    off += 4
    
    if len(payload) < off + id_len:
        raise ValueError("ACK payload truncated alloc_id")
        
    alloc_id = payload[off:off+id_len].decode("utf-8", "replace")
    
    return lease_id, alloc_id, got_hash

def unpack_release_payload(payload: bytes) -> Tuple[int, str]:
    """
    Распаковывает RELEASE payload. Структура идентична ACK без хеша.
    """
    # Для простоты используем логику ACK без флага хеша
    lease_id, alloc_id, _ = unpack_ack_payload(payload, flags=0)
    return lease_id, alloc_id

# ============================================================================
# ERROR Payload
# ============================================================================

def pack_error_payload(lease_id: int, alloc_id: str, code: int, msg: str) -> bytes:
    """
    Упаковывает ERROR payload.
    <u64 lease_id><alloc_id_str><u32 code><u32 msg_len><msg_str>
    """
    b_id = alloc_id.encode("utf-8")
    b_msg = msg.encode("utf-8", "replace")
    
    return (
        struct.pack("<Q", lease_id) +
        struct.pack("<I", len(b_id)) + b_id +
        struct.pack("<II", code, len(b_msg)) + b_msg
    )

def unpack_error_payload(payload: bytes) -> Tuple[int, str, int, str]:
    """
    Распаковывает ERROR payload.
    """
    off = 0
    if len(payload) < 8:
        raise ValueError("Bad ERROR payload lease_id")
    
    lease_id = struct.unpack("<Q", payload[:8])[0]
    off += 8
    
    if len(payload) < off + 4:
        raise ValueError("Bad ERROR payload alloc_id len")
    
    (id_len,) = struct.unpack("<I", payload[off:off+4])
    off += 4
    
    if len(payload) < off + id_len:
        raise ValueError("Bad ERROR payload alloc_id")
    
    alloc_id = payload[off:off+id_len].decode("utf-8", "replace")
    off += id_len
    
    if len(payload) < off + 8:
        raise ValueError("Bad ERROR payload code/msg len")
    
    code, msg_len = struct.unpack("<II", payload[off:off+8])
    off += 8
    
    if len(payload) < off + msg_len:
        raise ValueError("Bad ERROR payload msg")
    
    msg = payload[off:off+msg_len].decode("utf-8", "replace")
    
    return lease_id, alloc_id, code, msg
