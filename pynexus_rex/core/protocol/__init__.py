"""
zerolink/core/protocol

Протокольный уровень ZeroLink v2.0.
Объединяет логику сообщений (messages.py) и транспортный уровень (framing.py).
"""

# ============================================================================
# Импорт и экспорт констант сообщений
# ============================================================================

from .messages import (
    # Control Plane
    MSG_HDR_FMT,
    MSG_MAGIC,
    MSG_VER,
    MSG_HDR_SIZE,
    MSG_HELLO,
    MSG_ALLOC,
    MSG_ACK,
    MSG_RELEASE,
    MSG_ERROR,
    MSG_PING,
    MSG_PONG,
    CTRL_FLAG_HAS_HASH,

    # Mapping Payload (PNXIPC10)
    _PNX_HDR_FMT,
    _PNX_SEG_FMT,
    _PNX_HDR_SIZE,
    _PNX_SEG_SIZE,
)

# ============================================================================
# Импорт и экспорт функций обработки данных (messages.py)
# ============================================================================

from .messages import (
    # Хеширование
    hash32,

    # Control Frame
    pack_ctrl,
    unpack_ctrl,

    # Alloc Wrapper
    pack_alloc_payload,
    unpack_alloc_payload,

    # Ack Payload
    pack_ack_payload,
    unpack_ack_payload,

    # Release Payload
    pack_release_payload,
    unpack_release_payload,

    # Error Payload
    pack_error_payload,
    unpack_error_payload,

    # Mapping Payload (Geometry)
    unpack_ipc_payload,
)

# ============================================================================
# Импорт и экспорт функций фрейминга (framing.py)
# ============================================================================

from .framing import (
    # Отправка
    send_frame,
    send_frame_with_fds,

    # Прием
    recv_frame,
    recv_frame_with_fds,
)

# Версия модуля для совместимости
__version__ = "2.0.0"