"""
zerolink/core/protocol/framing.py

Транспортный уровень для Unix Domain Sockets (SOCK_SEQPACKET).
Обрабатывает низкоуровневый ввод/вывод, передачу файловых дескрипторов (FD)
через SCM_RIGHTS и управление буферами.
"""

import socket
import array
from typing import Tuple, List

# Импортируем функции работы с сообщениями
from .messages import (
    MSG_HDR_FMT,
    MSG_HDR_SIZE,
    pack_ctrl,
    unpack_ctrl
)

# ============================================================================
# Константы
# ============================================================================

# Размер элемента массива int (для FD)
_INT_ITEM_SIZE = array.array("i").itemsize

# ============================================================================
# Функции отправки (Sending)
# ============================================================================

def send_frame(sock: socket.socket, data: bytes) -> None:
    """
    Отправляет фрейм БЕЗ файловых дескрипторов.
    
    ВАЖНО: Использует sock.send() (один вызов), а не sendall(),
    чтобы гарантировать атомарность для SOCK_SEQPACKET.
    """
    n = sock.send(data)
    if n != len(data):
        raise RuntimeError(f"SEQPACKET short send: {n}/{len(data)}")


def send_frame_with_fds(
    sock: socket.socket,
    msg_type: int,
    req_id: int,
    payload: bytes,
    fds: List[int],
    flags: int = 0
) -> None:
    """
    Отправляет фрейм С файловыми дескрипторами (SCM_RIGHTS).
    
    Args:
        sock: Сокет (должен быть AF_UNIX).
        msg_type: Тип сообщения (например, MSG_ALLOC).
        req_id: ID запроса.
        payload: Полезная нагрузка.
        fds: Список файловых дескрипторов для передачи.
        flags: Флаги контрольного заголовка.
    """
    # 1. Формируем данные (Header + Payload) используя протокол
    data = pack_ctrl(msg_type, req_id, payload, flags=flags)

    # 2. Подготавливаем ancillary data для передачи FD
    fds_arr = array.array("i", fds)
    ancdata = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds_arr.tobytes())]

    # 3. Отправляем
    # Используем sendmsg для одновременной отправки данных и FD
    n = sock.sendmsg([data], ancdata)
    if n != len(data):
        raise RuntimeError(f"SEQPACKET sendmsg short send: {n}/{len(data)}")


# ============================================================================
# Функции приема (Receiving)
# ============================================================================

def recv_frame_with_fds(
    sock: socket.socket,
    max_payload: int = 1 << 20,
    max_fds: int = 256
) -> Tuple[int, int, int, bytes, List[int]]:
    """
    Принимает фрейм, возможно с файловыми дескрипторами.
    
    Returns:
        (msg_type, flags, req_id, payload, fds)
        
    Raises:
        EOFError: Если соединение закрыто.
        RuntimeError: Если данные были обрезаны (MSG_TRUNC/CTRUNC).
        ValueError: Если заголовок невалиден.
    """
    # 1. Вычисляем размер буфера для ancillary data
    # CMSG_SPACE гарантирует достаточно места для заголовков CMSG и данных
    ancbuf_size = socket.CMSG_SPACE(max_fds * _INT_ITEM_SIZE)

    # 2. Устанавливаем флаги получения
    # MSG_CMSG_CLOEXEC критически важно для безопасности (предотвращает утечку FD в fork)
    recv_flags = getattr(socket, "MSG_CMSG_CLOEXEC", 0)

    # 3. Получаем данные
    data, ancdata, msg_flags, _addr = sock.recvmsg(max_payload, ancbuf_size, recv_flags)

    # 4. Проверки на ошибки
    if not data:
        raise EOFError("Peer closed connection")
    if msg_flags & socket.MSG_TRUNC:
        raise RuntimeError("MSG_TRUNC: Payload truncated (buffer too small)")
    if msg_flags & socket.MSG_CTRUNC:
        raise RuntimeError("MSG_CTRUNC: Ancillary data truncated (lost FDs)")

    # 5. Распаковываем заголовок (делегируем messages.py)
    try:
        mtype, mflags, req_id, payload = unpack_ctrl(data)
    except ValueError as e:
        raise ValueError(f"Failed to unpack control frame: {e}") from e

    # 6. Извлекаем FD из ancillary data
    fds = []
    for level, ctype, cdata in ancdata:
        if level == socket.SOL_SOCKET and ctype == socket.SCM_RIGHTS:
            arr = array.array("i")
            # Обрезаем "мусор" в конце буфера (CMSG_SPACE может выделить чуть больше)
            # Убеждаемся, что длина кратна размеру элемента
            valid_len = len(cdata) - (len(cdata) % _INT_ITEM_SIZE)
            arr.frombytes(cdata[:valid_len])
            fds.extend(arr.tolist())

    return mtype, mflags, req_id, payload, fds


def recv_frame(sock: socket.socket, max_payload: int = 1 << 20) -> Tuple[int, int, int, bytes]:
    """
    Принимает фрейм, предполагая, что FD НЕ передавались.
    Обертка над recv_frame_with_fds, игнорирующая ancillary data.
    """
    mtype, mflags, req_id, payload, _fds = recv_frame_with_fds(sock, max_payload, 0)
    return mtype, mflags, req_id, payload