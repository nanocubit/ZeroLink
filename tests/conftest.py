"""
tests/conftest.py

Фикстуры для тестов ZeroLink v2.0.

Важно: conftest должен быть CPU-safe и не тянуть GPU/torch-зависимые
импорты на этапе collection, чтобы smoke-тесты могли запускаться
в минимальном окружении.
"""

import socket
import struct
from typing import Type

import pytest

from zerolink.core.protocol import MSG_HDR_FMT, MSG_MAGIC, MSG_VER


@pytest.fixture
def protocol_header_constants():
    """Базовые константы заголовка протокола для CPU-only тестов."""
    return {
        "fmt": MSG_HDR_FMT,
        "magic": MSG_MAGIC,
        "ver": MSG_VER,
        "size": struct.calcsize(MSG_HDR_FMT),
    }


@pytest.fixture
def fake_pool():
    """Фейковый пул, который просто запоминает вызовы free."""

    class SimpleFakePool:
        def __init__(self):
            self.freed = []

        def free(self, allocation):
            self.freed.append(allocation)

    return SimpleFakePool()


@pytest.fixture
def unix_socketpair():
    """Создает пару Unix Domain сокетов."""
    s1, s2 = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        yield s1, s2
    finally:
        s1.close()
        s2.close()


@pytest.fixture
def main_ipc_lease_manager_cls() -> Type:
    """
    Ленивая загрузка GPU/torch-зависимого менеджера.

    Использовать только в тестах, которым действительно нужен серверный слой.
    """
    pytest.importorskip("torch")
    from zerolink.server.main_server import MainIPCLeaseManager2P

    return MainIPCLeaseManager2P
