"""
tests/conftest.py

Фикстуры для тестов ZeroLink v2.0.
"""

import pytest
import socket
import struct

# Импорты для создания фейковых объектов
from zerolink.server.main_server import MainIPCLeaseManager2P
from zerolink.core.protocol import MSG_HDR_FMT, MSG_MAGIC, MSG_VER

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
    return s1, s2