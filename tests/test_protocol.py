"""
tests/test_protocol.py

Тесты для протокола ZeroLink v2.0.
"""

import struct
from zerolink.core.protocol import (
    pack_ctrl, unpack_ctrl,
    pack_alloc_payload, unpack_alloc_payload,
    pack_ack_payload, unpack_ack_payload,
    hash32, MSG_ALLOC, MSG_ACK, CTRL_FLAG_HAS_HASH
)

def test_ctrl_frame_pack_unpack():
    payload = b"test_data"
    data = pack_ctrl(MSG_ALLOC, 123, payload)
    mtype, flags, req_id, pl = unpack_ctrl(data)
    assert mtype == MSG_ALLOC
    assert req_id == 123
    assert pl == payload

def test_hash_integrity():
    data = b"important_mapping_data"
    h1 = hash32(data)
    h2 = hash32(data)
    assert h1 == h2
    assert len(h1) == 32 # BLAKE3 output size

def test_alloc_wrapper_with_hash():
    mapping = b"mapping_payload"
    h = hash32(mapping)
    
    # Пакуем с хешем
    wrapper = pack_alloc_payload(100, mapping, h)
    
    # Распаковываем
    lid, mp, exp_h = unpack_alloc_payload(wrapper, ctrl_flags=CTRL_FLAG_HAS_HASH)
    assert lid == 100
    assert mp == mapping
    assert exp_h == h