"""Binary WebSocket payload for combined audio+JPEG (Moshi reply path).

Layout (little-endian, 40-byte header + payloads):
  bytes 0:3   magic ASCII 'AV01'
  bytes 4:7   version u32 (must be 1)
  bytes 8:11  frame_number u32 (absolute frame index)
  bytes 12:15 chunk_id u32 (1-based, same as JSON chunk_id)
  bytes 16:19 gen_ms u32 (rounded chunk gen time)
  bytes 20:23 sample_rate u32 (e.g. 24000)
  bytes 24:27 jpeg_len u32
  bytes 28:31 pcm_s16le_len u32 (byte length of PCM int16 mono)
  bytes 32:35 text_utf8_len u32
  bytes 36:39 chunks_done u32 (avatar chunk index, same as JSON chunks_done)
  (44-byte header total)
Then: jpeg_bytes | pcm_s16le_bytes | text_utf8_bytes

Mirror this in static/index_v3_binary.html (parseBinaryAvFrame).
"""
from __future__ import annotations

import struct

MAGIC = b"AV01"
VERSION = 1
_HEADER = struct.Struct("<4s10I")  # 4 + 40 = 44 bytes


def pack_av_frame(
    frame_number: int,
    chunk_id_ui: int,
    gen_ms: int,
    sample_rate: int,
    jpeg: bytes,
    pcm_s16le: bytes,
    text: str,
    chunks_done: int,
) -> bytes:
    tb = text.encode("utf-8", errors="replace")[:16000]
    if len(jpeg) > 12_000_000:
        raise ValueError("jpeg payload too large")
    hdr = _HEADER.pack(
        MAGIC,
        VERSION & 0xFFFFFFFF,
        int(frame_number) & 0xFFFFFFFF,
        int(chunk_id_ui) & 0xFFFFFFFF,
        int(gen_ms) & 0xFFFFFFFF,
        int(sample_rate) & 0xFFFFFFFF,
        len(jpeg) & 0xFFFFFFFF,
        len(pcm_s16le) & 0xFFFFFFFF,
        len(tb) & 0xFFFFFFFF,
        int(chunks_done) & 0xFFFFFFFF,
        0,
    )
    return hdr + jpeg + pcm_s16le + tb
