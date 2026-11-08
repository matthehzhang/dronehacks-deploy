"""Minimal, dependency-free MCAP reader for rosbag2 recordings.

Reads the data section front to back, so it also works on a bag whose recorder was killed before it
wrote the summary/footer (no metadata.yaml, truncated last chunk): every complete record is returned
and the torn tail is reported, not raised.

Chunk compression: none always; zstd through the standard library (Python >= 3.14, `compression.zstd`)
or the `zstandard` package; lz4 through the `lz4` package. A chunk whose codec is unavailable raises
with the codec name, so the fix (pip install zstandard / lz4) is obvious.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

MAGIC = b"\x89MCAP0\r\n"

OP_HEADER, OP_FOOTER, OP_SCHEMA, OP_CHANNEL, OP_MESSAGE, OP_CHUNK = 0x01, 0x02, 0x03, 0x04, 0x05, 0x06
OP_DATA_END = 0x0F


@dataclass
class Schema:
    id: int
    name: str
    encoding: str
    data: bytes


@dataclass
class Channel:
    id: int
    schema_id: int
    topic: str
    message_encoding: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Message:
    channel: Channel
    schema: Schema | None
    log_time_ns: int
    publish_time_ns: int
    data: bytes


def _zstd_decompress():
    try:
        from compression import zstd            # Python 3.14+
        return lambda b, n: zstd.decompress(b)
    except ImportError:
        pass
    try:
        import zstandard
        return lambda b, n: zstandard.ZstdDecompressor().decompress(b, max_output_size=n)
    except ImportError:
        return None


def _lz4_decompress():
    try:
        import lz4.frame
        return lambda b, n: lz4.frame.decompress(b)
    except ImportError:
        return None


_CODECS: dict[str, Callable | None] = {}


def _codec(name: str):
    if name not in _CODECS:
        _CODECS[name] = {"zstd": _zstd_decompress, "lz4": _lz4_decompress}.get(name, lambda: None)()
    fn = _CODECS[name]
    if fn is None:
        raise RuntimeError(f"MCAP chunk compressed with '{name}' and no decoder is installed "
                           f"(pip install {'zstandard' if name == 'zstd' else name})")
    return fn


def _str(buf: memoryview, o: int) -> tuple[str, int]:
    n = struct.unpack_from("<I", buf, o)[0]
    o += 4
    return bytes(buf[o:o + n]).decode("utf-8"), o + n


def _map(buf: memoryview, o: int) -> tuple[dict, int]:
    n = struct.unpack_from("<I", buf, o)[0]
    o += 4
    end, out = o + n, {}
    while o < end:
        k, o = _str(buf, o)
        v, o = _str(buf, o)
        out[k] = v
    return out, end


class McapReader:
    """Iterate the messages of one .mcap file. `truncated` tells whether the file ended mid-record."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.schemas: dict[int, Schema] = {}
        self.channels: dict[int, Channel] = {}
        self.truncated = False
        self.profile = ""

    def _records(self, buf: memoryview, top_level: bool) -> Iterator[tuple[int, memoryview]]:
        o, n = 0, len(buf)
        while o < n:
            if n - o < 9:
                self.truncated |= top_level
                return
            op, length = struct.unpack_from("<BQ", buf, o)
            o += 9
            if o + length > n:
                self.truncated |= top_level
                return
            yield op, buf[o:o + length]
            o += length

    def _handle(self, op: int, rec: memoryview, topics: set | None) -> Iterator[Message]:
        if op == OP_SCHEMA:
            sid = struct.unpack_from("<H", rec, 0)[0]
            name, o = _str(rec, 2)
            enc, o = _str(rec, o)
            dlen = struct.unpack_from("<I", rec, o)[0]
            self.schemas[sid] = Schema(sid, name, enc, bytes(rec[o + 4:o + 4 + dlen]))
        elif op == OP_CHANNEL:
            cid, sid = struct.unpack_from("<HH", rec, 0)
            topic, o = _str(rec, 4)
            enc, o = _str(rec, o)
            meta, o = _map(rec, o)
            self.channels[cid] = Channel(cid, sid, topic, enc, meta)
        elif op == OP_MESSAGE:
            cid, _seq, log_t, pub_t = struct.unpack_from("<HIQQ", rec, 0)
            ch = self.channels.get(cid)
            if ch is not None and (topics is None or ch.topic in topics):
                yield Message(ch, self.schemas.get(ch.schema_id), log_t, pub_t, bytes(rec[22:]))
        elif op == OP_CHUNK:
            _start, _end, usize, _crc = struct.unpack_from("<QQQI", rec, 0)
            comp, o = _str(rec, 28)
            rlen = struct.unpack_from("<Q", rec, o)[0]
            o += 8
            raw = rec[o:o + rlen]
            data = memoryview(bytes(raw) if comp == "" else _codec(comp)(bytes(raw), usize))
            for op2, rec2 in self._records(data, top_level=False):
                yield from self._handle(op2, rec2, topics)

    def messages(self, topics: set | None = None) -> Iterator[Message]:
        """All messages in file order (rosbag2 writes them in receive order). topics=None: every topic."""
        blob = self.path.read_bytes()
        if not blob.startswith(MAGIC):
            raise ValueError(f"{self.path}: not an MCAP file (bad magic)")
        buf = memoryview(blob)[len(MAGIC):]
        for op, rec in self._records(buf, top_level=True):
            if op == OP_HEADER:
                self.profile, _ = _str(rec, 0)
            elif op in (OP_DATA_END, OP_FOOTER):
                return                            # summary section follows: everything needed was already seen
            else:
                yield from self._handle(op, rec, topics)


def bag_files(path: str | Path) -> list[Path]:
    """A rosbag2 directory (bag_0.mcap, bag_1.mcap ... in split order) or a single .mcap file."""
    p = Path(path)
    if p.is_file():
        return [p]
    files = sorted(p.glob("*.mcap"), key=lambda f: (len(f.stem), f.stem))
    if not files:
        raise FileNotFoundError(f"no .mcap files in {p}")
    return files
