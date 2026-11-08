"""Decode ROS 2 CDR messages from the `ros2msg` schema text an MCAP bag embeds. Dependency-free.

The schema of each channel is the full .msg definition of the type, followed by every dependency as
`MSG: pkg/Type` blocks separated by `====` lines. That is enough to decode any message in the bag,
px4_msgs and isaac_ros types included, without having those packages installed.

Decoded messages are plain dicts (nested types are dicts, arrays of numbers are lists).
"""
from __future__ import annotations

import re
import struct

PRIMITIVES = {
    "bool": ("?", 1), "byte": ("B", 1), "char": ("B", 1), "uint8": ("B", 1), "int8": ("b", 1),
    "int16": ("h", 2), "uint16": ("H", 2), "int32": ("i", 4), "uint32": ("I", 4),
    "int64": ("q", 8), "uint64": ("Q", 8), "float32": ("f", 4), "float64": ("d", 8),
}
STRINGS = ("string", "wstring")
_SEP = re.compile(r"^=+\s*$", re.M)
_FIELD = re.compile(r"^\s*([A-Za-z0-9_/]+)(<=\d+)?(\[(<=)?(\d*)\])?\s+([A-Za-z0-9_]+)\s*(=)?")


class Field:
    __slots__ = ("name", "type", "array", "length")

    def __init__(self, name, typ, array, length):
        self.name, self.type, self.array, self.length = name, typ, array, length  # array: None|'fixed'|'seq'


def _qualify(typ: str, pkg: str) -> str:
    if typ in PRIMITIVES or typ in STRINGS:
        return typ
    if typ == "Header":
        return "std_msgs/msg/Header"
    parts = typ.split("/")
    if len(parts) == 1:
        return f"{pkg}/msg/{typ}"
    if len(parts) == 2:
        return f"{parts[0]}/msg/{parts[1]}"
    return typ


def parse_schema(name: str, text: str) -> dict[str, list[Field]]:
    """{fully/qualified/Type: [Field, ...]} for the root type and every embedded dependency."""
    types = {}
    blocks = _SEP.split(text)
    for i, block in enumerate(blocks):
        lines = block.strip("\n").splitlines()
        if i == 0:
            tname = name
        else:
            head = next((ln for ln in lines if ln.startswith("MSG:")), None)
            if head is None:
                continue
            tname = _qualify(head.split(":", 1)[1].strip(), "")
            lines = lines[lines.index(head) + 1:]
        pkg = tname.split("/")[0]
        fields = []
        for ln in lines:
            ln = ln.split("#", 1)[0]
            if not ln.strip():
                continue
            m = _FIELD.match(ln)
            if not m or m.group(7):                  # unparsable, or a constant (NAME=value)
                continue
            typ, _bound, arr, bounded, size, fname, _ = m.groups()
            if arr is None:
                array, length = None, 0
            elif size and not bounded:
                array, length = "fixed", int(size)
            else:
                array, length = "seq", 0
            fields.append(Field(fname, _qualify(typ, pkg), array, length))
        types[tname] = fields
    return types


class _Cursor:
    __slots__ = ("buf", "o", "fmt")

    def __init__(self, buf: bytes):
        if len(buf) < 4:
            raise ValueError("CDR payload shorter than its encapsulation header")
        self.buf = buf
        self.fmt = "<" if buf[1] & 1 else ">"      # 0x0001 CDR_LE / 0x0000 CDR_BE
        self.o = 4

    def align(self, n):
        r = (self.o - 4) % n
        if r:
            self.o += n - r

    def prim(self, code, size):
        self.align(size)
        v = struct.unpack_from(self.fmt + code, self.buf, self.o)[0]
        self.o += size
        return v

    def prims(self, code, size, count):
        if count == 0:
            return []
        self.align(size)
        v = list(struct.unpack_from(f"{self.fmt}{count}{code}", self.buf, self.o))
        self.o += size * count
        return v

    def string(self):
        n = self.prim("I", 4)
        s = self.buf[self.o:self.o + n]
        self.o += n
        return s[:-1].decode("utf-8", "replace") if n else ""


class Decoder:
    """decoder = Decoder(schema.name, schema.data.decode()); decoder(payload) -> dict"""

    def __init__(self, name: str, text: str):
        self.root = _qualify(name, "")
        self.types = parse_schema(self.root, text)

    def __call__(self, payload: bytes) -> dict:
        return self._msg(_Cursor(payload), self.root)

    def _msg(self, c: _Cursor, tname: str) -> dict:
        fields = self.types.get(tname)
        if fields is None:
            raise KeyError(f"type {tname} not defined in the schema")
        out = {}
        for f in fields:
            if f.array is None:
                out[f.name] = self._one(c, f.type)
            else:
                n = f.length if f.array == "fixed" else c.prim("I", 4)
                if f.type in PRIMITIVES:
                    code, size = PRIMITIVES[f.type]
                    out[f.name] = c.prims(code, size, n)
                else:
                    out[f.name] = [self._one(c, f.type) for _ in range(n)]
        return out

    def _one(self, c: _Cursor, typ: str):
        p = PRIMITIVES.get(typ)
        if p is not None:
            return c.prim(*p)
        if typ in STRINGS:
            return c.string()
        return self._msg(c, typ)
