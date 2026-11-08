"""Tests for the bag replay tool. Standard library only:  python -m unittest discover -s replay/tests"""
from __future__ import annotations

import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from skyreplay.cdr import PRIMITIVES, Decoder, parse_schema  # noqa: E402
from skyreplay.course_map import DEFAULT_MAP, course_start, load_map  # noqa: E402
from skyreplay.extract import collapse_repeats, compose, extract, ned_to_odom  # noqa: E402
from skyreplay.mcap_reader import MAGIC, McapReader  # noqa: E402

# ------------------------------------------------------------------ schemas (trimmed, real field names)
TIME = "int32 sec\nuint32 nanosec\n"
HEADER = "builtin_interfaces/Time stamp\nstring frame_id\n"
SEP = "=" * 80 + "\n"
ODOM = ("std_msgs/Header header\nstring child_frame_id\ngeometry_msgs/PoseWithCovariance pose\n"
        "geometry_msgs/TwistWithCovariance twist\n"
        + SEP + "MSG: std_msgs/Header\n" + HEADER
        + SEP + "MSG: builtin_interfaces/Time\n" + TIME
        + SEP + "MSG: geometry_msgs/PoseWithCovariance\nPose pose\nfloat64[36] covariance\n"
        + SEP + "MSG: geometry_msgs/Pose\nPoint position\nQuaternion orientation\n"
        + SEP + "MSG: geometry_msgs/Point\nfloat64 x\nfloat64 y\nfloat64 z\n"
        + SEP + "MSG: geometry_msgs/Quaternion\nfloat64 x 0\nfloat64 y 0\nfloat64 z 0\nfloat64 w 1\n"
        + SEP + "MSG: geometry_msgs/TwistWithCovariance\nTwist twist\nfloat64[36] covariance\n"
        + SEP + "MSG: geometry_msgs/Twist\nVector3 linear\nVector3 angular\n"
        + SEP + "MSG: geometry_msgs/Vector3\nfloat64 x\nfloat64 y\nfloat64 z\n")
LPOS = ("uint32 MESSAGE_VERSION = 1\nuint64 timestamp\nbool xy_valid\nbool z_valid\n"
        "float32 x # North\nfloat32 y\nfloat32 z\nfloat32 heading\nfloat32 dist_bottom\n")
STRING = "string data\n"
STATUS = "std_msgs/Header header\nuint8 vo_state\nfloat64 track_execution_time\n" + SEP + "MSG: std_msgs/Header\n" + HEADER + SEP + "MSG: builtin_interfaces/Time\n" + TIME


# ------------------------------------------------------------------ independent CDR encoder for the tests
class Enc:
    def __init__(self):
        self.b = bytearray(b"\x00\x01\x00\x00")

    def align(self, n):
        while (len(self.b) - 4) % n:
            self.b.append(0)

    def prim(self, code, size, v):
        self.align(size)
        self.b += struct.pack("<" + code, v)

    def string(self, s):
        raw = s.encode() + b"\x00"
        self.prim("I", 4, len(raw))
        self.b += raw


def encode(schema_name, text, msg):
    types = parse_schema(schema_name, text)
    e = Enc()

    def one(typ, v):
        if typ in PRIMITIVES:
            e.prim(*PRIMITIVES[typ], v)
        elif typ == "string":
            e.string(v)
        else:
            enc_msg(typ, v)

    def enc_msg(tname, m):
        for f in types[tname]:
            v = m[f.name]
            if f.array is None:
                one(f.type, v)
            else:
                if f.array == "seq":
                    e.prim("I", 4, len(v))
                for item in v:
                    one(f.type, item)

    enc_msg(schema_name, msg)
    return bytes(e.b)


def odom_msg(x, y, z, yaw, t=0):
    return {"header": {"stamp": {"sec": int(t), "nanosec": 0}, "frame_id": "odom"}, "child_frame_id": "base_link",
            "pose": {"pose": {"position": {"x": x, "y": y, "z": z},
                              "orientation": {"x": 0.0, "y": 0.0, "z": math.sin(yaw / 2), "w": math.cos(yaw / 2)}},
                     "covariance": [0.0] * 36},
            "twist": {"twist": {"linear": {"x": 0.0, "y": 0.0, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": 0.0}},
                      "covariance": [0.0] * 36}}


# ------------------------------------------------------------------ minimal MCAP writer
def _s(x: str) -> bytes:
    b = x.encode()
    return struct.pack("<I", len(b)) + b


def _rec(op, body):
    return struct.pack("<BQ", op, len(body)) + body


class McapWriter:
    def __init__(self):
        self.out = bytearray(MAGIC) + _rec(0x01, _s("ros2") + _s("test"))
        self.schemas, self.channels, self.chunk = {}, {}, bytearray()

    def message(self, topic, schema_name, text, msg, t_s, in_chunk=False):
        if schema_name not in self.schemas:
            sid = len(self.schemas) + 1
            self.schemas[schema_name] = sid
            self.out += _rec(0x03, struct.pack("<H", sid) + _s(schema_name) + _s("ros2msg")
                             + struct.pack("<I", len(text)) + text.encode())
        if topic not in self.channels:
            cid = len(self.channels) + 1
            self.channels[topic] = cid
            self.out += _rec(0x04, struct.pack("<HH", cid, self.schemas[schema_name]) + _s(topic) + _s("cdr")
                             + struct.pack("<I", 0))
        ns = int(1_700_000_000e9 + t_s * 1e9)
        rec = _rec(0x05, struct.pack("<HIQQ", self.channels[topic], 0, ns, ns) + encode(schema_name, text, msg))
        if in_chunk:
            self.chunk += rec
        else:
            self.out += rec

    def flush_chunk(self):
        body = (struct.pack("<QQQI", 0, 0, len(self.chunk), 0) + _s("")
                + struct.pack("<Q", len(self.chunk)) + bytes(self.chunk))
        self.out += _rec(0x06, body)
        self.chunk = bytearray()

    def save(self, path, truncate_tail=False):
        data = bytes(self.out)
        if truncate_tail:
            data += _rec(0x05, b"\x01\x00" + b"\x00" * 40)[:20]       # torn record, as a killed recorder leaves
        else:
            data += _rec(0x0F, struct.pack("<I", 0)) + _rec(0x02, b"\x00" * 20) + MAGIC
        Path(path).write_bytes(data)


# ------------------------------------------------------------------ tests
class TestCdr(unittest.TestCase):
    def test_round_trip_alignment_arrays_strings(self):
        text = ("uint8 a\nfloat64 b\nstring s\nint16[] seq\nfloat32[3] fixed\nuint8 c\nint64 d\n"
                "pkg/Inner[] inner\n" + SEP + "MSG: pkg/Inner\nbool flag\nfloat64 v\nstring name\n")
        msg = {"a": 7, "b": -1.25, "s": "héllo", "seq": [1, -2, 3], "fixed": [0.5, 1.5, 2.5], "c": 9, "d": -2 ** 40,
               "inner": [{"flag": True, "v": 3.0, "name": "x"}, {"flag": False, "v": -4.0, "name": ""}]}
        out = Decoder("pkg/msg/Outer", text)(encode("pkg/msg/Outer", text, msg))
        self.assertEqual(out, msg)

    def test_constants_and_defaults_are_not_fields(self):
        types = parse_schema("px4_msgs/msg/VehicleLocalPosition", LPOS)
        self.assertEqual([f.name for f in types["px4_msgs/msg/VehicleLocalPosition"]],
                         ["timestamp", "xy_valid", "z_valid", "x", "y", "z", "heading", "dist_bottom"])
        types = parse_schema("nav_msgs/msg/Odometry", ODOM)
        self.assertEqual(len(types["geometry_msgs/msg/Quaternion"]), 4)

    def test_odometry(self):
        m = odom_msg(1.0, -2.0, 0.5, 0.3)
        self.assertEqual(Decoder("nav_msgs/msg/Odometry", ODOM)(encode("nav_msgs/msg/Odometry", ODOM, m)), m)


class TestMcap(unittest.TestCase):
    def test_reads_chunked_and_plain_and_survives_torn_tail(self):
        w = McapWriter()
        w.message("/a", "std_msgs/msg/String", STRING, {"data": "one"}, 0.0)
        w.message("/a", "std_msgs/msg/String", STRING, {"data": "two"}, 0.1, in_chunk=True)
        w.message("/b", "std_msgs/msg/String", STRING, {"data": "three"}, 0.2, in_chunk=True)
        w.flush_chunk()
        with tempfile.TemporaryDirectory() as d:
            for torn in (False, True):
                p = Path(d) / f"t{int(torn)}.mcap"
                w.save(p, truncate_tail=torn)
                r = McapReader(p)
                got = [(m.channel.topic, Decoder(m.schema.name, m.schema.data.decode())(m.data)["data"])
                       for m in r.messages()]
                self.assertEqual(got, [("/a", "one"), ("/a", "two"), ("/b", "three")])
                self.assertEqual(r.truncated, torn)
                self.assertEqual([m.channel.topic for m in McapReader(p).messages({"/b"})], ["/b"])


class TestFrames(unittest.TestCase):
    FIX = (-2.515, 0.914, math.radians(90))

    def test_px4_target_lands_on_the_tunnel_entry(self):
        # refnav's first target in run 20261107-235044 (PX4 local NED) vs the map's tunnel_entry (2.3114, 2.3692)
        ox, oy, _ = ned_to_odom(-4.829154372174663, 1.453442871517621)
        cx, cy, _ = compose(self.FIX, (ox, oy, 0.0))
        self.assertAlmostEqual(cx, 2.3142, places=3)
        self.assertAlmostEqual(cy, 2.3674, places=3)

    def test_vio_forward_is_course_north_when_launched_facing_north(self):
        cx, cy, cyaw = compose(self.FIX, (1.0, 0.0, 0.0))
        self.assertAlmostEqual(cx, -2.515)
        self.assertAlmostEqual(cy, 1.914)
        self.assertAlmostEqual(cyaw, math.radians(90))

    def test_px4_heading_to_odom_yaw(self):
        self.assertAlmostEqual(ned_to_odom(0, 0, math.pi / 2)[2], 0.0)       # NED heading east = odom +x
        self.assertAlmostEqual(ned_to_odom(0, 0, 0.0)[2], math.pi / 2)


class TestMap(unittest.TestCase):
    def test_default_map(self):
        m = load_map(DEFAULT_MAP)
        names = {g["name"]: g for g in m["mesh"]["groups"]}
        self.assertNotIn("Net roof", names)
        self.assertEqual(names["Tube oval 58 x 72 x 60"]["kind"], "obstacle")
        self.assertEqual(names["Floor 20 x 20 ft"]["kind"], "floor")
        self.assertEqual(names["Landing pad 5 ft"]["kind"], "mark")
        self.assertEqual([g["name"] for g in m["goals"]], ["tunnel_entry", "tunnel_exit", "pad"])
        self.assertEqual(len(m["mesh"]["verts"]) % 3, 0)
        self.assertTrue(all(i < len(m["mesh"]["verts"]) // 3 for g in m["mesh"]["groups"] for i in g["faces"]))

    def test_course_start_prefers_spawn_xy(self):
        self.assertEqual(course_start({"start": {"xy": [1, 2], "yaw_deg": 0}}), (1.0, 2.0, 0.0))
        self.assertEqual(course_start({"start": {"xy": [1, 2], "spawn_xy": [3, 4], "yaw_deg": 0}})[:2], (3.0, 4.0))
        self.assertIsNone(course_start({}))


class TestExtract(unittest.TestCase):
    def _bag(self, d, with_drift=True):
        """A 4 s flight: VIO moves 1 m forward (north), PX4 agrees, the localizer says 0.2 m further east."""
        w = McapWriter()
        fix = [-2.515, 0.914, math.pi / 2]
        for i in range(41):
            t = i * 0.1
            f = min(1.0, t / 3.0)
            w.message("/visual_slam/tracking/odometry", "nav_msgs/msg/Odometry", ODOM, odom_msg(f, 0.0, 0.5 * f, 0.0, t), t)
            w.message("/fmu/out/vehicle_local_position_v1", "px4_msgs/msg/VehicleLocalPosition", LPOS,
                      {"timestamp": 0, "xy_valid": True, "z_valid": True, "x": 0.0, "y": f, "z": -0.5 * f,
                       "heading": math.pi / 2, "dist_bottom": 0.1}, t + 0.01)
            w.message("/visual_slam/status", "isaac_ros_visual_slam_interfaces/msg/VisualSlamStatus", STATUS,
                      {"header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "map"},
                       "vo_state": 2 if 20 <= i < 23 else 1, "track_execution_time": 0.0}, t + 0.02)
            if with_drift and i % 5 == 0:
                w.message("/coursenav/drift", "std_msgs/msg/String", STRING, {"data": json.dumps(
                    {"stamp": t, "delta": [0.2, 0.0, 0.0], "nominal": [0, 0, 0], "fixture": fix, "sigma_xy": 0.03,
                     "accepted": 5, "rejected": 0, "resets": 0})}, t + 0.03)
                w.message("/coursenav/status", "std_msgs/msg/String", STRING, {"data": json.dumps(
                    {"t": t, "x": fix[0] + 0.2, "y": fix[1] + f, "yaw_deg": 90.0, "sigma_xy": 0.04})}, t + 0.04)
        p = Path(d) / "bag"
        p.mkdir()
        w.save(p / "bag_0.mcap")
        return p

    def test_traces_land_where_expected(self):
        with tempfile.TemporaryDirectory() as d:
            r = extract(self._bag(d))
            self.assertEqual(r["fixture"]["source"].split()[0], "/coursenav/drift")
            T = r["traces"]
            self.assertEqual(set(T), {"pf", "nav", "px4", "vio"})
            for k, dx in (("vio", 0.0), ("px4", 0.0), ("nav", 0.2), ("pf", 0.2)):
                self.assertAlmostEqual(T[k]["x"][-1], -2.515 + dx, places=3, msg=k)
                self.assertAlmostEqual(T[k]["y"][-1], 1.914, places=3, msg=k)
            self.assertAlmostEqual(T["vio"]["z"][-1], 0.6, places=3)              # landed 0.10 + 0.5 m rise
            self.assertAlmostEqual(T["px4"]["z"][-1], 0.6, places=3)
            self.assertAlmostEqual(T["vio"]["yaw"][-1], math.pi / 2, places=3)
            self.assertAlmostEqual(T["px4"]["yaw"][-1], math.pi / 2, places=3)
            texts = [e["text"] for e in r["events"]]
            self.assertTrue(any("vo_state -> 2" in s for s in texts))
            self.assertTrue(any("vo_state -> 1" in s for s in texts))
            self.assertTrue(any("map correction active" in s for s in texts))

    def test_without_fixture_vio_is_not_placed(self):
        with tempfile.TemporaryDirectory() as d:
            r = extract(self._bag(d, with_drift=False))
            self.assertIsNone(r["fixture"])
            self.assertNotIn("vio", r["traces"])
            r = extract(Path(d) / "bag", fallback_fixture=(0.0, 0.0, 0.0), fallback_source="test")
            self.assertAlmostEqual(r["traces"]["vio"]["x"][-1], 1.0, places=3)     # facing east: forward = +x
            self.assertEqual(r["fixture"]["source"], "test")

    def test_cli_writes_self_contained_html(self):
        import replay_bag
        with tempfile.TemporaryDirectory() as d:
            bag = self._bag(d)
            out = Path(d) / "r.html"
            self.assertEqual(replay_bag.main([str(bag), "-o", str(out), "--csv"]), 0)
            html = out.read_text(encoding="utf-8")
            self.assertIn("const MAP=", html)
            self.assertIn("const RUN=", html)
            self.assertNotIn("NaN", html.split("const RUN=")[1].split("\n")[0])
            self.assertNotIn("/*__DATA__*/", html)
            self.assertTrue(out.with_suffix(".csv").read_text().startswith("estimate,t_s"))

    def test_max_gap_interpolates(self):
        from skyreplay.extract import max_gap
        moving = {"t": [i * 0.1 for i in range(31)], "x": [i * 0.1 for i in range(31)], "y": [0.0] * 31}
        sparse = {"t": [0.0, 1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0, 3.0], "y": [0.2] * 4}
        gap, _t = max_gap(moving, sparse)
        self.assertAlmostEqual(gap, 0.2, places=6)        # a zero-order hold would report ~0.22+

    def test_collapse_repeats(self):
        ev = [{"t": 1.0, "kind": "log", "text": "[WARN] vs: Delta 66.6 ms"},
              {"t": 1.5, "kind": "log", "text": "[WARN] vs: Delta 70.1 ms"},
              {"t": 1.6, "kind": "state", "text": "ARMED"},
              {"t": 9.0, "kind": "log", "text": "[WARN] vs: Delta 99.0 ms"}]
        out = collapse_repeats(ev)
        self.assertEqual(len(out), 3)
        self.assertIn("(x2 until t=1.5 s)", out[0]["text"])


if __name__ == "__main__":
    unittest.main()
