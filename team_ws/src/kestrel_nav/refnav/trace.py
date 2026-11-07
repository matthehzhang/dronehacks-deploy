"""Optional, write-only flight evidence. No ROS or simulator dependencies.

Each flushed JSONL line is independently readable after interruption; discard an
incomplete final line. Source timestamps retain their original clock domains.
Non-finite numbers are strings (NaN, +Infinity, -Infinity), never JSON numbers.
Recording failure disables the writer and warns once, without raising into flight.
The caller serializes a snapshot then enqueues without waiting for disk. A bounded
daemon queue drops new events when full; every subsequent event and the clean
shutdown summary report cumulative drops. Abrupt termination may lose queued rows.
"""
from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
import time


LPOS_FIELDS = (
    "timestamp", "timestamp_sample", "x", "y", "z", "vx", "vy", "vz",
    "ax", "ay", "az", "heading", "heading_good_for_control",
    "xy_valid", "z_valid", "v_xy_valid", "v_z_valid",
    "delta_xy", "xy_reset_counter", "delta_z", "z_reset_counter",
    "delta_vxy", "vxy_reset_counter", "delta_vz", "vz_reset_counter",
    "delta_heading", "heading_reset_counter", "eph", "epv", "evh", "evv",
    "dead_reckoning", "dist_bottom", "dist_bottom_valid",
)


def fields(message, names):
    """Retain fields available on the installed PX4 message version."""
    return {name: getattr(message, name) for name in names if hasattr(message, name)}


def _json_value(value):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if hasattr(value, "tolist"):  # numpy and ROS fixed-size numeric arrays
        return _json_value(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("+Infinity" if value > 0 else "-Infinity")
    return value


class TraceWriter:
    def __init__(self, path="", warn=None, queue_size=256):
        self._file = None
        self._warn = warn
        self._queue = Queue(maxsize=max(1, int(queue_size)))
        self._stop = Event()
        self._thread = None
        self._closed = False
        self.failed = False
        self.dropped = 0
        self.written = 0
        if path:
            try:
                # Never erase evidence if a run directory/name was reused.
                self._file = Path(path).open("x", encoding="utf-8")
                self._thread = Thread(target=self._write_loop, name="flight-trace", daemon=True)
                self._thread.start()
                self.record("schema", None, version=1, frame="PX4 local NED; body FRD",
                            queue_size=self._queue.maxsize,
                            nonfinite="NaN/+Infinity/-Infinity strings",
                            clocks="ros_s: node clock; wall_ns: Unix; monotonic_ns: process host",
                            source_timestamps="original message clock, not interchangeable with ros_s")
            except Exception as exc:
                self._fail(exc)
                if self._thread is None or self._thread.ident is None:
                    f, self._file = self._file, None
                    if f is not None:
                        try:
                            f.close()
                        except Exception:
                            pass

    @property
    def enabled(self):
        return self._file is not None and not self.failed and not self._closed

    def _fail(self, exc):
        if not self.failed:
            self.failed = True
            self._stop.set()
            try:
                if self._warn is not None:
                    self._warn(f"flight trace disabled: {type(exc).__name__}: {exc}")
            except Exception:
                pass  # logging failure must not become a flight failure either

    def record(self, event, ros_s, **data):
        if not self.enabled:
            return
        try:
            row = dict(event=event, ros_s=ros_s, wall_ns=time.time_ns(),
                       monotonic_ns=time.monotonic_ns(), dropped=self.dropped, **data)
            line = json.dumps(_json_value(row), allow_nan=False, separators=(",", ":")) + "\n"
            self._queue.put_nowait(line)
        except Full:
            self.dropped += 1
        except Exception as exc:
            self._fail(exc)

    def _write_loop(self):
        f = self._file
        try:
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    line = self._queue.get(timeout=0.05)
                except Empty:
                    continue
                try:
                    if not self.failed:
                        f.write(line)
                        f.flush()
                        self.written += 1
                finally:
                    self._queue.task_done()
            if not self.failed:
                f.write(json.dumps(dict(event="trace_summary", written=self.written,
                                        dropped=self.dropped)) + "\n")
                f.flush()
        except Exception as exc:
            self._fail(exc)
        finally:
            try:
                f.close()
            except Exception as exc:
                self._fail(exc)
            self._file = None

    def close(self, timeout_s=1.0):
        self._closed = True
        self._stop.set()
        if self._thread is not None and self._thread.ident is not None:
            try:
                self._thread.join(timeout=max(0.0, timeout_s))
            except Exception as exc:
                self._fail(exc)
