"""Staged cost-to-go fields for a general navigator (Track B2). PURE (numpy + scipy.sparse.csgraph).

The mission is an ordered list of STAGES. Each stage is a set of gate alternatives (segments that
must be crossed in a given direction: the maze exit line, the tube's mouth and far end, the window
or Gate B) and the last stage is the pad region. The cost of a cell is TIME: cell length divided by
a speed limited by the local clearance, plus a clearance penalty. Crossing a gate is a straight,
perpendicular hop from a cell just in front of it to the cell just behind it (so a narrow gate is
crossed square-on), which moves the aircraft to the next stage and earns that alternative's time
credit (checkpoint points converted to seconds).

For every stage the cost-to-go to the pad is computed BACKWARDS with scipy's compiled Dijkstra from
a virtual source: the pad cells (last stage), or the post cells of the stage's gates seeded with the
next stage's cost-to-go minus the credit. From any pose the navigator follows the field downhill
(`descend`), re-planning whenever the cost map changes (localizer sigma, live obstacles).

Not modelled here (documented for B3/B4): a wall-facing penalty needs a direction state per cell;
the planner's leg-end rules cover it for Track A.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

from .course import Grid2D
from .planner import gate_crossings, gate_segment, segment_free, with_walls

SQ2 = math.sqrt(2.0)


@dataclass
class Gate:
    """A segment p->q to be crossed in direction `normal` (unit, pointing to the post side)."""
    name: str
    p: tuple
    q: tuple
    normal: tuple
    credit_s: float = 0.0           # time credit for crossing this alternative (points * s per point)
    margin_m: float = 0.0           # keep this far from the segment's ends (the drone's radius + margin)


@dataclass
class Stage:
    name: str
    gates: list = field(default_factory=list)    # alternatives (an OR)
    pad_xy: tuple = None                         # the last stage: the pad region instead of gates
    pad_radius_m: float = 0.3


@dataclass
class FieldParams:
    v_max: float = 0.3
    clear_min_m: float = 0.40        # cells closer than this to a surface are blocked
    clear_pref_m: float = 0.60       # full speed from here outward
    v_min_frac: float = 0.35         # speed at clear_min_m as a fraction of v_max
    clear_penalty_s_per_m: float = 2.0   # extra seconds per metre of travel inside clear_pref_m, scaled by how far inside
    gate_depth_m: float = 0.45       # the perpendicular hop: this far in front of the gate to this far behind it
    gate_speed: float = 0.25


class NavField:
    def __init__(self, blocked: Grid2D, clearance: np.ndarray, stages, params: FieldParams = None):
        self.grid = blocked
        self.clear = np.asarray(clearance, float)
        self.stages = list(stages)
        self.gates = [g for stage in self.stages for g in stage.gates]
        # Ordinary graph edges never cross a gate, including gates from earlier
        # stages and unchosen alternatives. Only that gate's directed hop may.
        self.p = params or FieldParams()
        self.nx, self.ny = blocked.occ.shape
        self.n = self.nx * self.ny
        self.ctg = []                 # per stage: (nx, ny) cost-to-go seconds (inf = unreachable)
        self.gate_hops = []           # per stage: list of (pre linear idx array, post linear idx array, gate)
        self.build_s = 0.0
        self._adj = None
        self.rebuild()

    # ------------------------------------------------------------------ the grid graph
    def _cell_cost(self):
        """Seconds to traverse one metre in each cell (inf where blocked)."""
        p = self.p
        c = self.clear
        speed = p.v_max * np.clip(p.v_min_frac + (1 - p.v_min_frac) * (c - p.clear_min_m) / max(p.clear_pref_m - p.clear_min_m, 1e-6),
                                  p.v_min_frac, 1.0)
        inside = np.clip((p.clear_pref_m - c) / max(p.clear_pref_m - p.clear_min_m, 1e-6), 0.0, 1.0)
        cost = 1.0 / speed + p.clear_penalty_s_per_m * inside
        cost[self.travel_grid.occ | (c < p.clear_min_m)] = np.inf
        return cost

    def _adjacency(self):
        """8-connected grid, edge weight = step length * mean of the two cells' per-metre costs."""
        cost = self._cell_cost()
        res = self.grid.res
        ok = np.isfinite(cost)
        rows, cols, vals = [], [], []
        idx = np.arange(self.n).reshape(self.nx, self.ny)
        for dx, dy, step in ((1, 0, 1.0), (0, 1, 1.0), (1, 1, SQ2), (1, -1, SQ2)):
            a = idx[max(0, -dx):self.nx - max(0, dx), max(0, -dy):self.ny - max(0, dy)]
            b = idx[max(0, dx):self.nx - max(0, -dx), max(0, dy):self.ny - max(0, -dy)]
            ca, cb = cost.reshape(-1)[a.ravel()], cost.reshape(-1)[b.ravel()]
            good = np.isfinite(ca) & np.isfinite(cb)
            if dx and dy:                                     # no corner cutting through a blocked cell
                c1 = ok[max(0, -dx):self.nx - max(0, dx), max(0, dy):self.ny - max(0, -dy)].ravel()
                c2 = ok[max(0, dx):self.nx - max(0, -dx), max(0, -dy):self.ny - max(0, dy)].ravel()
                good &= c1 & c2
            w = step * res * 0.5 * (ca + cb)
            rows += [a.ravel()[good], b.ravel()[good]]
            cols += [b.ravel()[good], a.ravel()[good]]
            vals += [w[good], w[good]]
        rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
        return sparse.csr_matrix((vals, (rows, cols)), shape=(self.n, self.n)), cost

    def _gate_cells(self, g: Gate):
        """Cells just in front of (pre) and just behind (post) a gate, paired across it."""
        p, q = np.asarray(g.p, float), np.asarray(g.q, float)
        n = np.asarray(g.normal, float)
        n = n / np.linalg.norm(n)
        t = q - p
        L = float(np.linalg.norm(t))
        t = t / L
        res = self.grid.res
        pre, post = [], []
        physical = Grid2D(self.grid.occ | (self.clear < self.p.clear_min_m), self.grid.origin, res)
        hop_grid = with_walls(physical, [(h.p, h.q) for h in self.gates if h is not g])
        for s in np.arange(g.margin_m, L - g.margin_m + 1e-9, res * 0.5):
            c = p + t * s
            a = self.grid.ij(c - n * self.p.gate_depth_m)
            b = self.grid.ij(c + n * self.p.gate_depth_m)
            if self.grid.inside(a) and self.grid.inside(b):
                xy = self.grid.xy(np.array([a, b]))
                if not segment_free(hop_grid, *xy, step=res / 3):
                    continue
                if [d for _, d in gate_crossings(xy, (g.p, g.q), g.normal)] != [1]:
                    continue
                pre.append(a[0] * self.ny + a[1])
                post.append(b[0] * self.ny + b[1])
        if not pre:
            return np.zeros(0, int), np.zeros(0, int)
        pre, post = np.array(pre), np.array(post)
        uniq = np.unique(np.stack([pre, post], 1), axis=0)
        return uniq[:, 0], uniq[:, 1]

    # ------------------------------------------------------------------ the fields
    def rebuild(self):
        t0 = time.perf_counter()
        self.travel_grid = with_walls(self.grid, [(g.p, g.q) for g in self.gates])
        adj, cost = self._adjacency()
        self._adj = adj
        finite = np.isfinite(cost.reshape(-1))
        self.ctg = [None] * len(self.stages)
        self.gate_hops = [None] * len(self.stages)
        nxt = None
        for k in range(len(self.stages) - 1, -1, -1):
            st = self.stages[k]
            # the virtual source: pad cells (last stage) or seeded post cells of the stage's gates
            src_cells, src_w = [], []
            if st.pad_xy is not None:
                ij = np.argwhere(np.ones((self.nx, self.ny), bool))
                xy = self.grid.xy(ij)
                near = np.linalg.norm(xy - np.asarray(st.pad_xy), axis=1) <= st.pad_radius_m
                cells = ij[near][:, 0] * self.ny + ij[near][:, 1]
                cells = cells[finite[cells]]
                src_cells.append(cells)
                src_w.append(np.zeros(len(cells)))
                self.gate_hops[k] = []
            else:
                hops = []
                for g in st.gates:
                    pre, post = self._gate_cells(g)
                    keep = finite[pre] & finite[post] & np.isfinite(nxt.reshape(-1)[post])
                    pre, post = pre[keep], post[keep]
                    hops.append((pre, post, g))
                    hop = 2 * self.p.gate_depth_m / self.p.gate_speed
                    src_cells.append(pre)
                    src_w.append(nxt.reshape(-1)[post] + hop - g.credit_s)
                self.gate_hops[k] = hops
            if not src_cells or sum(len(c) for c in src_cells) == 0:
                self.ctg[k] = np.full((self.nx, self.ny), np.inf)
                nxt = self.ctg[k]
                continue
            cells = np.concatenate(src_cells)
            w = np.concatenate(src_w)
            # credits make seeds negative: shift them positive (sparse addition drops explicit zeros,
            # so strictly positive), then undo the shift
            c0 = float(max(0.0, -w.min())) + 1e-6
            w = w + c0
            v = self.n                                       # the virtual node
            extra = sparse.csr_matrix((w, (np.full(len(cells), v), cells)), shape=(self.n + 1, self.n + 1))
            G = sparse.vstack([sparse.hstack([adj, sparse.csr_matrix((self.n, 1))]),
                               sparse.csr_matrix((1, self.n + 1))]).tocsr() + extra
            d = dijkstra(G, directed=True, indices=v)
            field_k = (d[:self.n] - c0).reshape(self.nx, self.ny)
            field_k[~finite.reshape(self.nx, self.ny)] = np.inf
            self.ctg[k] = field_k
            nxt = field_k
        self.build_s = time.perf_counter() - t0

    # ------------------------------------------------------------------ following it
    def cost_to_go(self, xy, stage):
        ij = self.grid.ij(xy)
        if not self.grid.inside(ij):
            return math.inf
        return float(self.ctg[stage][ij[0], ij[1]])

    def descend(self, xy, stage, lookahead_m=2.5):
        """Follow the stage's field downhill from xy. -> (path (n,2) course xy, stage at the end,
        crossed gate names). The path ends at the look-ahead distance, the pad, or when the field
        runs out; a gate hop advances the stage."""
        ij = tuple(self.grid.ij(xy))
        if not self.grid.inside(np.array(ij)) or not math.isfinite(self.ctg[stage][ij]):
            # Moving to a reachable cell would require a separately checked path;
            # snapping can jump through the very gate that made us unreachable.
            return np.asarray([xy]), stage, []
        path = [np.asarray(xy, float)]
        crossed = []
        length = 0.0
        st = stage
        hop_cost = 2 * self.p.gate_depth_m / self.p.gate_speed

        def hops_of(stage):
            table = {}
            for pre, post, g in (self.gate_hops[stage] or []):
                for a, b in zip(pre, post):
                    table[int(a)] = (int(b), g)
            return table
        hop_lookup = hops_of(st)
        for _ in range(4000):
            lin = ij[0] * self.ny + ij[1]
            here = self.ctg[st][ij]
            if st == len(self.stages) - 1 and here <= 1e-6:        # on the pad
                break
            if lin in hop_lookup and st + 1 < len(self.stages):
                b, g = hop_lookup[lin]
                nb = (b // self.ny, b % self.ny)
                seed = self.ctg[st + 1][nb] + hop_cost - g.credit_s
                if here >= seed - 1e-6:                            # this cell's cost IS the hop: take it
                    path.append(self.grid.xy(np.array(nb)))
                    length += float(np.linalg.norm(path[-1] - path[-2]))
                    crossed.append(g.name)
                    st += 1
                    ij = nb
                    hop_lookup = hops_of(st)
                    continue
            best, best_c = None, here
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if not (dx or dy):
                        continue
                    nb = (ij[0] + dx, ij[1] + dy)
                    if 0 <= nb[0] < self.nx and 0 <= nb[1] < self.ny:
                        if self._adj[lin, nb[0] * self.ny + nb[1]] == 0:
                            continue                 # respect the graph's no-corner-cutting rule
                        c = self.ctg[st][nb]
                        if c < best_c - 1e-12:
                            best, best_c = nb, c
            if best is None:
                break
            ij = best
            path.append(self.grid.xy(np.array(ij)))
            length += float(np.linalg.norm(path[-1] - path[-2]))
            if length >= lookahead_m:
                break
        return np.asarray(path), st, crossed


def stages_from_course(doc, seconds_per_point=20.0, vehicle_margin=None):
    """The course file's ordered sequence -> stages of gate segments (names only in the file; the
    geometry comes from coursenav.semantics' derived goals)."""
    v = doc["vehicle"]
    margin = v["radius_m"] + v["margin_m"] if vehicle_margin is None else vehicle_margin
    stages = []
    d = doc["derived"]
    for g in doc["sequence"]:
        name = g["name"]
        if name == "maze_exit":
            end, dirn = np.asarray(d["maze_exit"]["line_end_xy"]), np.asarray(d["maze_exit"]["direction"])
            t = np.array([-dirn[1], dirn[0]])
            stages.append(Stage("maze_exit", [Gate("maze_exit", tuple(end - t * 0.6), tuple(end + t * 0.6), tuple(dirn))]))
        elif name == "tunnel_entry":                       # the resolved sequence expands the tube into entry/exit
            t_ = d["tunnel"]
            entry, exit_ = np.asarray(t_["entry"]), np.asarray(t_["exit"])
            axis = (exit_ - entry) / np.linalg.norm(exit_ - entry)
            side = np.array([-axis[1], axis[0]])
            mouth = entry + axis * (np.linalg.norm(exit_ - entry) - t_["length_m"]) / 2
            far = exit_ - axis * (np.linalg.norm(exit_ - entry) - t_["length_m"]) / 2
            stages.append(Stage("tunnel_in", [Gate("tunnel_mouth", tuple(mouth - side * 0.5), tuple(mouth + side * 0.5), tuple(axis))]))
            stages.append(Stage("tunnel_out", [Gate("tunnel_far", tuple(far - side * 0.5), tuple(far + side * 0.5), tuple(axis))]))
        elif "choose" in g:
            gates = []
            for key, opt in doc["chicane_options"].items():
                if "center" not in opt:
                    continue
                n = np.asarray(opt["normal"], float)
                if "cross_deg" in opt:
                    h = math.radians(opt["cross_deg"])
                    n = np.array([math.cos(h), math.sin(h)])
                n = n / np.linalg.norm(n)
                a, b = gate_segment(opt)
                gates.append(Gate(key, tuple(a), tuple(b), tuple(n),
                                  credit_s=float(opt.get("points", 0)) * seconds_per_point, margin_m=margin))
            stages.append(Stage("chicane", gates))
        elif g.get("land"):
            stages.append(Stage("pad", pad_xy=tuple(g["xy"]), pad_radius_m=0.3))
    return stages
