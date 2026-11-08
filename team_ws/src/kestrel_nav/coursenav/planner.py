"""A* grid search primitive used exclusively by depth horizons."""
import heapq
import math
import numpy as np
from .course import Grid2D
SQ2=math.sqrt(2.)
NBRS=[(1,0,1.),(-1,0,1.),(0,1,1.),(0,-1,1.),(1,1,SQ2),(1,-1,SQ2),(-1,1,SQ2),(-1,-1,SQ2)]
class PlanError(RuntimeError):
    pass

def astar(blocked: Grid2D, clearance, a_xy, b_xy, prefer_m=0.6, weight=4.0):
    """Shortest path on 8-connected cells, each step costing length * (1 + weight * crowding),
    crowding = how far inside prefer_m of an obstacle the cell is (0..1). -> (n,2) xy."""
    occ = blocked.occ
    nx, ny = occ.shape
    a, b = tuple(blocked.ij(a_xy)), tuple(blocked.ij(b_xy))
    for name, c in (("start", a), ("goal", b)):
        if not blocked.inside(np.array(c)) or occ[c]:
            raise PlanError(f"{name} {a_xy if name == 'start' else b_xy} is blocked or off the map")
    crowd = np.clip((prefer_m - clearance) / prefer_m, 0.0, 1.0)
    g = np.full(occ.shape, np.inf)
    parent = {}
    g[a] = 0.0
    h = lambda c: math.hypot(c[0] - b[0], c[1] - b[1])   # noqa: E731  admissible (cost >= length)
    pq = [(h(a), a)]
    done = np.zeros(occ.shape, bool)
    while pq:
        _, c = heapq.heappop(pq)
        if done[c]:
            continue
        if c == b:
            break
        done[c] = True
        for dx, dy, step in NBRS:
            n = (c[0] + dx, c[1] + dy)
            if not (0 <= n[0] < nx and 0 <= n[1] < ny) or occ[n] or done[n]:
                continue
            if dx and dy and (occ[c[0] + dx, c[1]] or occ[c[0], c[1] + dy]):
                continue                                   # no corner cutting
            cost = g[c] + step * (1.0 + weight * 0.5 * (crowd[c] + crowd[n]))
            if cost < g[n]:
                g[n] = cost
                parent[n] = c
                heapq.heappush(pq, (cost + h(n), n))
    if not np.isfinite(g[b]):
        raise PlanError(f"no path from {a_xy} to {b_xy}")
    cells = [b]
    while cells[-1] != a:
        cells.append(parent[cells[-1]])
    pts = blocked.xy(np.array(cells[::-1]))
    pts[0], pts[-1] = a_xy, b_xy
    return pts
