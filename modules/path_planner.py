"""
path_planner.py
===============

Author: Vahid Moeinifar - AGH university of Krakow

"""

import heapq
import numpy as np
from typing import Dict, List, Optional, Tuple
from modules.terrain_analysis import (
    TerrainLayers,
    edge_cost_combined,
    flat_2d_cost,
)

# Pixel adjacency: 8-connected grid
_NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1),
              (-1, -1), (-1, 1), (1, -1), (1, 1)]


def _heuristic(r0: int, c0: int, r1: int, c1: int, scale: float) -> float:
    """Euclidean distance heuristic (admissible for time/dist costs)."""
    return scale * np.sqrt((r1 - r0) ** 2 + (c1 - c0) ** 2)


def astar_25d(
    layers: TerrainLayers,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    w_time: float = 0.5,
    w_damage: float = 0.5,
) -> Optional[List[Tuple[int, int]]]:
    """
    A* on 2.5D terrain.  Returns list of (row, col) waypoints, or None.

    Cost per edge = w_time · time_cost + w_damage · damage_cost
    """
    rows, cols = layers.shape
    scale = layers.params.pixel_scale

    def is_valid(r, c):
        return (0 <= r < rows and 0 <= c < cols and
                np.isfinite(layers.traversability[r, c]))

    open_heap: List[Tuple[float, Tuple[int, int]]] = []
    heapq.heappush(open_heap, (0.0, start))
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    g_score: Dict[Tuple[int, int], float] = {start: 0.0}

    sr, sc = start
    gr, gc = goal

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            return _reconstruct_path(came_from, goal)

        cr, cc = current
        for dr, dc in _NEIGHBORS:
            nr, nc = cr + dr, cc + dc
            if not is_valid(nr, nc):
                continue

            tentative_g = g_score[current] + edge_cost_combined(

                layers, cr, cc, nr, nc, w_time, w_damage
            )

            if (nr, nc) not in g_score or tentative_g < g_score[(nr, nc)]:
                g_score[(nr, nc)] = tentative_g
                came_from[(nr, nc)] = current
                f = tentative_g + _heuristic(nr, nc, gr, gc, scale)
                heapq.heappush(open_heap, (f, (nr, nc)))

    return None  # No path found


def astar_2d(
    rows: int,
    cols: int,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    pixel_scale: float = 1.0,
    V_max: float = 2.0,
) -> Optional[List[Tuple[int, int]]]:
    """
    Baseline A* ignoring terrain (flat 2D).
    Used to compare path cost with the 2.5D planner.
    """
    def is_valid(r, c):
        return 0 <= r < rows and 0 <= c < cols

    open_heap = [(0.0, start)]
    came_from = {start: None}
    g_score = {start: 0.0}
    gr, gc = goal

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            return _reconstruct_path(came_from, goal)
        cr, cc = current
        for dr, dc in _NEIGHBORS:
            nr, nc = cr + dr, cc + dc
            if not is_valid(nr, nc):
                continue
            tentative_g = g_score[current] + flat_2d_cost(
                cr, cc, nr, nc, pixel_scale, V_max)
            if (nr, nc) not in g_score or tentative_g < g_score[(nr, nc)]:
                g_score[(nr, nc)] = tentative_g
                came_from[(nr, nc)] = current
                f = tentative_g + _heuristic(nr, nc, gr, gc, pixel_scale)
                heapq.heappush(open_heap, (f, (nr, nc)))

    return None


def _reconstruct_path(
    came_from: Dict, goal: Tuple[int, int]
) -> List[Tuple[int, int]]:
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Path statistics
# ---------------------------------------------------------------------------

def path_stats(path: List[Tuple[int, int]],
               layers: TerrainLayers) -> dict:
    """Compute aggregate statistics for a path on 2.5D terrain."""
    if not path or len(path) < 2:
        return {}

    p = layers.params
    total_time = 0.0
    total_damage = 0.0
    total_dist_2d = 0.0
    total_dist_3d = 0.0
    trav_vals = []

    for i in range(len(path) - 1):
        r0, c0 = path[i]
        r1, c1 = path[i + 1]

        dr = (r1 - r0) * p.pixel_scale
        dc = (c1 - c0) * p.pixel_scale
        dz = float(layers.elevation[r1, c1]) - float(layers.elevation[r0, c0])

        d2d = np.sqrt(dr ** 2 + dc ** 2)
        d3d = np.sqrt(dr ** 2 + dc ** 2 + dz ** 2)
        total_dist_2d += d2d
        total_dist_3d += d3d

        v_avg = 0.5 * (float(layers.velocity[r0, c0]) + float(layers.velocity[r1, c1]))
        v_avg = max(v_avg, 0.01)
        total_time += d2d / v_avg

        t_avg = 0.5 * (float(layers.traversability[r0, c0]) +
                       float(layers.traversability[r1, c1]))
        trav_vals.append(t_avg)
        total_damage += (1 - t_avg) * d3d * p.beta_safety

    return {
        "n_waypoints": len(path),
        "total_dist_2d_m": round(total_dist_2d, 2),
        "total_dist_3d_m": round(total_dist_3d, 2),
        "total_time_s": round(total_time, 2),
        "total_damage_cost": round(total_damage, 3),
        "mean_traversability": round(float(np.mean(trav_vals)), 3),
        "min_traversability": round(float(np.min(trav_vals)), 3),
    }


def path_stats_flat(path: List[Tuple[int, int]],
                    pixel_scale: float = 1.0,
                    V_max: float = 2.0) -> dict:
    """Statistics for a flat-2D path (no terrain data)."""
    if not path or len(path) < 2:
        return {}

    total = 0.0
    for i in range(len(path) - 1):
        r0, c0 = path[i]
        r1, c1 = path[i + 1]
        total += flat_2d_cost(r0, c0, r1, c1, pixel_scale, V_max)

    dist = sum(
        pixel_scale * np.sqrt((path[i+1][0]-path[i][0])**2 +
                               (path[i+1][1]-path[i][1])**2)
        for i in range(len(path) - 1)
    )

    return {
        "n_waypoints": len(path),
        "total_dist_2d_m": round(dist, 2),
        "total_time_s": round(dist / V_max, 2),
        "total_damage_cost": 0.0,   # Not modelled in 2D
        "mean_traversability": 1.0,  # Assumed flat
    }
