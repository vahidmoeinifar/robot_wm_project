"""
robot.py
========
Robot agent that wraps WorldModel, PathPlanner, and CommGraph.
Author: Vahid Moeinifar - AGH university of Krakow

"""

import numpy as np
from typing import List, Optional, Tuple
from modules.world_model import WorldModel
from modules.terrain_analysis import TerrainLayers
from modules.path_planner import astar_25d, astar_2d, path_stats, path_stats_flat
from modules.communication import RobotCommState


class Robot:
    """
    A single planetary rover agent.

    Parameters
    ----------
    robot_id    : e.g. "R1", "R2", "R3", or "L" (lander)
    start       : (row, col) starting pixel on the DEM
    target      : (row, col) investigation target pixel
    comm_range  : communication range in metres
    scan_radius : scanning radius in pixels per step
    """

    def __init__(self,
                 robot_id: str,
                 start: Tuple[int, int],
                 target: Tuple[int, int],
                 comm_range_m: float = 150.0,
                 scan_radius: int = 20,
                 color: str = "blue"):
        self.robot_id = robot_id
        self.start = start
        self.pos = list(start)              # current [row, col]
        self.target = target
        self.comm_range_m = comm_range_m
        self.scan_radius = scan_radius
        self.color = color

        # World Model
        self.wm = WorldModel(robot_id)

        # Path
        self.path_25d: Optional[List[Tuple[int, int]]] = None
        self.path_2d: Optional[List[Tuple[int, int]]] = None
        self.path_step: int = 0

        # Simulation state
        self.t: float = 0.0
        self.target_reached: bool = False
        self.history: List[dict] = []      # position history

        # Communication
        self.comm_state = RobotCommState(
            robot_id=robot_id,
            pos=(float(start[1]), float(start[0])),   # (x=col, y=row)
            comm_range_m=comm_range_m,
        )

    def attach_terrain(self, layers: TerrainLayers):
        """Attach 2.5D terrain to this robot's WM."""
        self.wm.attach_terrain(layers)

    def plan_paths(self, layers: TerrainLayers, pixel_scale: float = 1.0):
        """
        Plan both a 2.5D-aware path and a flat-2D path for comparison.
        """
        # 2.5D path (multi-criteria A*)
        self.path_25d = astar_25d(
            layers, tuple(self.start), tuple(self.target),
            w_time=0.5, w_damage=0.5
        )
        # Flat 2D baseline
        self.path_2d = astar_2d(
            layers.shape[0], layers.shape[1],
            tuple(self.start), tuple(self.target),
            pixel_scale=pixel_scale
        )

    def step(self, layers: TerrainLayers, t: float, steps_per_tick: int = 5):
        """
        Advance robot along its planned 2.5D path by `steps_per_tick` pixels.
        Scan the environment at each position.
        """
        self.t = t
        if self.target_reached or self.path_25d is None:
            return

        for _ in range(steps_per_tick):
            if self.path_step >= len(self.path_25d):
                self.target_reached = True
                break

            r, c = self.path_25d[self.path_step]
            self.pos = [r, c]
            self.path_step += 1

            # Update comm state position
            self.comm_state.pos = (float(c), float(r))

            # Scan surroundings
            sr = self.scan_radius
            r0 = max(0, r - sr)
            r1 = min(layers.shape[0], r + sr)
            c0 = max(0, c - sr)
            c1 = min(layers.shape[1], c + sr)
            self.wm.RTM.scan_region(range(r0, r1), range(c0, c1), t)

            # Run one DMC step
            self.wm.step(r, c, t)

        # Log state
        self.history.append({
            "t": t,
            "pos": tuple(self.pos),
            "H1": self.wm.RTM.H1_completeness,
            "H2": self.wm.RTM.H2_accuracy,
            "target_reached": self.target_reached,
        })

    def get_path_comparison(self, layers: TerrainLayers) -> dict:
        """Compare 2.5D path vs flat-2D path statistics."""
        stats_25d = (path_stats(self.path_25d, layers)
                     if self.path_25d else {})
        stats_2d = (path_stats_flat(self.path_2d,
                                    pixel_scale=layers.params.pixel_scale)
                    if self.path_2d else {})
        return {
            "robot": self.robot_id,
            "path_25d": stats_25d,
            "path_2d": stats_2d,
        }

    def __repr__(self):
        return (f"Robot({self.robot_id}, pos={tuple(self.pos)}, "
                f"target={self.target}, H1={self.wm.RTM.H1_completeness:.2f})")
