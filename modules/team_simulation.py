"""
team_simulation.py
==================
Author: Vahid Moeinifar - AGH university of Krakow

"""

import numpy as np
from typing import List, Optional, Dict
from modules.dem_loader import DEMData
from modules.terrain_analysis import TerrainAnalyzer, TerrainLayers, terrain_stats
from modules.robot import Robot
from modules.world_model import WorldModel
from modules.communication import CommGraph, WMSharingProtocol, RobotCommState


class TeamSimulation:
    """
    Full multi-robot WM building simulation.

    Robots: R1, R2, R3 (rovers) + L (lander, stationary coordinator)
    """

    # COLOURS for plotting
    ROBOT_COLORS = {
        "R1": "#E74C3C",    # Red
        "R2": "#27AE60",    # Green
        "R3": "#3498DB",    # Blue
        "L":  "#F39C12",    # Orange (lander)
    }

    def __init__(self, dem: DEMData, n_rovers: int = 3,
                 comm_range_m: float = 150.0,
                 scan_radius: int = 18,
                 tmax_steps: int = 80,
                 share_interval: int = 10,
                 pixel_scale: float = None):

        self.dem = dem
        self.n_rovers = n_rovers
        self.tmax = tmax_steps
        self.share_interval = share_interval

        # Compute terrain layers
        params_override = None
        if pixel_scale:
            from modules.terrain_analysis import TerrainParams
            params_override = TerrainParams(pixel_scale=pixel_scale)
        analyzer = TerrainAnalyzer(params_override)
        self.layers: TerrainLayers = analyzer.analyze(dem)

        rows, cols = self.layers.shape

        # ----- Robot placement -----
        # Divide DEM into quadrants; start robots near safe areas
        def safe_start(r_frac, c_frac, jitter=20):
            r = int(r_frac * rows) + np.random.randint(-jitter, jitter)
            c = int(c_frac * cols) + np.random.randint(-jitter, jitter)
            r = np.clip(r, 10, rows - 10)
            c = np.clip(c, 10, cols - 10)
            return (int(r), int(c))

        starts = [
            safe_start(0.15, 0.15),
            safe_start(0.15, 0.80),
            safe_start(0.80, 0.50),
        ]

        # Investigation targets (sites of scientific interest Oj)
        # Place in high-roughness zones (interesting terrain)
        rough = self.layers.roughness
        flat_idx = np.argsort(rough.flatten())[::-1]  # Sort by roughness desc
        target_positions = []
        for idx in flat_idx[:20]:
            r, c = divmod(int(idx), cols)
            if 20 < r < rows - 20 and 20 < c < cols - 20:
                # Ensure targets are spread out
                if all(abs(r - tr) + abs(c - tc) > 80
                       for tr, tc in target_positions):
                    target_positions.append((r, c))
                if len(target_positions) >= n_rovers:
                    break

        # Fallback targets
        while len(target_positions) < n_rovers:
            target_positions.append(safe_start(0.5, 0.5))

        # Lander position (centre-ish, stationary)
        lander_pos = safe_start(0.45, 0.45, jitter=10)

        # Build robots
        self.robots: List[Robot] = []
        for i in range(n_rovers):
            rid = f"R{i+1}"
            robot = Robot(
                robot_id=rid,
                start=starts[i],
                target=target_positions[i],
                comm_range_m=comm_range_m,
                scan_radius=scan_radius,
                color=self.ROBOT_COLORS[rid],
            )
            robot.attach_terrain(self.layers)
            robot.plan_paths(self.layers, self.layers.params.pixel_scale)
            self.robots.append(robot)

        # Lander (stationary)
        self.lander = Robot(
            robot_id="L",
            start=lander_pos,
            target=lander_pos,
            comm_range_m=comm_range_m * 1.5,
            scan_radius=scan_radius * 2,
            color=self.ROBOT_COLORS["L"],
        )
        self.lander.attach_terrain(self.layers)
        self.lander.target_reached = True  # Lander doesn't move

        # Communication graph
        self.comm = CommGraph(min_rate_bps=1e5)
        for robot in self.robots + [self.lander]:
            self.comm.add_robot(robot.comm_state)
        self.comm.recompute()
        self.protocol = WMSharingProtocol(self.comm)

        # History for plotting
        self.time_log: List[float] = []
        self.H_log: List[dict] = []          # Quality indicator history
        self.share_log_ext: List[dict] = []  # Extended sharing log
        self.comm_history: List[dict] = []   # Comm graph snapshots

        print(f"TeamSimulation ready:")
        print(f"  DEM shape : {rows}×{cols}")
        print(f"  Terrain   : {terrain_stats(self.layers)['elevation_m']}")
        for r in self.robots:
            print(f"  {r.robot_id}: start={r.start} → target={r.target}")
        print(f"  Lander L  : pos={self.lander.start}")

    # ------------------------------------------------------------------
    # Run (Algorithm 2)
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Execute TeamSimulation(RT) — Algorithm 2 from the paper.
        Returns final results dict.
        """
        print(f"\nRunning simulation (tmax={self.tmax} steps)...")
        t = 0.0

        for step in range(self.tmax):
            t = float(step)

            # --- Algorithm 1: simulate each robot ---
            for robot in self.robots:
                robot.step(self.layers, t, steps_per_tick=3)
                # Update comm graph with new position
                self.comm.update_position(robot.robot_id, robot.comm_state.pos)

            # --- WM Sharing (coordinator-commanded every share_interval) ---
            if step % self.share_interval == 0:
                self._wm_sharing_round(t)

            # --- Log state ---
            self.time_log.append(t)
            H_snapshot = {}
            for robot in self.robots:
                H_snapshot[robot.robot_id] = robot.wm.quality_indicators()
            self.H_log.append(H_snapshot)

            # Log communication graph
            self.comm.log_state(t)

            # --- Check termination ---
            all_done = all(r.target_reached for r in self.robots)
            if all_done:
                print(f"  All robots reached targets at step {step}!")
                break

        print(f"Simulation complete. Final step: {step}")
        return self._collect_results(t)

    # ------------------------------------------------------------------
    # WM sharing round (⊕ merge operator)
    # ------------------------------------------------------------------

    def _wm_sharing_round(self, t: float):
        """
        One round of WM sharing: each robot shares with reachable peers.
        Implements the ⊕ merge operator (Eq. 5 of the paper).
        """
        for robot in self.robots:
            shared = robot.wm.get_shareable_wm()
            receivers = self.protocol.spontaneous_share(
                robot.robot_id, shared, t)

            # Find receiver robot objects and merge
            for recv_id in receivers:
                recv_robot = self._find_robot(recv_id)
                if recv_robot:
                    recv_robot.wm.merge_wm(shared)
                    # Update comm data volume stats
                    robot.comm_state.mu += 0.1  # ~100 kB per share
                    robot.comm_state.nu = robot.comm_state.mu / max(t, 1)
                    recv_robot.comm_state.delta += 0.1

    def _find_robot(self, robot_id: str) -> Optional[Robot]:
        if robot_id == "L":
            return self.lander
        return next((r for r in self.robots if r.robot_id == robot_id), None)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def _collect_results(self, t_final: float) -> dict:
        results = {
            "t_final": t_final,
            "robots": {},
            "team": {},
            "comm_history": self.comm.history,
            "H_log": self.H_log,
            "time_log": self.time_log,
        }

        # Per-robot path comparison
        for robot in self.robots:
            results["robots"][robot.robot_id] = {
                "path_comparison": robot.get_path_comparison(self.layers),
                "final_H1": robot.wm.RTM.H1_completeness,
                "final_H2": robot.wm.RTM.H2_accuracy,
                "target_reached": robot.target_reached,
                "history": robot.history,
                "mm_version": robot.wm.MM.version,
                "path_25d": robot.path_25d,
                "path_2d": robot.path_2d,
                "pos_history": [(h["pos"]) for h in robot.history],
            }

        # Team aggregate
        all_H1 = [r.wm.RTM.H1_completeness for r in self.robots]
        all_H2 = [r.wm.RTM.H2_accuracy for r in self.robots]
        results["team"] = {
            "mean_H1": float(np.mean(all_H1)),
            "mean_H2": float(np.mean(all_H2)),
            "n_shares": len(self.protocol.share_log),
            "share_log": self.protocol.share_log,
        }

        # Print summary table
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        for rid, rdata in results["robots"].items():
            pc = rdata["path_comparison"]
            p25 = pc["path_25d"]
            p2d = pc["path_2d"]
            print(f"\n{rid}:")
            print(f"  H1 (coverage)       : {rdata['final_H1']:.3f}")
            print(f"  H2 (water accuracy) : {rdata['final_H2']:.3f}")
            print(f"  2.5D path waypoints : {p25.get('n_waypoints', 'N/A')}")
            print(f"  2.5D path time [s]  : {p25.get('total_time_s', 'N/A')}")
            print(f"  2.5D damage cost    : {p25.get('total_damage_cost', 'N/A')}")
            print(f"  2D  path time [s]   : {p2d.get('total_time_s', 'N/A')}")
            print(f"  2D  damage cost     : {p2d.get('total_damage_cost', 0)} (not modelled)")
            print(f"  MM updates (g)      : {rdata['mm_version']}")

        print(f"\nTeam H1 mean : {results['team']['mean_H1']:.3f}")
        print(f"Team H2 mean : {results['team']['mean_H2']:.3f}")
        print(f"WM shares    : {results['team']['n_shares']}")
        print("="*60)

        return results
