"""
communication.py
================
Author: Vahid Moeinifar - AGH university of Krakow
"""

import math
import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class RobotCommState:
    """Communication parameters for one robot at one time step."""
    robot_id: str
    pos: Tuple[float, float]         # (x, y) in metres on DEM
    comm_range_m: float = 500.0      # Maximum reliable range [m]
    bandwidth_hz: float = 4e6        # Channel bandwidth [Hz] (Proximity-1: 4 Mbps)
    tx_power_dbm: float = 10.0       # Transmit power [dBm]
    mu: float = 0.0                  # Total data gathered [Mb]
    nu: float = 0.0                  # Mean transmission rate [Mbaud]
    beta_buf: float = 0.0            # Buffer waiting for transmission [Mb]
    delta: float = 0.0               # Data received from others [Mb]


def shannon_rate(bandwidth: float, snr_linear: float) -> float:
    """ρ = b · log₂(1 + γ)  [bits/s]"""
    return bandwidth * math.log2(1.0 + max(snr_linear, 0.0))


def estimate_snr(tx_power_dbm: float, dist_m: float,
                 path_loss_exp: float = 2.5,
                 noise_floor_dbm: float = -100.0) -> float:
    """
    Simple log-distance path loss model.
    Returns SNR as a linear ratio.
    """
    if dist_m < 1.0:
        dist_m = 1.0
    path_loss_db = 20 * math.log10(dist_m) * path_loss_exp / 2.0
    rx_power_dbm = tx_power_dbm - path_loss_db
    snr_db = rx_power_dbm - noise_floor_dbm
    return 10 ** (snr_db / 10.0)


class CommGraph:
    """
    Dynamic directed communication graph for a robot team.
    Uses NetworkX DiGraph under the hood.
    """

    def __init__(self, min_rate_bps: float = 1e5):
        """
        Parameters
        ----------
        min_rate_bps : float
            Minimum acceptable error-free transmission rate ρ [bits/s].
            Edges with rate < ρ are pruned from the active graph.
        """
        self.min_rate = min_rate_bps
        self.G: nx.DiGraph = nx.DiGraph()
        self.states: Dict[str, RobotCommState] = {}
        self.history: List[dict] = []

    def add_robot(self, state: RobotCommState):
        """Register a robot in the communication graph."""
        self.states[state.robot_id] = state
        self.G.add_node(state.robot_id, pos=state.pos,
                        comm_range=state.comm_range_m)

    def update_position(self, robot_id: str, pos: Tuple[float, float]):
        """Update robot position and recompute communication graph."""
        if robot_id in self.states:
            self.states[robot_id].pos = pos
            self.G.nodes[robot_id]["pos"] = pos
        self.recompute()

    def recompute(self):
        """
        Recompute all edges based on current positions.
        Adds edge (Ri → Rj) if Shannon rate ≥ min_rate and
        dist(Ri, Rj) ≤ comm_range of Ri.
        """
        self.G.remove_edges_from(list(self.G.edges()))

        ids = list(self.states.keys())
        for i, rid_i in enumerate(ids):
            si = self.states[rid_i]
            for rid_j in ids:
                if rid_i == rid_j:
                    continue
                sj = self.states[rid_j]
                dist = self._dist(si.pos, sj.pos)
                if dist > si.comm_range_m:
                    continue
                snr = estimate_snr(si.tx_power_dbm, dist)
                rate = shannon_rate(si.bandwidth_hz, snr)
                if rate >= self.min_rate:
                    self.G.add_edge(rid_i, rid_j,
                                    rate_bps=rate,
                                    dist_m=dist,
                                    snr_linear=snr)

    def reachable_from(self, robot_id: str) -> List[str]:
        """Return all robots reachable from robot_id (multi-hop, Definition 5)."""
        if robot_id not in self.G:
            return []
        return list(nx.descendants(self.G, robot_id))

    def direct_links(self, robot_id: str) -> List[Tuple[str, float]]:
        """Return direct (1-hop) neighbours and their transmission rates."""
        return [(v, self.G[robot_id][v]["rate_bps"])
                for v in self.G.successors(robot_id)]

    def can_communicate(self, ri: str, rj: str) -> bool:
        """True if Ri can send data to Rj (directly or via multi-hop)."""
        return nx.has_path(self.G, ri, rj)

    def log_state(self, t: float):
        """Record graph state for later plotting (Figure 6 equivalent)."""
        edge_data = []
        for u, v, d in self.G.edges(data=True):
            edge_data.append({
                "from": u, "to": v,
                "rate_Mbps": d["rate_bps"] / 1e6,
                "dist_m": d["dist_m"],
            })
        node_data = {
            rid: {
                "pos": s.pos,
                "mu_Mb": s.mu,
                "nu_Mbaud": s.nu,
                "beta_Mb": s.beta_buf,
                "delta_Mb": s.delta,
            }
            for rid, s in self.states.items()
        }
        self.history.append({"t": t, "edges": edge_data, "nodes": node_data})

    def summary(self, t: float = None) -> str:
        lines = [f"CommGraph (t={t})"]
        for u, v, d in self.G.edges(data=True):
            lines.append(f"  {u}→{v}  rate={d['rate_bps']/1e6:.2f} Mbps  "
                         f"dist={d['dist_m']:.1f} m")
        if not self.G.edges():
            lines.append("  (no active links)")
        return "\n".join(lines)

    @staticmethod
    def _dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# ---------------------------------------------------------------------------
# Sharing protocol
# ---------------------------------------------------------------------------

class WMSharingProtocol:
    def __init__(self, comm_graph: CommGraph):
        self.comm = comm_graph
        self.share_log: List[dict] = []

    def spontaneous_share(self, sender_id: str,
                          shared_wm: dict, t: float) -> List[str]:
        receivers = []
        for rid in self.comm.direct_links(sender_id):
            rid_id = rid[0]
            self._send(sender_id, rid_id, shared_wm, "spontaneous", t)
            receivers.append(rid_id)
        return receivers

    def peer_request(self, requester_id: str,
                     target_id: str, t: float) -> bool:
        """
        Peer-requested mode: requester asks target for its WM data.
        Returns True if communication is possible.
        """
        if self.comm.can_communicate(requester_id, target_id):
            self._log_request(requester_id, target_id, "peer_request", t)
            return True
        return False

    def coordinator_command(self, coordinator_id: str,
                            shared_wm: dict, t: float) -> List[str]:
        """
        Coordinator-commanded mode: lead robot broadcasts to all reachable.
        """
        reachable = self.comm.reachable_from(coordinator_id)
        for rid in reachable:
            self._send(coordinator_id, rid, shared_wm, "coordinator_commanded", t)
        return reachable

    def _send(self, sender: str, receiver: str, wm: dict,
              mode: str, t: float):
        # Estimate transfer time based on WM data size and link rate
        wm_size_bits = self._estimate_wm_size(wm)
        links = dict(self.comm.direct_links(sender))
        if receiver in links:
            rate = links[receiver]
        else:
            # Multi-hop: use minimum rate along path
            rate = 1e5  # Fallback
        transfer_time_s = wm_size_bits / max(rate, 1.0)

        self.share_log.append({
            "t": t,
            "sender": sender,
            "receiver": receiver,
            "mode": mode,
            "wm_size_kb": wm_size_bits / 8000,
            "transfer_time_s": transfer_time_s,
        })

    def _log_request(self, requester: str, target: str,
                     mode: str, t: float):
        self.share_log.append({
            "t": t, "requester": requester, "target": target, "mode": mode
        })

    @staticmethod
    def _estimate_wm_size(wm: dict) -> int:
        """Rough estimate: each feature/object ~1 kB, images ~100 kB."""
        n_features = len(wm.get("rtm", {}).get("features", []))
        n_objects = len(wm.get("rtm", {}).get("objects", []))
        return (n_features + n_objects) * 8000 + 100 * 8000  # bits
