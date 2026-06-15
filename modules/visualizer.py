"""
visualizer.py
=============
Author: Vahid Moeinifar - AGH university of Krakow

"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend for file output
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from typing import List, Optional, Dict
import networkx as nx

from modules.terrain_analysis import TerrainLayers
from modules.robot import Robot
from modules.communication import CommGraph

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Custom colormaps
TRAV_CMAP = LinearSegmentedColormap.from_list(
    "traversability", ["#C0392B", "#E67E22", "#F1C40F", "#2ECC71"], N=256)
ELEV_CMAP = "terrain"
SLOPE_CMAP = "hot_r"
VEL_CMAP   = "Blues"


def _save(fig, name: str):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 1 — DEM Overview (4-panel)
# ---------------------------------------------------------------------------

def plot_dem_overview(layers: TerrainLayers, title: str = "DEM Overview"):
    """Plot elevation, slope, roughness, and traversability side by side."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    panels = [
        (layers.elevation,      "Elevation (m)",         ELEV_CMAP),
        (layers.slope_deg,      "Slope (°)",             SLOPE_CMAP),
        (layers.roughness,      "Roughness (m)",         "YlOrRd"),
        (layers.traversability, "Traversability L₁ [0,1]", TRAV_CMAP),
    ]

    for ax, (data, label, cmap) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap, origin="upper", aspect="auto")
        ax.set_title(label, fontsize=11)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    _save(fig, "dem_overview.png")


# ---------------------------------------------------------------------------
# Figure 2 — Traversability map with label overlay
# ---------------------------------------------------------------------------

def plot_traversability_map(layers: TerrainLayers):
    """Large traversability map with label annotations."""
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(layers.traversability, cmap=TRAV_CMAP,
                   vmin=0, vmax=1, origin="upper", aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("Traversability L₁", fontsize=11)

    # Contour lines for traversability thresholds
    rows, cols = layers.shape
    y, x = np.mgrid[0:rows, 0:cols]
    thresholds = [0.2, 0.4, 0.6, 0.8]
    colors_t = ["#922B21", "#D35400", "#D4AC0D", "#1D8348"]
    for thr, col in zip(thresholds, colors_t):
        ax.contour(x, y, layers.traversability, levels=[thr],
                   colors=[col], linewidths=0.8, alpha=0.7)

    ax.set_title("Traversability Map\n"
                 r"$L_1(x,y)=1-[\alpha\cdot\theta/\theta_{max}+\beta\cdot R/R_{max}]$",
                 fontsize=12)
    ax.set_xlabel("Column (pixels)", fontsize=10)
    ax.set_ylabel("Row (pixels)", fontsize=10)

    # Legend patches for labels
    patches = [
        mpatches.Patch(color="#C0392B", label="Impassable (<0.2)"),
        mpatches.Patch(color="#E67E22", label="High risk (0.2–0.4)"),
        mpatches.Patch(color="#F1C40F", label="Moderate (0.4–0.6)"),
        mpatches.Patch(color="#2ECC71", label="Low risk (0.6–0.8)"),
        mpatches.Patch(color="#0B5345",  label="Free (>0.8)"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=9)

    plt.tight_layout()
    _save(fig, "traversability_map.png")


# ---------------------------------------------------------------------------
# Figure 3 — Velocity heatmap
# ---------------------------------------------------------------------------

def plot_velocity_heatmap(layers: TerrainLayers):
    """Physics-based velocity field."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Physics-Based Velocity Model (Slide 22)", fontsize=13, fontweight="bold")

    im0 = axes[0].imshow(layers.velocity, cmap=VEL_CMAP, origin="upper")
    axes[0].set_title("Velocity V(x,y)  [m/s]", fontsize=11)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # Scatter: slope vs velocity
    r_idx = np.random.choice(layers.shape[0], 2000, replace=True)
    c_idx = np.random.choice(layers.shape[1], 2000, replace=True)
    slopes = layers.slope_deg[r_idx, c_idx]
    vels   = layers.velocity[r_idx, c_idx]
    valid = np.isfinite(slopes) & np.isfinite(vels)

    sc = axes[1].scatter(slopes[valid], vels[valid],
                         c=layers.roughness[r_idx[valid], c_idx[valid]],
                         cmap="YlOrRd", alpha=0.4, s=8)
    plt.colorbar(sc, ax=axes[1], label="Roughness R [m]")
    axes[1].set_xlabel("Slope θ [°]", fontsize=10)
    axes[1].set_ylabel("Speed V [m/s]", fontsize=10)
    axes[1].set_title("V = V_max · f_slope · f_roughness", fontsize=11)
    axes[1].set_xlim(0, 45)
    axes[1].set_ylim(0, 2.1)

    # Theoretical curve for roughness=0
    th_slope = np.linspace(0, 45, 100)
    th_vel = 2.0 * np.maximum(0, 1 - th_slope / 45.0)
    axes[1].plot(th_slope, th_vel, "k-", lw=2, label="R=0 (theory)")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    _save(fig, "velocity_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 4 — Path Comparison: 2D vs 2.5D
# ---------------------------------------------------------------------------

def plot_path_comparison(layers: TerrainLayers, robots: List[Robot]):
    """Side-by-side comparison of 2D and 2.5D paths for all robots."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Path Planning: Flat 2D vs 2.5D Terrain-Aware (Multi-criteria A*)",
                 fontsize=13, fontweight="bold")

    for ax, (path_attr, title) in zip(
        axes,
        [("path_2d",  "Flat 2D Path\n(ignores elevation — baseline)"),
         ("path_25d", "2.5D Path\n(slope + roughness + damage cost)")]
    ):
        ax.imshow(layers.traversability, cmap=TRAV_CMAP,
                  vmin=0, vmax=1, origin="upper", alpha=0.85)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

        for robot in robots:
            path = getattr(robot, path_attr)
            if path is None or len(path) < 2:
                continue
            ys, xs = zip(*path)
            ax.plot(xs, ys, "-", color=robot.color, lw=1.8, alpha=0.9,
                    label=robot.robot_id)
            # Start marker
            ax.plot(path[0][1], path[0][0], "o",
                    color=robot.color, ms=10, markeredgecolor="white", mew=1.5)
            # Target marker
            ax.plot(path[-1][1], path[-1][0], "*",
                    color=robot.color, ms=14, markeredgecolor="white", mew=1.5)

        ax.legend(loc="upper right", fontsize=9,
                  framealpha=0.8)

    plt.tight_layout()
    _save(fig, "robot_paths_2d_vs_25d.png")


# ---------------------------------------------------------------------------
# Figure 5 — Communication graph snapshots
# ---------------------------------------------------------------------------

def plot_communication_graphs(comm_history: List[dict], robots: List[Robot],
                              lander_pos: tuple, n_snapshots: int = 4):
    """Plot evolution of communication graph (like Figure 4 & 6 in the paper)."""
    if not comm_history:
        return

    # Pick evenly spaced snapshots
    indices = np.linspace(0, len(comm_history) - 1, n_snapshots, dtype=int)

    fig, axes = plt.subplots(1, n_snapshots, figsize=(5 * n_snapshots, 5))
    if n_snapshots == 1:
        axes = [axes]
    fig.suptitle("Communication Graph Evolution (cf. Figure 6 in paper)",
                 fontsize=13, fontweight="bold")

    robot_colors_map = {r.robot_id: r.color for r in robots}
    robot_colors_map["L"] = "#F39C12"

    for ax, idx in zip(axes, indices):
        snap = comm_history[idx]
        t = snap["t"]
        nodes = snap["nodes"]
        edges = snap["edges"]

        G = nx.DiGraph()
        pos_dict = {}
        for rid, nd in nodes.items():
            G.add_node(rid)
            px, py = nd["pos"]
            pos_dict[rid] = (px, -py)   # Flip y for display

        for e in edges:
            G.add_edge(e["from"], e["to"], rate=e["rate_Mbps"])

        node_colors = [robot_colors_map.get(n, "grey") for n in G.nodes()]

        nx.draw_networkx_nodes(G, pos_dict, ax=ax,
                               node_color=node_colors,
                               node_size=800, alpha=0.9)
        nx.draw_networkx_labels(G, pos_dict, ax=ax, font_size=10,
                                font_color="white", font_weight="bold")
        nx.draw_networkx_edges(G, pos_dict, ax=ax,
                               edge_color="#2C3E50", arrows=True,
                               arrowsize=20, width=2.0, alpha=0.8,
                               connectionstyle="arc3,rad=0.1")

        if G.edges():
            edge_labels = {(e["from"], e["to"]): f"{e['rate_Mbps']:.1f}"
                           for e in edges}
            nx.draw_networkx_edge_labels(G, pos_dict, edge_labels=edge_labels,
                                         ax=ax, font_size=7)

        ax.set_title(f"t = {int(t)}s", fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    _save(fig, "communication_graph.png")


# ---------------------------------------------------------------------------
# Figure 6 — WM Quality Indicators over time
# ---------------------------------------------------------------------------

def plot_quality_indicators(time_log: List[float],
                            H_log: List[dict],
                            robots: List[Robot]):
    """Plot H1 and H2 quality indicators over time for all robots."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("World Model Quality Indicators Over Time\n"
                 "H₁: Surface Mapping Completeness · H₂: Subsurface Water Accuracy",
                 fontsize=13, fontweight="bold")

    for robot in robots:
        rid = robot.robot_id
        h1_vals = [h[rid]["H1_completeness"] if rid in h else 0 for h in H_log]
        h2_vals = [h[rid]["H2_accuracy"] if rid in h else 0 for h in H_log]

        ax1.plot(time_log, h1_vals, color=robot.color,
                 lw=2.0, label=rid, alpha=0.9)
        ax2.plot(time_log, h2_vals, color=robot.color,
                 lw=2.0, label=rid, alpha=0.9, linestyle="--")

    ax1.set_ylabel("H₁ Completeness", fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=0.8, color="grey", ls=":", lw=1.2, label="Target h₁=0.8")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("H₁: Surface Crack Mapping Completeness", fontsize=10)

    ax2.set_ylabel("H₂ Accuracy", fontsize=11)
    ax2.set_xlabel("Simulation Step (t)", fontsize=11)
    ax2.set_ylim(0, 1.05)
    ax2.axhline(y=0.7, color="grey", ls=":", lw=1.2, label="Target h₂=0.7")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("H₂: Subsurface Water Detection Accuracy", fontsize=10)

    plt.tight_layout()
    _save(fig, "scientific_yield.png")


# ---------------------------------------------------------------------------
# Figure 7 — Path statistics bar chart (2D vs 2.5D comparison)
# ---------------------------------------------------------------------------

def plot_path_stats_comparison(robots: List[Robot], layers: TerrainLayers):
    """Bar chart comparing path time and damage cost for 2D vs 2.5D."""
    valid_robots = [r for r in robots
                    if r.path_25d is not None and r.path_2d is not None]
    if not valid_robots:
        return

    n = len(valid_robots)
    x = np.arange(n)
    width = 0.35

    times_2d = []
    times_25d = []
    dmg_25d = []

    for robot in valid_robots:
        from modules.path_planner import path_stats, path_stats_flat
        s25 = path_stats(robot.path_25d, layers)
        s2d = path_stats_flat(robot.path_2d,
                              pixel_scale=layers.params.pixel_scale)
        times_2d.append(s2d.get("total_time_s", 0))
        times_25d.append(s25.get("total_time_s", 0))
        dmg_25d.append(s25.get("total_damage_cost", 0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Path Statistics: Flat 2D vs 2.5D Terrain-Aware",
                 fontsize=13, fontweight="bold")

    bars1 = ax1.bar(x - width/2, times_2d, width,
                    label="2D (flat)", color="#3498DB", alpha=0.8)
    bars2 = ax1.bar(x + width/2, times_25d, width,
                    label="2.5D (terrain)", color="#E74C3C", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([r.robot_id for r in valid_robots])
    ax1.set_ylabel("Travel Time [s]", fontsize=11)
    ax1.set_title("Travel Time Comparison", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.bar_label(bars1, fmt="%.0f", fontsize=8)
    ax1.bar_label(bars2, fmt="%.0f", fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    bars3 = ax2.bar(x, dmg_25d, width * 1.5,
                    color="#E67E22", alpha=0.85, label="2.5D damage cost")
    ax2.set_xticks(x)
    ax2.set_xticklabels([r.robot_id for r in valid_robots])
    ax2.set_ylabel("Damage Cost  (1−trav)·dist₃D·β", fontsize=11)
    ax2.set_title("Damage Cost (only in 2.5D mode)", fontsize=11)
    ax2.bar_label(bars3, fmt="%.1f", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    note = ("Note: 2D flat cost assumes constant V_max, ignores elevation.\n"
            "2.5D cost uses slope+roughness→velocity and 3D distance→damage.")
    fig.text(0.5, -0.04, note, ha="center", fontsize=9,
             style="italic", color="grey")

    plt.tight_layout()
    _save(fig, "path_stats_comparison.png")


# ---------------------------------------------------------------------------
# Figure 8 — Combined simulation overview
# ---------------------------------------------------------------------------

def plot_simulation_overview(layers: TerrainLayers, robots: List[Robot],
                             lander_start: tuple, results: dict):
    """Full simulation overview on traversability background."""
    fig = plt.figure(figsize=(14, 11))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Main map
    ax_main = fig.add_subplot(gs[:2, :2])
    ax_main.imshow(layers.traversability, cmap=TRAV_CMAP,
                   vmin=0, vmax=1, origin="upper", alpha=0.9)
    ax_main.set_title("Simulation Overview — 2.5D Paths & Knowledge Zones",
                      fontsize=11, fontweight="bold")
    ax_main.axis("off")

    for robot in robots:
        # Draw 2.5D path
        if robot.path_25d and len(robot.path_25d) > 1:
            ys, xs = zip(*robot.path_25d)
            ax_main.plot(xs, ys, "-", color=robot.color, lw=2.0,
                         alpha=0.85, label=f"{robot.robot_id} path")

        # Knowledge zone boundary
        if robot.wm.RTM.knowledge_zone:
            kz_r = [p[0] for p in robot.wm.RTM.knowledge_zone]
            kz_c = [p[1] for p in robot.wm.RTM.knowledge_zone]
            if kz_r:
                r_min, r_max = min(kz_r), max(kz_r)
                c_min, c_max = min(kz_c), max(kz_c)
                rect = plt.Rectangle((c_min, r_min),
                                     c_max - c_min, r_max - r_min,
                                     fill=False, edgecolor=robot.color,
                                     linestyle="--", lw=1.5, alpha=0.7)
                ax_main.add_patch(rect)

        # Markers
        ax_main.plot(robot.start[1], robot.start[0], "o",
                     color=robot.color, ms=10, mew=2, mec="white")
        ax_main.plot(robot.target[1], robot.target[0], "*",
                     color=robot.color, ms=14, mew=2, mec="white")

    # Lander
    ax_main.plot(lander_start[1], lander_start[0], "^",
                 color="#F39C12", ms=15, mew=2, mec="white", label="Lander")
    ax_main.legend(loc="upper right", fontsize=8, framealpha=0.8)

    # Slope panel
    ax_slope = fig.add_subplot(gs[0, 2])
    ax_slope.imshow(layers.slope_deg, cmap=SLOPE_CMAP, origin="upper")
    ax_slope.set_title("Slope θ [°]", fontsize=9)
    ax_slope.axis("off")

    # Roughness panel
    ax_rough = fig.add_subplot(gs[1, 2])
    ax_rough.imshow(layers.roughness, cmap="YlOrRd", origin="upper")
    ax_rough.set_title("Roughness R [m]", fontsize=9)
    ax_rough.axis("off")

    # H1 quality evolution
    ax_h1 = fig.add_subplot(gs[2, :2])
    time_log = results["time_log"]
    H_log = results["H_log"]
    for robot in robots:
        rid = robot.robot_id
        vals = [h[rid]["H1_completeness"] if rid in h else 0 for h in H_log]
        ax_h1.plot(time_log, vals, color=robot.color, lw=2.0, label=rid)
    ax_h1.set_ylabel("H₁ Completeness", fontsize=9)
    ax_h1.set_xlabel("Step t", fontsize=9)
    ax_h1.set_ylim(0, 1.05)
    ax_h1.axhline(y=0.8, color="grey", ls=":", lw=1)
    ax_h1.legend(fontsize=8)
    ax_h1.grid(alpha=0.3)
    ax_h1.set_title("Quality H₁ vs. Time", fontsize=9)

    # Velocity panel
    ax_vel = fig.add_subplot(gs[2, 2])
    ax_vel.imshow(layers.velocity, cmap=VEL_CMAP, origin="upper")
    ax_vel.set_title("Velocity V [m/s]", fontsize=9)
    ax_vel.axis("off")

    _save(fig, "simulation_overview.png")


# ---------------------------------------------------------------------------
# 2.5D surface plot (matplotlib 3D)
# ---------------------------------------------------------------------------

def plot_3d_surface(layers: TerrainLayers, downsample: int = 4):
    """3D surface coloured by traversability."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    step = max(1, downsample)
    elev = layers.elevation[::step, ::step]
    trav = layers.traversability[::step, ::step]
    rows, cols = elev.shape

    x = np.arange(cols) * layers.params.pixel_scale * step
    y = np.arange(rows) * layers.params.pixel_scale * step
    X, Y = np.meshgrid(x, y)

    # Map traversability to RGBA
    norm_trav = (trav - trav.min()) / (trav.max() - trav.min() + 1e-6)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, elev,
                           facecolors=plt.cm.RdYlGn(norm_trav),
                           linewidth=0, antialiased=False, alpha=0.9)
    ax.set_xlabel("X [m]", fontsize=9)
    ax.set_ylabel("Y [m]", fontsize=9)
    ax.set_zlabel("Elevation [m]", fontsize=9)
    ax.set_title("2.5D DEM Surface — Coloured by Traversability L₁",
                 fontsize=12, fontweight="bold")

    m = plt.cm.ScalarMappable(cmap=TRAV_CMAP)
    m.set_array(trav)
    plt.colorbar(m, ax=ax, fraction=0.02, pad=0.05,
                 label="Traversability L₁")

    plt.tight_layout()
    _save(fig, "surface_3d.png")


# ---------------------------------------------------------------------------
# Run all figures
# ---------------------------------------------------------------------------

def generate_all_figures(layers: TerrainLayers,
                         robots: List[Robot],
                         lander_start: tuple,
                         results: dict):
    print("\nGenerating figures...")
    plot_dem_overview(layers)
    plot_traversability_map(layers)
    plot_velocity_heatmap(layers)
    plot_path_comparison(layers, robots)
    plot_path_stats_comparison(robots, layers)
    plot_quality_indicators(results["time_log"], results["H_log"], robots)
    plot_communication_graphs(results["comm_history"], robots, lander_start)
    plot_simulation_overview(layers, robots, lander_start, results)
    plot_3d_surface(layers)
    print("All figures saved to results/")
