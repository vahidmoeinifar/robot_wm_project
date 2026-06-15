"""
run_simulation.py
=================
Main entry point for the full simulation with real NASA .IMG DEM files.

Author: Vahid Moeinifar - AGH university of Krakow
"""

import os
import glob
import numpy as np
import sys

# Make sure modules is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.dem_loader import DEMLoader, load_dem_or_synthetic
from modules.terrain_analysis import TerrainAnalyzer, terrain_stats
from modules.team_simulation import TeamSimulation
from modules.visualizer import generate_all_figures

DATA_DIR = "data"
IMG_PATTERN = os.path.join(DATA_DIR, "*.IMG")


def pick_best_dem(filepaths: list):
    """
    Load all DEMs and pick the one with the most interesting terrain
    (highest elevation range → more varied terrain for the simulation).
    """
    best = None
    best_range = -1
    best_path = None

    for path in filepaths:
        print(f"\n  Trying: {os.path.basename(path)}")
        try:
            loader = DEMLoader(path)
            dem = loader.load()
            print(dem.info())
            if dem.elevation_range > best_range and dem.valid_mask.sum() > 1000:
                best = dem
                best_range = dem.elevation_range
                best_path = path
        except Exception as e:
            print(f"    [WARN] Skipped: {e}")

    return best, best_path


def main():
    print("=" * 65)
    print("Coordinated World Model Learning — 2.5D Terrain Enhancement")
    print("AGH University of Science and Technology, Kraków")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Load DEM
    # ------------------------------------------------------------------
    img_files = sorted(glob.glob(IMG_PATTERN))
    print(f"\nFound {len(img_files)} .IMG files in '{DATA_DIR}/'")

    if img_files:
        dem, chosen_path = pick_best_dem(img_files)
        if dem is None:
            print("\nAll .IMG files failed to load. Falling back to synthetic DEM.")
            from modules.dem_loader import make_synthetic_dem
            dem = make_synthetic_dem(rows=400, cols=400, scale=1.0)
        else:
            print(f"\n✔ Using DEM: {os.path.basename(chosen_path)}")
    else:
        print(f"\nNo .IMG files found in '{DATA_DIR}/'. Using synthetic DEM.")
        from modules.dem_loader import make_synthetic_dem
        dem = make_synthetic_dem(rows=400, cols=400, scale=1.0)

    # Resize to manageable size for A* (≤512×512 recommended)
    if dem.rows > 512 or dem.cols > 512:
        print(f"\nDEM is large ({dem.rows}×{dem.cols}). Cropping to 400×400 centre.")
        r0 = (dem.rows - 400) // 2
        c0 = (dem.cols - 400) // 2
        import numpy as np
        dem.elevation = dem.elevation[r0:r0+400, c0:c0+400]
        dem.valid_mask = dem.valid_mask[r0:r0+400, c0:c0+400]
        dem.rows, dem.cols = 400, 400

    # ------------------------------------------------------------------
    # 2. Run team simulation
    # ------------------------------------------------------------------
    sim = TeamSimulation(
        dem=dem,
        n_rovers=3,
        comm_range_m=max(100.0, dem.cols * dem.scale * 0.35),
        scan_radius=max(15, dem.rows // 25),
        tmax_steps=100,
        share_interval=10,
    )

    results = sim.run()

    # ------------------------------------------------------------------
    # 3. Generate all figures
    # ------------------------------------------------------------------
    generate_all_figures(
        layers=sim.layers,
        robots=sim.robots,
        lander_start=sim.lander.start,
        results=results,
    )

    print("\n" + "=" * 65)
    print("Done! Check the 'results/' directory for all output figures.")
    print("=" * 65)


if __name__ == "__main__":
    main()
