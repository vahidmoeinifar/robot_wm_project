"""
run_demo_synthetic.py
=====================
Runs the complete simulation using a SYNTHETIC fractal DEM.
No .IMG files required. Ideal for first-time testing.

Produces identical output to run_simulation.py but with generated terrain.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.dem_loader import make_synthetic_dem
from modules.team_simulation import TeamSimulation
from modules.visualizer import generate_all_figures

def main():
    print("=" * 65)
    print("Coordinated World Model Learning — 2.5D Terrain Enhancement")
    print("DEMO MODE: Synthetic fractal DEM (no .IMG files needed)")
    print("=" * 65)

    # Generate a realistic synthetic icy-moon surface
    dem = make_synthetic_dem(rows=350, cols=350, scale=1.0, seed=2026)
    print(f"\nSynthetic DEM: {dem.rows}×{dem.cols}, "
          f"elev range = [{dem.min_elevation:.1f}, {dem.max_elevation:.1f}] m")

    # Run team simulation
    sim = TeamSimulation(
        dem=dem,
        n_rovers=3,
        comm_range_m=120.0,
        scan_radius=18,
        tmax_steps=120,
        share_interval=10,
    )

    results = sim.run()

    # Generate all figures
    generate_all_figures(
        layers=sim.layers,
        robots=sim.robots,
        lander_start=sim.lander.start,
        results=results,
    )

    print("\n✔ Demo complete! See results/ for all figures.")


if __name__ == "__main__":
    main()
