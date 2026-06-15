# Coordinated World Model Learning for Robot Teams — 2.5D Terrain Enhancement

**AGH University of Science and Technology, Kraków**  
ML Course Project · Based on: *Skulimowski, A.M.J., "Coordinated World Model Learning for Deep Space Robot Teams", IEEE AeroConf 2026*

By: Vahid Moeinifar (vmoeinifar@agh.edu.pl)
---

## Overview

This project implements a **multi-robot World Model (WM) learning framework** enhanced with **NASA 2.5D Digital Elevation Model (DEM) data** (`.IMG` format). It demonstrates how 3D terrain geometry improves every stage of the Decision Making Cycle (DMC) compared to flat 2D imagery.

### Key Contributions
- Formal WM = `(MM(R), RTM(R,E,t), X, g)` implemented in Python
- DEM-driven traversability: `L₁(x,y) = 1 − [α·(θ/θ_max) + β·(R/R_max)]`
- Physics-based velocity: `V = V_max · f_slope · f_roughness`
- 3D-aware damage cost: `cost = (1−avg_trav) · dist_3D · β`
- Multi-robot WM sharing with communication graphs (NetworkX)
- Coordinated WM Building (CWMB / P4) simulation with 3 rovers + lander
- Multi-criteria A\* path planning on 2.5D terrain

---

## Project Structure

```
robot_wm_project/
│
├── README.md
├── requirements.txt
│
├── data/                        
│   └── (1 × NASA HiRISE .IMG files)  
│
├── modules/
│   ├── dem_loader.py            # Load & parse NASA .IMG (PDS) DEMs
│   ├── terrain_analysis.py      # Slope, roughness, traversability, velocity
│   ├── world_model.py           # WM = (MM, RTM, X, g) classes
│   ├── robot.py                 # Robot agent with WM, sensors, navigation
│   ├── path_planner.py          # Multi-criteria A* on 2.5D terrain
│   ├── communication.py         # Communication graphs (NetworkX)
│   └── team_simulation.py       # Algorithm 1 & 2 from the paper
│
├── run_simulation.py            # Main entry point — run everything
├── run_demo_synthetic.py        # Demo with synthetic DEM (no .IMG needed)
└── results/                     # Output figures, logs, CSVs saved here
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your DEM files
Copy your `.IMG` files into the `data/` folder. One sample exist. You can download more DEM images from: https://www.uahirise.org/dtm/index.php

### 3. Run with real DEM data
```bash
python run_simulation.py
```

### 4. Run with synthetic terrain (no DEM files needed)
```bash
python run_demo_synthetic.py
```

---

## Mathematical Background

### World Model Definition (Eq. 1 from paper)
```
WM(R, E, t) = (MM(R), RTM(R,E,t), X, g)
```

| Component | Role |
|-----------|------|
| `MM(R)` | Meta-model: object libraries, ontologies, rule base |
| `RTM(R,E,t)` | Real-time model: maps, images, features, measurements |
| `X` | Decision Engine: classifies, infers rules, plans actions |
| `g` | Generalization: updates MM when observations don't fit |

### 2.5D Terrain Pipeline
```
DEM Z(x,y)  →  slope θ  →  roughness R  →  traversability L₁  →  velocity V  →  damage cost
```

**Traversability** (continuous [0,1]):
```
L₁(x,y) = 1 − [α·(θ/θ_max) + β·(R/R_max)]
α = 0.6, β = 0.4, θ_max = 45°, R_max = 0.5m
```

**Physics-based velocity**:
```
V = V_max · max(0, 1 − θ/45°) · max(0, 1 − R/0.5)
V_max = 2.0 m/s
```

**3D-aware damage cost** (per graph edge):
```
damage = (1 − avg_traversability) · dist_3D · β_safety
dist_3D = √(Δx² + Δy² + Δz²)
```

### Shared WM (Eq. 5 from paper)
```
WM(RT, E, t) = ⊕_{1≤i≤n} WMs(Ri, E, t)
```

---

## Output Figures

| Figure | Description |
|--------|-------------|
| `dem_overview.png` | Raw DEM elevation, slope, roughness, traversability |
| `traversability_map.png` | Per-pixel L₁ traversability across full DEM |
| `robot_paths_2d_vs_25d.png` | Path comparison: flat 2D cost vs 2.5D-aware cost |
| `communication_graph.png` | Robot team communication graph at each time step |
| `wm_sharing_progress.png` | Knowledge zone coverage over time |
| `scientific_yield.png` | H₁, H₂ quality indicators vs. time |
| `velocity_heatmap.png` | Physics-based velocity field across DEM |

---

## References

1. Skulimowski, A.M.J. (2026). *Coordinated World Model Learning for Deep Space Robot Teams*. IEEE AeroConf 2026. doi:10.1109/...
2. NASA HiRISE DEM data: https://www.uahirise.org
3. Geromichalos et al. (2020). *SLAM for autonomous planetary rovers*. J. Field Robotics 37(5).
