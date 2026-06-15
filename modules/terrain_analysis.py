"""
terrain_analysis.py
===================
Author: Vahid Moeinifar - AGH university of Krakow

"""

import numpy as np
from scipy.ndimage import uniform_filter, generic_filter
from dataclasses import dataclass
from modules.dem_loader import DEMData


# -------------------------------------------------------------------------
# Parameters (match the paper/presentation exactly)
# -------------------------------------------------------------------------

@dataclass
class TerrainParams:
    """Tunable parameters for the 2.5D terrain model."""
    alpha: float = 0.6          # Slope weight in traversability
    beta: float = 0.4           # Roughness weight in traversability
    theta_max: float = 45.0     # Max traversable slope [degrees]
    R_max: float = 0.5          # Max traversable roughness [metres]
    V_max: float = 2.0          # Max robot speed [m/s]
    slope_limit: float = 45.0   # Speed → 0 at this slope
    roughness_limit: float = 0.5
    roughness_window: int = 5   # Size of roughness estimation window
    beta_safety: float = 5.0   # Safety amplifier for damage cost
    pixel_scale: float = 1.0   # Metres per pixel (overridden by DEM)


@dataclass
class TerrainLayers:
    """All computed terrain layers for one DEM."""
    elevation: np.ndarray       # Z(x,y)  [m]
    slope_grad: np.ndarray      # |∇Z|    [rise/run, dimensionless]
    slope_deg: np.ndarray       # θ       [degrees]
    roughness: np.ndarray       # R(x,y)  [m]
    traversability: np.ndarray  # L₁      [0,1]
    velocity: np.ndarray        # V(x,y)  [m/s]
    params: TerrainParams

    @property
    def shape(self):
        return self.elevation.shape


# -------------------------------------------------------------------------
# Core analysis
# -------------------------------------------------------------------------

class TerrainAnalyzer:
    """
    Compute all 2.5D terrain layers from a loaded DEMData object.
    Implements the mathematical pipeline from Slides 20–23.
    """

    def __init__(self, params: TerrainParams = None):
        self.params = params or TerrainParams()

    def analyze(self, dem: DEMData) -> TerrainLayers:
        """Run the full 2.5D pipeline on a DEM and return all layers."""
        p = self.params
        p.pixel_scale = dem.scale  # Use DEM's actual resolution

        elev = dem.elevation.copy()
        # Fill NaN with neighbour-interpolated values for gradient computation
        elev_filled = self._fill_nan(elev)

        slope_grad, slope_deg = self._compute_slope(elev_filled, p.pixel_scale)
        roughness = self._compute_roughness(elev_filled, p.roughness_window)
        traversability = self._compute_traversability(slope_deg, roughness, p)
        velocity = self._compute_velocity(slope_deg, roughness, p)

        # Re-apply NaN mask where elevation is invalid
        nan_mask = ~dem.valid_mask
        for arr in (slope_grad, slope_deg, roughness, traversability, velocity):
            arr[nan_mask] = np.nan

        return TerrainLayers(
            elevation=elev,
            slope_grad=slope_grad,
            slope_deg=slope_deg,
            roughness=roughness,
            traversability=traversability,
            velocity=velocity,
            params=p,
        )

    # ------------------------------------------------------------------
    # Step 1: Slope  (Slide 20)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_slope(elev: np.ndarray,
                       pixel_scale: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Slide 20: ∇Z(x,y) = √[(∂Z/∂x)² + (∂Z/∂y)²]
        θ(x,y)  = arctan(∇Z)   [degrees]
        """
        dz_dy, dz_dx = np.gradient(elev, pixel_scale)
        grad_mag = np.sqrt(dz_dx ** 2 + dz_dy ** 2)          # |∇Z| [rise/run]
        slope_deg = np.degrees(np.arctan(grad_mag))           # θ [deg]
        return grad_mag.astype(np.float32), slope_deg.astype(np.float32)

    # ------------------------------------------------------------------
    # Step 2: Roughness  (Slide 20)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_roughness(elev: np.ndarray, window: int = 5) -> np.ndarray:
        """
        Slide 20: R(x,y) = std(Z) over a window×window neighbourhood.
        Uses the identity: Var = E[Z²] − E[Z]²
        """
        mean_z = uniform_filter(elev, size=window)
        mean_z2 = uniform_filter(elev ** 2, size=window)
        var = np.maximum(mean_z2 - mean_z ** 2, 0.0)
        roughness = np.sqrt(var)
        return roughness.astype(np.float32)

    # ------------------------------------------------------------------
    # Step 3: Traversability  (Slide 21)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_traversability(slope_deg: np.ndarray,
                                roughness: np.ndarray,
                                p: TerrainParams) -> np.ndarray:
        """
        Slide 21:
        L₁(x,y) = 1 − [α·(θ/θ_max) + β·(R/R_max)]
        Clipped to [0, 1].
        """
        term = p.alpha * (slope_deg / p.theta_max) + p.beta * (roughness / p.R_max)
        trav = np.clip(1.0 - term, 0.0, 1.0)
        return trav.astype(np.float32)

    # ------------------------------------------------------------------
    # Step 4: Velocity  (Slide 22)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_velocity(slope_deg: np.ndarray,
                          roughness: np.ndarray,
                          p: TerrainParams) -> np.ndarray:
        """
        Slide 22:
        V = V_max · f_slope · f_roughness
        f_slope    = max(0, 1 − θ/slope_limit)
        f_roughness = max(0, 1 − R/roughness_limit)
        """
        f_slope = np.maximum(0.0, 1.0 - slope_deg / p.slope_limit)
        f_rough = np.maximum(0.0, 1.0 - roughness / p.roughness_limit)
        velocity = p.V_max * f_slope * f_rough
        return velocity.astype(np.float32)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _fill_nan(arr: np.ndarray) -> np.ndarray:
        """Replace NaN with nearest-valid-neighbour (simple row-fill)."""
        out = arr.copy()
        nan_mask = np.isnan(out)
        if not nan_mask.any():
            return out
        # Forward fill along rows, then backward
        idx = np.where(~nan_mask, np.arange(arr.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        out[nan_mask] = out[np.arange(arr.shape[0])[:, None], idx][nan_mask]
        # Any remaining NaN → zero
        out[np.isnan(out)] = 0.0
        return out


# -------------------------------------------------------------------------
# Edge-cost functions used by the path planner
# -------------------------------------------------------------------------

def edge_cost_time(layers: TerrainLayers,
                   r0: int, c0: int, r1: int, c1: int) -> float:
    """
    Time cost for moving from pixel (r0,c0) to (r1,c1).
    time = dist_2D / V(mid)
    """
    p = layers.params
    dr = (r1 - r0) * p.pixel_scale
    dc = (c1 - c0) * p.pixel_scale
    dist_2d = np.sqrt(dr ** 2 + dc ** 2)
    v = 0.5 * (layers.velocity[r0, c0] + layers.velocity[r1, c1])
    v = max(v, 0.01)  # avoid division by zero
    return dist_2d / v


def edge_cost_damage(layers: TerrainLayers,
                     r0: int, c0: int, r1: int, c1: int) -> float:
    """
    Damage cost for moving from pixel (r0,c0) to (r1,c1).
    Slide 23: cost = (1 − avg_trav) · dist_3D · β_safety
    dist_3D = √(Δx² + Δy² + Δz²)
    """
    p = layers.params
    dr = (r1 - r0) * p.pixel_scale
    dc = (c1 - c0) * p.pixel_scale
    dz = float(layers.elevation[r1, c1]) - float(layers.elevation[r0, c0])
    dist_3d = np.sqrt(dr ** 2 + dc ** 2 + dz ** 2)
    avg_trav = 0.5 * (float(layers.traversability[r0, c0]) +
                      float(layers.traversability[r1, c1]))
    return (1.0 - avg_trav) * dist_3d * p.beta_safety


def edge_cost_combined(layers: TerrainLayers,
                       r0: int, c0: int, r1: int, c1: int,
                       w_time: float = 0.5, w_damage: float = 0.5) -> float:
    """Weighted combination of time and damage costs (for multi-criteria A*)."""
    ct = edge_cost_time(layers, r0, c0, r1, c1)
    cd = edge_cost_damage(layers, r0, c0, r1, c1)
    return w_time * ct + w_damage * cd


def flat_2d_cost(r0: int, c0: int, r1: int, c1: int,
                 pixel_scale: float = 1.0, V_max: float = 2.0) -> float:
    """Baseline flat-2D cost (ignoring elevation). Used for comparison."""
    dr = (r1 - r0) * pixel_scale
    dc = (c1 - c0) * pixel_scale
    dist = np.sqrt(dr ** 2 + dc ** 2)
    return dist / V_max


# -------------------------------------------------------------------------
# Statistics helper
# -------------------------------------------------------------------------

def terrain_stats(layers: TerrainLayers) -> dict:
    """Return summary statistics for all terrain layers."""
    def s(arr):
        v = arr[np.isfinite(arr)]
        if len(v) == 0:
            return {}
        return {"mean": float(v.mean()), "std": float(v.std()),
                "min": float(v.min()), "max": float(v.max())}

    return {
        "elevation_m":      s(layers.elevation),
        "slope_deg":        s(layers.slope_deg),
        "roughness_m":      s(layers.roughness),
        "traversability":   s(layers.traversability),
        "velocity_ms":      s(layers.velocity),
    }
