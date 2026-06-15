"""
world_model.py
==============
Author: Vahid Moeinifar - AGH university of Krakow
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from modules.terrain_analysis import TerrainLayers, terrain_stats


# ==========================================================================
# Label sets (Table 1 in paper, extended for icy-moon surface)
# ==========================================================================

LABEL_SETS = {
    "L1_traversability": [
        "free", "low_risk", "moderate_risk", "high_risk", "impassable"
    ],
    "L2_surface": [
        "smooth_ice", "rough_ice", "jagged_ice", "penitentes", "subsurface_anomaly"
    ],
    "L3_features": [
        "crack", "fissure", "shallow_crack", "deep_fissure", "crag",
        "ridge", "geyser", "thermal_vent", "vapor_plume",
        "crater", "impact_ejecta", "subsurface_gas_source"
    ],
    "L4_scientific": [
        "low_interest", "worth_investigation", "high_priority", "biosignature_candidate"
    ],
}


def traversability_to_label(trav: float) -> str:
    """Map continuous L₁ ∈ [0,1] to a discrete label."""
    if trav >= 0.8:   return "free"
    if trav >= 0.6:   return "low_risk"
    if trav >= 0.4:   return "moderate_risk"
    if trav >= 0.2:   return "high_risk"
    return "impassable"


# ==========================================================================
# MetaModel  MM(R)
# ==========================================================================

@dataclass
class ModelObject:
    """An entry in the model object library OM."""
    label: str
    depth_range: Tuple[float, float]        # (min, max) metres
    width_range: Tuple[float, float]
    traversability_range: Tuple[float, float]
    scientific_priority: float              # 0–1
    description: str = ""

    def fits(self, depth: float, width: float) -> bool:
        return (self.depth_range[0] <= depth <= self.depth_range[1] and
                self.width_range[0] <= width <= self.width_range[1])


class MetaModel:
    """
    MM(R) — meta-information shared across all environments.
    Contains:
      - OM: library of model objects (pre-seeded with icy-moon objects)
      - Rule base Ψ (if-then rules)
      - Label sets L
      - Image understanding algorithms ϕ (here: terrain analysis functions)
      - Ontology (simple dict-based)
    Updated by the generalization operator g.
    """

    def __init__(self, robot_id: str = "R"):
        self.robot_id = robot_id
        self.label_sets = LABEL_SETS.copy()
        self.OM: Dict[str, ModelObject] = self._init_object_library()
        self.rules: List[dict] = self._init_rules()
        self.ontology: Dict[str, List[str]] = self._init_ontology()
        self.supervision_log: List[dict] = []
        self.version: int = 0

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    @staticmethod
    def _init_object_library() -> Dict[str, ModelObject]:
        return {
            "shallow_crack": ModelObject(
                "shallow_crack", (0.01, 2.0), (0.05, 0.5), (0.4, 0.9),
                scientific_priority=0.5,
                description="Surface fracture < 2 m deep"),
            "fissure": ModelObject(
                "fissure", (2.0, 20.0), (0.1, 5.0), (0.1, 0.5),
                scientific_priority=0.8,
                description="Deep linear fracture in ice"),
            "deep_fissure": ModelObject(
                "deep_fissure", (20.0, 500.0), (0.5, 50.0), (0.0, 0.2),
                scientific_priority=1.0,
                description="Major penetrating fracture, possible subsurface access"),
            "geyser": ModelObject(
                "geyser", (0.5, 10.0), (0.5, 20.0), (0.0, 0.3),
                scientific_priority=1.0,
                description="Active water/vapour eruption feature"),
            "ridge": ModelObject(
                "ridge", (1.0, 50.0), (5.0, 500.0), (0.1, 0.5),
                scientific_priority=0.4,
                description="Elongated elevated terrain feature"),
            "crater": ModelObject(
                "crater", (5.0, 2000.0), (20.0, 5000.0), (0.0, 0.4),
                scientific_priority=0.7,
                description="Impact crater"),
            "smooth_plain": ModelObject(
                "smooth_plain", (0.0, 0.1), (100.0, 10000.0), (0.8, 1.0),
                scientific_priority=0.2,
                description="Flat, easily traversable ice plain"),
        }

    @staticmethod
    def _init_rules() -> List[dict]:
        """
        Rule base Ψ — if-then rules with XAI explanation ξ.
        Each rule: {condition: callable, action: str, explanation: str}
        """
        return [
            {
                "id": "R1_label_reassignment",
                "condition": lambda obs: (obs.get("label") == "fissure" and
                                         obs.get("adjacent_label") == "worth_investigation"),
                "action": "activate_narrow_camera; move_forward_10m",
                "explanation": "Fissure adjacent to interesting feature warrants "
                               "detailed structural analysis before traversal.",
            },
            {
                "id": "R2_metamodel_learning",
                "condition": lambda obs: obs.get("depth", 0) > 20.0,
                "action": "update_RTM; update_MM; reclassify_as_deep_fissure",
                "explanation": "Observed depth exceeds fissure model range — "
                               "meta-model must be extended to 'deep fissure'.",
            },
            {
                "id": "R3_high_risk_warning",
                "condition": lambda obs: obs.get("traversability", 1.0) < 0.2,
                "action": "halt; broadcast_warning; request_alternative_path",
                "explanation": "Traversability < 0.2 indicates impassable terrain. "
                               "Robot must stop and coordinate with team.",
            },
            {
                "id": "R4_scientific_priority",
                "condition": lambda obs: obs.get("subsurface_anomaly", False),
                "action": "increase_priority; notify_lander",
                "explanation": "Subsurface anomaly detected — potential water/biosignature. "
                               "Escalate to highest scientific priority.",
            },
            {
                "id": "R5_25D_slope_warning",
                "condition": lambda obs: obs.get("slope_deg", 0) > 30.0,
                "action": "reduce_speed; activate_thermal_imaging",
                "explanation": "Slope > 30° significantly reduces traversal speed "
                               "and increases tip-over risk. Activate imaging.",
            },
        ]

    @staticmethod
    def _init_ontology() -> Dict[str, List[str]]:
        return {
            "is_a": {
                "shallow_crack": ["crack"],
                "deep_fissure": ["fissure", "crack"],
                "geyser": ["active_feature", "water_source"],
                "ridge": ["terrain_feature"],
                "crater": ["impact_feature", "terrain_feature"],
            },
            "adjacent_to": {},
            "overlaps_with": {},
        }

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def match_object(self, depth: float, width: float) -> Optional[ModelObject]:
        """Find best-matching model object by geometry."""
        for obj in self.OM.values():
            if obj.fits(depth, width):
                return obj
        return None

    def fire_rules(self, observation: dict) -> List[dict]:
        """Run the inference engine — return list of triggered rules."""
        triggered = []
        for rule in self.rules:
            try:
                if rule["condition"](observation):
                    triggered.append(rule)
            except Exception:
                pass
        return triggered

    def update_object(self, label: str, depth: float, width: float):
        """
        g operator applied to MM: extend or split model object when
        observed geometry falls outside the existing range.
        """
        if label in self.OM:
            obj = self.OM[label]
            new_d_min = min(obj.depth_range[0], depth)
            new_d_max = max(obj.depth_range[1], depth)
            new_w_min = min(obj.width_range[0], width)
            new_w_max = max(obj.width_range[1], width)

            if depth > obj.depth_range[1] * 2:
                # Split into shallow and deep variants
                shallow_key = f"shallow_{label}"
                deep_key = f"deep_{label}"
                self.OM[shallow_key] = ModelObject(
                    shallow_key, obj.depth_range, obj.width_range,
                    obj.traversability_range, obj.scientific_priority,
                    f"Shallow variant of {label}")
                self.OM[deep_key] = ModelObject(
                    deep_key, (obj.depth_range[1], depth * 1.5), obj.width_range,
                    (0.0, 0.2), min(1.0, obj.scientific_priority + 0.2),
                    f"Deep variant of {label}")
                self._log_supervision(f"Split '{label}' → '{shallow_key}' + '{deep_key}'")
            else:
                # Just extend ranges
                self.OM[label] = ModelObject(
                    label, (new_d_min, new_d_max), (new_w_min, new_w_max),
                    obj.traversability_range, obj.scientific_priority,
                    obj.description)

            self.version += 1

    def _log_supervision(self, event: str):
        self.supervision_log.append({"time": time.time(), "event": event})


# ==========================================================================
# RealTimeModel  RTM(R,E,t)
# ==========================================================================

class RealTimeModel:
    """
    RTM(R,E,t) — the robot's live snapshot of its environment.

    2.5D Enhancement (vs. paper's 2D version):
    -------------------------------------------
    maps M(t)         : stores 3D elevation + slope + roughness layers
    traversability L₁ : continuous [0,1] field  (not binary)
    velocity V        : physics-based field from DEM
    physical P(t)     : temperature, seismic, ice elasticity (placeholders)
    """

    def __init__(self, robot_id: str, env_id: str):
        self.robot_id = robot_id
        self.env_id = env_id
        self.t = 0.0

        # Terrain layers (set when robot scans a DEM tile)
        self.terrain: Optional[TerrainLayers] = None

        # Knowledge zone: set of pixels (r,c) this robot has scanned
        self.knowledge_zone: set = set()

        # Feature and object collections
        self.features: List[dict] = []      # F(t): low-level features
        self.objects: List[dict] = []       # O(t): segmented objects
        self.labeled_objects: List[dict] = []  # OE(t): labeled objects

        # Physical measurements P(t)
        self.physical: List[dict] = []

        # Maps M(t): list of scanned sub-regions
        self.scanned_regions: List[dict] = []

        # Quality indicators H = (H1, H2)
        self.H1_completeness: float = 0.0   # Surface crack mapping completeness
        self.H2_accuracy: float = 0.0       # Subsurface water probability accuracy

    def set_terrain(self, layers: TerrainLayers):
        """Attach 2.5D terrain layers from DEM analysis."""
        self.terrain = layers

    def scan_region(self, rows: range, cols: range, t: float):
        """
        Simulate scanning a region: extract terrain features,
        update knowledge zone, and log measurements.
        """
        if self.terrain is None:
            return

        self.t = t
        pixel_count = 0

        for r in rows:
            for c in cols:
                if (0 <= r < self.terrain.shape[0] and
                        0 <= c < self.terrain.shape[1]):
                    self.knowledge_zone.add((r, c))
                    pixel_count += 1

        # Extract representative features from this scan window
        r_arr = np.array([rc[0] for rc in self.knowledge_zone])
        c_arr = np.array([rc[1] for rc in self.knowledge_zone])

        if len(r_arr) == 0:
            return

        trav_vals = self.terrain.traversability[r_arr, c_arr]
        slope_vals = self.terrain.slope_deg[r_arr, c_arr]

        # Detect high-slope ridges (f₁ Feature Detection with 2.5D)
        ridge_mask = slope_vals > 25.0
        if ridge_mask.any():
            self.features.append({
                "type": "ridge",
                "count": int(ridge_mask.sum()),
                "mean_slope": float(slope_vals[ridge_mask].mean()),
                "t": t,
            })

        # Detect low-traversability zones (potential fissures)
        hazard_mask = trav_vals < 0.3
        if hazard_mask.any():
            self.objects.append({
                "type": "hazard_zone",
                "pixel_count": int(hazard_mask.sum()),
                "mean_traversability": float(trav_vals[hazard_mask].mean()),
                "t": t,
            })

        # Detect deep depressions (potential fissures/craters via elevation curvature)
        elev_vals = self.terrain.elevation[r_arr, c_arr]
        region_mean = float(np.nanmean(elev_vals))
        deep_mask = elev_vals < region_mean - 20.0
        if deep_mask.any():
            depth_est = float(region_mean - np.nanmean(elev_vals[deep_mask]))
            self.labeled_objects.append({
                "label": "fissure" if depth_est > 5 else "shallow_crack",
                "depth_estimate_m": depth_est,
                "pixel_count": int(deep_mask.sum()),
                "traversability": float(np.nanmean(trav_vals[deep_mask])),
                "subsurface_anomaly": depth_est > 50,
                "t": t,
            })

        # Log physical measurement (simulated temperature for icy moon)
        temp_k = 110.0 + np.random.normal(0, 5)   # Europa ~110 K
        self.physical.append({
            "type": "temperature_K",
            "value": temp_k,
            "location": (float(np.mean(r_arr)), float(np.mean(c_arr))),
            "t": t,
        })

        # Update quality indicators
        total_pixels = self.terrain.shape[0] * self.terrain.shape[1]
        self.H1_completeness = min(1.0, len(self.knowledge_zone) / total_pixels)
        self.H2_accuracy = min(1.0, len(self.labeled_objects) * 0.05 + self.H1_completeness * 0.3)

    def get_shareable(self) -> dict:
        """
        Return the transferable portion of this RTM (Table 1 from paper).
        Non-transferable parts (robot-specific pixel maps) are excluded.
        """
        return {
            "robot_id": self.robot_id,
            "t": self.t,
            "features": self.features,           # General feature types (transferable)
            "objects": self.objects,              # Geological entities (transferable)
            "labeled_objects": self.labeled_objects,  # Common labels (transferable)
            "H1": self.H1_completeness,
            "H2": self.H2_accuracy,
            # Note: per-robot pixel maps and specific sensor readings NOT included
        }

    def merge_from(self, shared: dict):
        """
        Merge shared WM data received from another robot (⊕ operator).
        Implements the WM merge: set union for collections, max for quality.
        """
        # Merge features (union, avoid exact duplicates)
        existing_types = {f["type"] for f in self.features}
        for f in shared.get("features", []):
            if f["type"] not in existing_types:
                self.features.append(f)

        # Merge objects
        self.objects.extend(shared.get("objects", []))

        # Merge labeled objects (more careful — check for contradictions)
        for new_obj in shared.get("labeled_objects", []):
            # Check if we have a conflicting label for a similar location
            conflict = False
            for own_obj in self.labeled_objects:
                if (own_obj["label"] != new_obj["label"] and
                        abs(own_obj.get("depth_estimate_m", 0) -
                            new_obj.get("depth_estimate_m", 0)) < 5.0):
                    # Conflict: reconcile by creating "hybrid" label
                    hybrid_label = f"hybrid_{own_obj['label']}_{new_obj['label']}"
                    own_obj["label"] = hybrid_label
                    own_obj["reconciliation_note"] = (
                        f"Conflict between {own_obj['label']} (self) "
                        f"and {new_obj['label']} (peer {shared['robot_id']})"
                    )
                    conflict = True
                    break
            if not conflict:
                self.labeled_objects.append(new_obj)

        # Quality indicators: take max (best-case merge)
        self.H1_completeness = max(self.H1_completeness, shared.get("H1", 0))
        self.H2_accuracy = max(self.H2_accuracy, shared.get("H2", 0))

    def summary(self) -> dict:
        return {
            "robot": self.robot_id,
            "t": self.t,
            "knowledge_zone_pixels": len(self.knowledge_zone),
            "features": len(self.features),
            "objects": len(self.objects),
            "labeled_objects": len(self.labeled_objects),
            "H1_completeness": round(self.H1_completeness, 3),
            "H2_accuracy": round(self.H2_accuracy, 3),
        }


# ==========================================================================
# Decision Engine  X
# ==========================================================================

class DecisionEngine:
    """
    X — implements the Decision Making Cycle (DMC) from Section 3.2.

    Functions:
      f₁  Feature Detection  (slope edges, textures from 2.5D DEM)
      f₂  Object Identification (ridges, craters from elevation curvature)
      f₃  Entity Recognition  (geysers, subsurface anomalies)
      f₄  Label Assignment    (L₁ continuous traversability labels)
      Ψ   Inference Engine    (fires if-then rules from MM)
    """

    def __init__(self, mm: MetaModel):
        self.mm = mm
        self.action_log: List[dict] = []

    # f₁ — Feature detection from 2.5D slope gradients
    def f1_feature_detection(self, layers: TerrainLayers,
                             r: int, c: int, radius: int = 5) -> List[dict]:
        """Detect low-level features near pixel (r,c) using DEM derivatives."""
        features = []
        r0, r1 = max(0, r - radius), min(layers.shape[0], r + radius)
        c0, c1 = max(0, c - radius), min(layers.shape[1], c + radius)

        window_slope = layers.slope_deg[r0:r1, c0:c1]
        window_rough = layers.roughness[r0:r1, c0:c1]

        mean_slope = float(np.nanmean(window_slope))
        max_slope = float(np.nanmax(window_slope))

        if max_slope > 20:
            features.append({"type": "slope_edge", "max_slope_deg": max_slope,
                              "mean_slope_deg": mean_slope, "loc": (r, c)})
        if float(np.nanmean(window_rough)) > 0.2:
            features.append({"type": "rough_surface", "roughness_m": float(np.nanmean(window_rough)),
                              "loc": (r, c)})
        return features

    # f₂ — Object identification from curvature
    def f2_object_identification(self, layers: TerrainLayers,
                                 r: int, c: int) -> Optional[dict]:
        """Identify geological objects using elevation curvature (2nd derivative)."""
        if not (1 < r < layers.shape[0] - 1 and 1 < c < layers.shape[1] - 1):
            return None

        elev = layers.elevation
        # Laplacian curvature
        curv = (elev[r-1, c] + elev[r+1, c] + elev[r, c-1] + elev[r, c+1]
                - 4 * elev[r, c])

        if curv < -10:
            return {"type": "depression", "curvature": float(curv), "loc": (r, c),
                    "candidate": "fissure_or_crater"}
        if curv > 10:
            return {"type": "peak", "curvature": float(curv), "loc": (r, c),
                    "candidate": "ridge_or_geyser"}
        return None

    # f₃ — Entity recognition
    def f3_entity_recognition(self, object_dict: Optional[dict],
                              layers: TerrainLayers) -> Optional[dict]:
        """Recognise specific entities from detected objects."""
        if object_dict is None:
            return None
        r, c = object_dict["loc"]
        trav = float(layers.traversability[r, c])
        slope = float(layers.slope_deg[r, c])

        if object_dict["type"] == "depression" and trav < 0.2:
            return {"entity": "subsurface_anomaly", "traversability": trav,
                    "slope_deg": slope, "loc": (r, c)}
        if object_dict["type"] == "peak" and slope > 35:
            return {"entity": "geyser_candidate", "slope_deg": slope, "loc": (r, c)}
        return None

    # f₄ — Label assignment (2.5D gives continuous L₁ directly from slope/roughness)
    def f4_label_assignment(self, layers: TerrainLayers,
                            r: int, c: int) -> dict:
        """
        Slide 21: L₁ is computed directly from slope and roughness.
        2.5D advantage: label is geometry-grounded, not heuristic.
        """
        trav = float(layers.traversability[r, c])
        vel = float(layers.velocity[r, c])
        slope = float(layers.slope_deg[r, c])
        rough = float(layers.roughness[r, c])

        discrete_label = traversability_to_label(trav)

        return {
            "L1_continuous": trav,
            "L1_label": discrete_label,
            "velocity_ms": vel,
            "slope_deg": slope,
            "roughness_m": rough,
            "loc": (r, c),
        }

    # Inference Engine Ψ
    def run_inference(self, observation: dict) -> List[dict]:
        """
        Fire all applicable rules from MM against the current observation.
        Returns list of triggered rules with their actions and explanations.
        """
        triggered = self.mm.fire_rules(observation)
        for rule in triggered:
            self.action_log.append({
                "t": time.time(),
                "rule_id": rule["id"],
                "action": rule["action"],
                "observation": observation,
            })
        return triggered

    # Full DMC cycle at one pixel
    def decision_making_cycle(self, layers: TerrainLayers,
                              r: int, c: int) -> dict:
        """Run the complete f₁→f₂→f₃→f₄→Ψ pipeline at pixel (r,c)."""
        features = self.f1_feature_detection(layers, r, c)
        obj = self.f2_object_identification(layers, r, c)
        entity = self.f3_entity_recognition(obj, layers)
        labels = self.f4_label_assignment(layers, r, c)

        # Build observation dict for the inference engine
        obs = {
            "traversability": labels["L1_continuous"],
            "label": obj["candidate"].split("_")[0] if obj else "unknown",
            "adjacent_label": entity["entity"] if entity else "none",
            "depth": abs(layers.elevation[r, c] - float(np.nanmean(layers.elevation))),
            "slope_deg": labels["slope_deg"],
            "subsurface_anomaly": (entity is not None and
                                   "subsurface" in entity.get("entity", "")),
        }

        triggered_rules = self.run_inference(obs)

        return {
            "pixel": (r, c),
            "features": features,
            "object": obj,
            "entity": entity,
            "labels": labels,
            "triggered_rules": [r["id"] for r in triggered_rules],
            "actions": [r["action"] for r in triggered_rules],
        }


# ==========================================================================
# World Model  WM(R,E,t)
# ==========================================================================

class WorldModel:
    """
    Full WM = (MM, RTM, X, g)  for robot R in environment E at time t.
    """

    def __init__(self, robot_id: str, env_id: str = "Europa"):
        self.robot_id = robot_id
        self.env_id = env_id
        self.MM = MetaModel(robot_id)
        self.RTM = RealTimeModel(robot_id, env_id)
        self.X = DecisionEngine(self.MM)
        self.t = 0.0

    def attach_terrain(self, layers: TerrainLayers):
        """Attach 2.5D terrain layers to the RTM."""
        self.RTM.set_terrain(layers)

    def g_generalize(self, observation: dict):
        """
        Generalization operator g: update MM when observation doesn't fit.
        Implements Eq. 2 from the paper.
        """
        label = observation.get("label", "")
        depth = observation.get("depth", 0.0)
        width = observation.get("width", 1.0)

        if label in self.MM.OM:
            obj = self.MM.OM[label]
            if not obj.fits(depth, width):
                self.MM.update_object(label, depth, width)
                return True  # MM was updated
        return False

    def step(self, r: int, c: int, t: float) -> dict:
        """One DMC step at pixel (r,c)."""
        self.t = t
        if self.RTM.terrain is None:
            return {}
        result = self.X.decision_making_cycle(self.RTM.terrain, r, c)
        # Check if generalization needed
        obs = {
            "label": (result["object"]["candidate"].split("_")[0]
                      if result["object"] else "unknown"),
            "depth": abs(self.RTM.terrain.elevation[r, c] -
                         float(np.nanmean(self.RTM.terrain.elevation))),
            "width": 1.0,
        }
        updated = self.g_generalize(obs)
        result["mm_updated"] = updated
        return result

    def get_shareable_wm(self) -> dict:
        """Return the sharable sub-model WMs(Ri,E,t) for team merging."""
        return {
            "robot_id": self.robot_id,
            "t": self.t,
            "rtm": self.RTM.get_shareable(),
            "mm_version": self.MM.version,
            "rule_count": len(self.MM.rules),
            "object_library_keys": list(self.MM.OM.keys()),
        }

    def merge_wm(self, shared_wm: dict):
        """⊕ operator: merge received shared WM into this robot's RTM."""
        if "rtm" in shared_wm:
            self.RTM.merge_from(shared_wm["rtm"])

    def quality_indicators(self) -> dict:
        return {
            "H1_completeness": self.RTM.H1_completeness,
            "H2_accuracy": self.RTM.H2_accuracy,
        }
