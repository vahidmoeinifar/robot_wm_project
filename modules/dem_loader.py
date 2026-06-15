"""
dem_loader.py
=============
Load NASA PDS (Planetary Data System) .IMG Digital Elevation Model files.

Author: Vahid Moeinifar - AGH university of Krakow

"""

import os
import re
import struct
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DEMData:
    """Container for a loaded DEM."""
    elevation: np.ndarray          # 2-D float32 array of elevations in metres
    rows: int
    cols: int
    scale: float                   # metres per pixel (horizontal)
    scale_z: float                 # vertical scale factor (already applied)
    offset_z: float                # vertical offset (already applied)
    valid_mask: np.ndarray         # bool array, True where data is valid
    filename: str
    source: str = "NASA PDS .IMG"

    @property
    def shape(self) -> Tuple[int, int]:
        return self.elevation.shape

    @property
    def min_elevation(self) -> float:
        return float(np.nanmin(self.elevation[self.valid_mask]))

    @property
    def max_elevation(self) -> float:
        return float(np.nanmax(self.elevation[self.valid_mask]))

    @property
    def elevation_range(self) -> float:
        return self.max_elevation - self.min_elevation

    def info(self) -> str:
        return (
            f"DEM: {os.path.basename(self.filename)}\n"
            f"  Shape     : {self.rows} x {self.cols} pixels\n"
            f"  Scale     : {self.scale:.4f} m/pixel\n"
            f"  Elevation : [{self.min_elevation:.2f}, {self.max_elevation:.2f}] m  "
            f"(range {self.elevation_range:.2f} m)\n"
            f"  Valid px  : {self.valid_mask.sum():,} / {self.rows * self.cols:,}"
        )


# ---------------------------------------------------------------------------
# PDS3 value converter  — handles 16#FF7FFFFB# hex literals, units, etc.
# ---------------------------------------------------------------------------

def _pds_to_float(value: str, default: float = 0.0) -> float:
    if not value:
        return default
    # Take only the first whitespace-delimited token (drops unit strings like <m>)
    v = str(value).strip().split()[0].rstrip('>')

    # PDS3 radix literal: base#digits#
    m = re.match(r'^(\d+)#([0-9A-Fa-f]+)#$', v)
    if m:
        base = int(m.group(1))
        digits = m.group(2)
        try:
            int_val = int(digits, base)
            # Reinterpret 32-bit pattern as IEEE-754 float
            packed = struct.pack('>I', int_val & 0xFFFFFFFF)
            return float(struct.unpack('>f', packed)[0])
        except Exception:
            return default

    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _pds_to_int(value: str, default: int = 0) -> int:
    v = str(value).strip().split()[0]
    m = re.match(r'^(\d+)#([0-9A-Fa-f]+)#$', v)
    if m:
        try:
            return int(m.group(2), int(m.group(1)))
        except Exception:
            return default
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

class DEMLoader:

    _NULL_INT_VALUES = {-32768, -9999, 0x7FFFFFFF}

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._label: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> DEMData:
        """Parse and load the DEM from disk."""
        self._label = self._read_label()
        elevation, valid_mask = self._read_binary()
        scale    = self._get_scale()
        scale_z, offset_z = self._get_z_params()

        # Apply vertical scaling (only if scale_z is meaningful)
        elevation = elevation.astype(np.float32) * scale_z + offset_z

        # Mask nodata
        elevation[~valid_mask] = np.nan

        rows, cols = elevation.shape
        return DEMData(
            elevation=elevation,
            rows=rows,
            cols=cols,
            scale=scale,
            scale_z=scale_z,
            offset_z=offset_z,
            valid_mask=valid_mask,
            filename=self.filepath,
        )

    # ------------------------------------------------------------------
    # Label parsing
    # ------------------------------------------------------------------

    def _read_label(self) -> dict:
        """Read PDS3 label from the file header or a companion .LBL file."""
        label_text = ""

        # Try companion .LBL first (same name, different extension)
        for ext in (".LBL", ".lbl", ".lbl"):
            lbl_path = os.path.splitext(self.filepath)[0] + ext
            if os.path.exists(lbl_path):
                with open(lbl_path, "r", errors="ignore") as f:
                    label_text = f.read(16384)
                break

        if not label_text:
            # Try embedded label at file start (read up to 16 kB)
            with open(self.filepath, "rb") as f:
                raw = f.read(16384)
            label_text = raw.decode("ascii", errors="ignore")

        return self._parse_pds_label(label_text)

    @staticmethod
    def _parse_pds_label(text: str) -> dict:
        """Parse key = value pairs from a PDS3 ASCII label."""
        label = {}
        for line in text.splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("/*") and not line.startswith("END"):
                key, _, value = line.partition("=")
                # Strip comments, quotes, and surrounding whitespace
                value = value.split("/*")[0].strip().strip('"').strip("'")
                if key.strip():
                    label[key.strip().upper()] = value
        return label

    # ------------------------------------------------------------------
    # Binary data reading
    # ------------------------------------------------------------------

    def _read_binary(self) -> Tuple[np.ndarray, np.ndarray]:
        """Read the raw binary elevation data."""
        lbl = self._label

        rows        = _pds_to_int(lbl.get("LINES",        lbl.get("LINE_SAMPLES", "0")))
        cols        = _pds_to_int(lbl.get("LINE_SAMPLES",  lbl.get("SAMPLES", "0")))
        sample_bits = _pds_to_int(lbl.get("SAMPLE_BITS", "16"))
        sample_type = lbl.get("SAMPLE_TYPE", "LSB_INTEGER").upper()

        # ----- Byte offset to data -----
        label_records = _pds_to_int(lbl.get("LABEL_RECORDS", "0"))
        record_bytes  = _pds_to_int(lbl.get("RECORD_BYTES",  "512"))
        if record_bytes == 0:
            record_bytes = 512

        if label_records:
            offset = label_records * record_bytes
        else:
            # HiRISE DTMs: ^IMAGE = <record_number>
            ptr = lbl.get("^IMAGE", "")
            if ptr:
                rec_no = _pds_to_int(ptr, default=1)
                offset = max(0, rec_no - 1) * record_bytes
            else:
                offset = 0

        file_size = os.path.getsize(self.filepath)

        # ----- Determine numpy dtype -----
        is_float  = ("REAL" in sample_type or "FLOAT" in sample_type)
        is_msb    = ("MSB" in sample_type or "SUN"   in sample_type or
                     "MAC" in sample_type or "IEEE"  in sample_type)
        byteorder = ">" if is_msb else "<"

        if sample_bits == 32:
            base_dt = "f4" if is_float else "i4"
        elif sample_bits == 16:
            base_dt = "u2" if "UNSIGNED" in sample_type else "i2"
        else:
            base_dt = "i2"

        dt = np.dtype(byteorder + base_dt)

        expected_bytes = rows * cols * dt.itemsize

        # ----- Auto-detect if label gave us nothing usable -----
        if rows == 0 or cols == 0 or offset + expected_bytes > file_size:
            rows, cols, offset, dt = self._auto_detect(file_size)

        with open(self.filepath, "rb") as f:
            f.seek(offset)
            raw = np.frombuffer(f.read(rows * cols * dt.itemsize), dtype=dt)

        if len(raw) < rows * cols:
            pad = np.zeros(rows * cols - len(raw), dtype=dt)
            raw = np.concatenate([raw, pad])

        elevation = raw[: rows * cols].reshape(rows, cols).astype(np.float32)

        # ----- Build valid mask -----
        # MISSING_CONSTANT / NULL may be a PDS3 hex literal — use _pds_to_float
        null_raw = lbl.get("MISSING_CONSTANT", lbl.get("NULL", ""))
        if null_raw:
            null_float = _pds_to_float(null_raw, default=np.nan)
        else:
            null_float = np.nan

        valid_mask = np.ones(elevation.shape, dtype=bool)

        # Mask IEEE special values (NaN, Inf)
        valid_mask &= np.isfinite(elevation)

        # Mask exact null float value
        if np.isfinite(null_float):
            valid_mask &= (np.abs(elevation - null_float) > 0.5)

        # Mask common integer sentinel values (before scaling)
        for sentinel in self._NULL_INT_VALUES:
            valid_mask &= (elevation.astype(np.int64) != sentinel)

        return elevation, valid_mask

    def _auto_detect(self, file_size: int):
        """Heuristic auto-detection for unknown IMG formats."""
        # Try common HiRISE 16-bit signed square sizes
        candidates = []
        for n in range(64, 8192, 64):
            if n * n * 2 <= file_size:
                candidates.append(n)

        if candidates:
            n = candidates[-1]
            dt = np.dtype("<i2")
            # Estimate label offset: scan first 2 kB for binary start
            with open(self.filepath, "rb") as f:
                head = f.read(min(2048, file_size))
            offset = 0
            # Find last printable-ASCII byte before binary data
            for i in range(min(len(head) - 2, 2048)):
                if head[i] == 0 and head[i+1] == 0:
                    offset = (i // 512) * 512
                    break
            return n, n, offset, dt

        # Last resort
        return 512, 512, 0, np.dtype("<i2")

    def _get_scale(self) -> float:
        """Extract horizontal pixel scale in metres."""
        for key in ("MAP_SCALE", "PIXEL_SCALE", "RESOLUTION",
                    "HORIZONTAL_PIXEL_SCALE", "IMAGE_RESOLUTION"):
            val = self._label.get(key, "")
            if val:
                f = _pds_to_float(val, default=0.0)
                if f > 0:
                    return f
        return 1.0  # Default: 1 m/pixel

    def _get_z_params(self) -> Tuple[float, float]:
        """Extract vertical scale and offset."""
        scale_z  = _pds_to_float(
            self._label.get("SCALING_FACTOR",
            self._label.get("OFFSET_MULTIPLIER", "1.0")), default=1.0)
        offset_z = _pds_to_float(
            self._label.get("OFFSET", "0.0"), default=0.0)
        if scale_z == 0.0:
            scale_z = 1.0
        return scale_z, offset_z


# ---------------------------------------------------------------------------
# Synthetic DEM generator — used when no .IMG files are available
# ---------------------------------------------------------------------------

def make_synthetic_dem(rows: int = 512, cols: int = 512,
                       scale: float = 1.0, seed: int = 42) -> DEMData:
    """
    Generate a realistic synthetic DEM using fractal terrain simulation.
    Mimics the rugged icy-moon terrain described in the paper.
    """
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)

    elev = np.zeros((rows, cols), dtype=np.float32)
    amplitude = 50.0
    for octave in range(8):
        freq = 2 ** octave
        amp  = amplitude / (freq ** 0.9)
        noise = rng.standard_normal((rows, cols)).astype(np.float32)
        smoothed = gaussian_filter(noise, sigma=max(1, rows // (freq * 2)))
        elev += amp * smoothed

    # Add a few craters
    for _ in range(5):
        cx = rng.integers(50, rows - 50)
        cy = rng.integers(50, cols - 50)
        r  = rng.integers(20, 60)
        yy, xx = np.ogrid[:rows, :cols]
        dist = np.sqrt((yy - cx) ** 2 + (xx - cy) ** 2)
        rim  = np.exp(-((dist - r) / 5) ** 2) * 20
        bowl = -np.exp(-(dist / (r * 0.7)) ** 2) * 30
        elev += (rim + bowl).astype(np.float32)

    # Add a ridge
    ridge_col = cols // 3
    elev[:, ridge_col - 5: ridge_col + 5] += 40

    # Normalise to [-50, 200] m
    elev = (elev - elev.min()) / (elev.max() - elev.min()) * 250 - 50

    valid_mask = np.ones((rows, cols), dtype=bool)

    return DEMData(
        elevation=elev,
        rows=rows,
        cols=cols,
        scale=scale,
        scale_z=1.0,
        offset_z=0.0,
        valid_mask=valid_mask,
        filename="synthetic_dem",
        source="Synthetic (fractal terrain)",
    )


def load_dem_or_synthetic(filepath: Optional[str],
                          rows: int = 512, cols: int = 512) -> DEMData:
    """Try to load a real .IMG file; fall back to synthetic if unavailable."""
    if filepath and os.path.exists(filepath):
        try:
            loader = DEMLoader(filepath)
            dem = loader.load()
            print(f"  [OK] Loaded real DEM: {os.path.basename(filepath)}")
            print(dem.info())
            return dem
        except Exception as e:
            print(f"  [WARN] Could not load {filepath}: {e}. Using synthetic.")

    print(f"  [INFO] Generating synthetic DEM ({rows}x{cols})")
    return make_synthetic_dem(rows, cols)