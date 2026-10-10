"""Shared conventions for the PROJ -> PyTorch feasibility spike.

Conventions (mirror PROJ's fwd.cpp / inv.cpp split):
  * A Projection subclass implements `_fwd(lam, phi)` and `_inv(x, y)` on the
    UNIT ellipsoid (a = 1) with `lam` already reduced by lon_0, exactly like a
    PROJ `P->fwd` / `P->inv` function. k_0 is applied inside `_fwd`/`_inv` when
    PROJ does so (most projections use P->k0 internally).
  * The base class does what fwd_prepare/fwd_finalize/inv_prepare/inv_finalize
    do: latitude range check, lam -= lon_0 + adjlon (unless +over), scale by a,
    add x_0/y_0 (and the inverse of all that).
  * Coordinates: tensors of shape [..., D] with D >= 2. Geodetic input is in
    RADIANS (lon, lat), projected output in metres. Extra components (z, t)
    pass through untouched.
  * Errors: PROJ returns HUGE_VAL + errno per point. Here a failed point is NaN.
  * dtype: whatever the input tensor is (float64 expected; float32 for
    precision experiments). Constants are Python floats (computed in float64).
"""
import math
from typing import Dict, Union

import torch

ELLPS = {
    # name: (a, rf)
    "WGS84": (6378137.0, 298.257223563),
    "GRS80": (6378137.0, 298.257222101),
    "intl": (6378388.0, 297.0),
    "clrk66": (6378206.4, None),  # b = 6356583.8
    "bessel": (6377397.155, 299.1528128),
    "airy": (6377563.396, 299.3249646),
    "clrk80ign": (6378249.2, 293.4660212936269),
    "krass": (6378245.0, 298.3),
    "sphere": (6370997.0, None),  # PROJ "sphere": R = 6370997
}

HALFPI = math.pi / 2
PJ_EPS_LAT = 1e-12


def parse_proj_string(s: str) -> Dict[str, Union[str, bool]]:
    """'+proj=lcc +lat_1=33 +no_defs' -> {'proj':'lcc','lat_1':'33','no_defs':True}.
    Only for single (non-pipeline) operations."""
    out = {}
    for tok in s.replace("\\", " ").split():
        tok = tok.lstrip("+")
        if not tok:
            continue
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
        else:
            out[tok] = True
    return out


def dms_or_float(v) -> float:
    """PROJ angles may be given as 10d30'N etc.; the spike only supports plain floats."""
    return float(v)


class Ellipsoid:
    def __init__(self, params: dict):
        a = None
        rf = None
        b = None
        es = None
        if "R" in params:
            a = float(params["R"])
            es = 0.0
        else:
            if "ellps" in params:
                a, rf = ELLPS[params["ellps"]]
                if params["ellps"] == "clrk66":
                    b = 6356583.8
                if params["ellps"] == "sphere":
                    es = 0.0
            elif "a" not in params and "b" not in params:
                a, rf = ELLPS["GRS80"]  # PROJ default ellipsoid
            if "a" in params:
                a = float(params["a"])
            if "rf" in params:
                rf = float(params["rf"])
            if "f" in params:
                rf = 1.0 / float(params["f"]) if float(params["f"]) != 0 else None
            if "b" in params:
                b = float(params["b"])
            if "es" in params:
                es = float(params["es"])
            if "e" in params:
                es = float(params["e"]) ** 2
            if es is None:
                if b is not None:
                    es = 1.0 - (b * b) / (a * a)
                elif rf is not None:
                    f = 1.0 / rf
                    es = 2 * f - f * f
                else:
                    es = 0.0
        self.a = a
        self.es = es
        self.e = math.sqrt(es)
        self.one_es = 1.0 - es
        self.rone_es = 1.0 / self.one_es
        self.f = 1.0 - math.sqrt(1.0 - es)
        self.b = a * math.sqrt(1.0 - es)
        self.ra = 1.0 / a


def adjlon(lon: torch.Tensor) -> torch.Tensor:
    """PROJ adjlon(): reduce to [-pi, pi]; identity (and identity gradient) if already inside."""
    wrapped = torch.remainder(lon + math.pi, 2 * math.pi) - math.pi
    return torch.where(lon.abs() <= math.pi + 1e-12, lon, wrapped)


class Projection(torch.nn.Module):
    """Base class: PROJ's generic fwd/inv pre- and post-processing."""

    def __init__(self, params: Union[str, dict]):
        super().__init__()
        if isinstance(params, str):
            params = parse_proj_string(params)
        self.params = params
        self.ell = Ellipsoid(params)
        self.a = self.ell.a
        self.es = self.ell.es
        self.e = self.ell.e
        self.one_es = self.ell.one_es
        self.rone_es = self.ell.rone_es
        self.lam0 = math.radians(float(params.get("lon_0", 0.0)))
        self.phi0 = math.radians(float(params.get("lat_0", 0.0)))
        self.x0 = float(params.get("x_0", 0.0))
        self.y0 = float(params.get("y_0", 0.0))
        self.k0 = float(params.get("k_0", params.get("k", 1.0)))
        self.over = bool(params.get("over", False))
        # to_meter / units are ignored by the spike (assume metres)

    # --- to be implemented by subclasses (unit ellipsoid, lam relative to lon_0)
    def _fwd(self, lam, phi):
        raise NotImplementedError

    def _inv(self, x, y):
        raise NotImplementedError

    # --- generic wrappers
    def forward(self, coo: torch.Tensor) -> torch.Tensor:
        lam, phi = coo[..., 0], coo[..., 1]
        bad = phi.abs() > HALFPI + PJ_EPS_LAT
        phi = phi.clamp(-HALFPI, HALFPI)
        lam = lam - self.lam0
        if not self.over:
            lam = adjlon(lam)
        x, y = self._fwd(lam, phi)
        x = self.a * x + self.x0
        y = self.a * y + self.y0
        nan = torch.full_like(x, float("nan"))
        x = torch.where(bad, nan, x)
        y = torch.where(bad, nan, y)
        return torch.cat([torch.stack([x, y], -1), coo[..., 2:]], -1)

    def inverse(self, coo: torch.Tensor) -> torch.Tensor:
        x = (coo[..., 0] - self.x0) * self.ell.ra
        y = (coo[..., 1] - self.y0) * self.ell.ra
        lam, phi = self._inv(x, y)
        lam = lam + self.lam0
        if not self.over:
            lam = adjlon(lam)
        return torch.cat([torch.stack([lam, phi], -1), coo[..., 2:]], -1)
