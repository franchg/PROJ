"""geos, ob_tran, eqc, latlong (+ moll, for ob_tran composition) ported from
PROJ (sources in src/ of this PROJ tree) to PyTorch.

Ported C++ (file:line refers to src/ of this PROJ tree):
  * projections/geos.cpp     Geostationary Satellite View, sphere + ellipsoid,
                             +sweep=x / +sweep=y                (geos.cpp:53-246)
  * projections/ob_tran.cpp  General Oblique Transformation (rotated pole),
                             oblique + transverse forms, the three ways of
                             giving the new pole, composition with any torch
                             Projection as o_proj               (ob_tran.cpp:28-301)
  * projections/eqc.cpp      Equidistant Cylindrical, sphere (EPSG:1029) and
                             ellipsoid (EPSG:1028)              (eqc.cpp:41-145)
  * projections/latlong.cpp  latlong / longlat / latlon / lonlat (latlong.cpp:39-98)
  * projections/moll.cpp     Mollweide (only so that the gie ob_tran+moll test
                             vectors can be checked; shows composition with an
                             iterative projection)              (moll.cpp:22-82)
  * helpers: aasincos.cpp (aasin, aatan2), mlfn.cpp + latitudes.cpp (meridian
    distance, pj_enfn/pj_mlfn/pj_inv_mlfn via Clenshaw), ell_set.cpp (shape
    parameters, so that es is bit-identical to PROJ), fwd.cpp/inv.cpp (the
    PJ_IO_UNITS_RADIANS / PJ_IO_UNITS_WHATEVER handling that core.Projection
    does not cover).

Conventions are those of core.py (unit ellipsoid _fwd/_inv, lam relative to
lon_0, failures -> NaN, tensors [..., D]).

Branches: every per-point C++ "if" is a torch.where. Where an untaken branch
could produce NaN/inf (sqrt of a negative discriminant, asin(|v|>1), atan2(0,0),
division by cos(phi)=0) the *inputs* of that branch are first replaced by
harmless values ("double where"), so that the backward pass gets exact zeros
instead of 0*inf = NaN for the masked points.

Iterations: only moll's forward iterates (Newton, MAX_ITER=30, LOOP_TOL=1e-7).
It is a masked loop with exactly PROJ's iteration cap: a point stops being
updated after the iteration in which |V| < LOOP_TOL (the update of that
iteration is applied, like the C++), and points that never converge are snapped
to +-pi/2 like moll.cpp:34-35. The Python loop has a constant trip count, so
there is no host sync and torch.compile unrolls it.
"""
import math
from typing import Dict, Optional, Union

import torch

from core import ELLPS, HALFPI, PJ_EPS_LAT, Projection, adjlon, parse_proj_string

M_PI = math.pi
M_TWOPI = 2.0 * math.pi
DEG_TO_RAD = math.pi / 180.0


# ---------------------------------------------------------------------------
# Ellipsoid shape exactly as PROJ computes it (ell_set.cpp:70-340, init.cpp:336-365).
# core.Ellipsoid derives es from b as 1 - b^2/a^2 whereas PROJ uses
# f = (a-b)/a; es = 2f - f^2 (ell_set.cpp:332-333). The two differ in the last
# bit, which is enough to move geos' visibility boundary by an ulp, so the
# classes below recompute es the PROJ way.
_ELLPS_SHAPE = {  # name -> ("rf"|"b", value), from PROJ's ellps.cpp table
    "WGS84": ("rf", 298.257223563), "GRS80": ("rf", 298.257222101),
    "intl": ("rf", 297.0), "clrk66": ("b", 6356583.8),
    "bessel": ("rf", 299.1528128), "airy": ("rf", 299.3249646),
    "clrk80ign": ("rf", 293.4660212936269), "krass": ("rf", 298.3),
    "sphere": ("b", 6370997.0),
}


def proj_ellipsoid(params) -> (float, float):
    """Return (a, es) following pj_ellipsoid() (ell_set.cpp:70-131)."""
    if "R" in params:  # ell_set.cpp:93-100: R overrules everything
        return float(params["R"]), 0.0
    p = dict(params)
    # init.cpp:336-365: append ellps=GRS80 if no size/shape is given at all
    if not any(k in p for k in ("datum", "ellps", "a", "b", "rf", "f", "e", "es")):
        p["ellps"] = "GRS80"
    a, es = 0.0, 0.0
    if "ellps" in p:  # ellps_ellps, ell_set.cpp:137-194
        a = ELLPS[p["ellps"]][0]
        kind, val = _ELLPS_SHAPE[p["ellps"]]
        if kind == "rf":
            f = 1.0 / val
            es = 2 * f - f * f
        elif val != a:
            f = (a - val) / a
            es = 2 * f - f * f
    if "a" in p:  # ellps_size, ell_set.cpp:197-225
        a = float(p["a"])
    # ellps_shape, ell_set.cpp:228-340: first of rf, f, es, e, b wins
    for key in ("rf", "f", "es", "e", "b"):
        if key in p:
            v = float(p[key])
            if key == "rf":
                f = 1.0 / v
                es = 2 * f - f * f
            elif key == "f":
                es = 2 * v - v * v
            elif key == "es":
                es = v
            elif key == "e":
                es = v * v
            else:
                f = (a - v) / a  # ell_set.cpp:328-333
                es = 0.0 if v == a else 2 * f - f * f
            break
    return a, es


class _ProjShapeMixin:
    """Re-derive es-dependent attributes exactly like pj_calc_ellipsoid_params
    (ell_set.cpp:576-630). Only es-derived attributes are replaced; a, ra, x0...
    stay those of core.Projection (identical to PROJ for all supported inputs)."""

    def _set_proj_shape(self):
        a, es = proj_ellipsoid(self.params)
        if a != self.a:
            raise ValueError("core.Ellipsoid and PROJ disagree on a: %r vs %r" % (self.a, a))
        self.es = es
        self.e = math.sqrt(es)
        self.one_es = 1.0 - es
        self.rone_es = 1.0 / self.one_es
        alpha = math.asin(self.e)
        self.n = math.tan(alpha / 2) ** 2  # third flattening, ell_set.cpp:611


# ---------------------------------------------------------------------------
# aasincos.cpp
ONE_TOL = 1.00000000000001  # aasincos.cpp:8
ATOL = 1e-50                # aasincos.cpp:9


def _one_tol(dtype) -> float:
    """ONE_TOL is a float64 tolerance (1 + 1e-14 == 1 in float32). For lower
    precision we widen it to a few ulps so that rounding overshoot of |v| > 1
    does not turn into spurious failures. float64 keeps PROJ's exact value."""
    if dtype == torch.float64:
        return ONE_TOL
    return 1.0 + 8.0 * torch.finfo(dtype).eps


def aasin(v: torch.Tensor):
    """aasincos.cpp:11-21. Returns (asin value, failed mask). |v| >= 1 returns
    +-pi/2 (zero gradient); |v| > ONE_TOL additionally sets PROJ's errno, which
    makes the whole coordinate fail -> caller turns it into NaN."""
    av = v.abs()
    sat = av >= 1.0
    v_safe = torch.where(sat, torch.zeros_like(v), v)  # double where: no inf grad
    out = torch.where(sat, _signed_halfpi(v), torch.asin(v_safe))
    failed = av > _one_tol(v.dtype)
    return out, failed


def aatan2(n: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """aasincos.cpp:35-37: atan2 that returns 0 when both |n|,|d| < 1e-50.
    A Python scalar is compared in the tensor's dtype and 1e-50 is 0 in float32
    (|n| < 0 never holds), so lower precisions use the smallest normal number;
    this keeps aatan2(+-0, -0) = 0 (IEEE atan2 gives +-pi)."""
    atol = ATOL if n.dtype == torch.float64 else torch.finfo(n.dtype).tiny
    tiny = (n.abs() < atol) & (d.abs() < atol)
    n_safe = torch.where(tiny, torch.zeros_like(n), n)
    d_safe = torch.where(tiny, torch.ones_like(d), d)
    return torch.where(tiny, torch.zeros_like(n), torch.atan2(n_safe, d_safe))


def _signed_halfpi(v: torch.Tensor) -> torch.Tensor:
    # (v < 0 ? -HALFPI : HALFPI) in v's dtype. NB torch.where(c, float, float)
    # would produce a default-dtype (float32) tensor.
    return torch.where(v < 0, torch.full_like(v, -HALFPI), torch.full_like(v, HALFPI))


def _nan_where(mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, torch.full_like(x, float("nan")), x)


# ---------------------------------------------------------------------------
# Meridian distance: mlfn.cpp:9-31 with latitudes.cpp:238-420 (plain-float setup)
AUX_ORDER = 6  # AuxLat::ORDER
# latitudes.cpp:248-250  C[phi,mu] (rectifying -> geographic), even coeffs only
_C_PHI_MU = [3.0 / 2.0, -27.0 / 32.0, 269.0 / 512.0, 21.0 / 16.0, -55.0 / 32.0,
             6759.0 / 4096.0, 151.0 / 96.0, -417.0 / 128.0, 1097.0 / 512.0,
             -15543.0 / 2560.0, 8011.0 / 2560.0, 293393.0 / 61440.0]
# latitudes.cpp:278-280  C[mu,phi] (geographic -> rectifying), even coeffs only
_C_MU_PHI = [-3.0 / 2.0, 9.0 / 16.0, -3.0 / 32.0, 15.0 / 16.0, -15.0 / 32.0,
             135.0 / 2048.0, -35.0 / 48.0, 105.0 / 256.0, 315.0 / 512.0,
             -189.0 / 512.0, -693.0 / 1280.0, 1001.0 / 2048.0]


def pj_polyval(x: float, p, N: int) -> float:
    """latitudes.cpp:363-368 (Horner)."""
    y = 0.0 if N < 0 else p[N]
    while N > 0:
        N -= 1
        y = y * x + p[N]
    return y


def _auxlat_coeffs_even(n: float, coeffs) -> list:
    """latitudes.cpp:341-349 (the auxin,auxout <= RECTIFYING branch)."""
    F, o, d, n2 = [], 0, n, n * n
    for l in range(AUX_ORDER):
        m = (AUX_ORDER - l - 1) // 2
        F.append(d * pj_polyval(n2, coeffs[o:], m))
        o += m + 1
        d *= n
    return F


def pj_rectifying_radius(n: float) -> float:
    """latitudes.cpp:415-420."""
    return pj_polyval(n * n, [1.0, 1.0 / 4, 1.0 / 64, 1.0 / 256], 3) / (1 + n)


def pj_enfn(n: float) -> list:
    """mlfn.cpp:9-22: [rectifying radius] + F[mu<-phi] (6) + F[phi<-mu] (6)."""
    return ([pj_rectifying_radius(n)] + _auxlat_coeffs_even(n, _C_MU_PHI)
            + _auxlat_coeffs_even(n, _C_PHI_MU))


def pj_clenshaw(szeta, czeta, F) -> torch.Tensor:
    """latitudes.cpp:372-382; works on tensors or Python floats."""
    u0, u1 = 0.0, 0.0
    X = 2 * (czeta - szeta) * (czeta + szeta)
    for k in range(len(F) - 1, -1, -1):
        t = X * u0 - u1 + F[k]
        u1 = u0
        u0 = t
    return 2 * szeta * czeta * u0


def pj_mlfn(phi, sphi, cphi, en):
    """mlfn.cpp:24-26 + latitudes.cpp:395-398."""
    return en[0] * (phi + pj_clenshaw(sphi, cphi, en[1:1 + AUX_ORDER]))


def pj_inv_mlfn(mu, en):
    """mlfn.cpp:28-31 + latitudes.cpp:387-389."""
    zeta = mu / en[0]
    if isinstance(zeta, torch.Tensor):
        s, c = torch.sin(zeta), torch.cos(zeta)
    else:
        s, c = math.sin(zeta), math.cos(zeta)
    return zeta + pj_clenshaw(s, c, en[1 + AUX_ORDER:])


# ---------------------------------------------------------------------------
# Shared PJ_IO_UNITS helpers for projections whose output (right) side is
# angular (latlong: PJ_IO_UNITS_RADIANS) or untouched (ob_tran over latlong:
# PJ_IO_UNITS_WHATEVER). core.Projection only implements PJ_IO_UNITS_CLASSIC.
def _fwd_prepare(P: Projection, coo: torch.Tensor):
    """fwd.cpp:40-125 restricted to what core.Projection.forward does
    (latitude over-range check and clamp, lam -= lon_0, adjlon)."""
    lam, phi = coo[..., 0], coo[..., 1]
    bad = phi.abs() > HALFPI + PJ_EPS_LAT
    phi = phi.clamp(-HALFPI, HALFPI)
    lam = lam - P.lam0
    if not P.over:
        lam = adjlon(lam)
    return lam, phi, bad


def _inv_finalize(P: Projection, lam: torch.Tensor) -> torch.Tensor:
    """inv.cpp:108-117 (left side PJ_IO_UNITS_RADIANS)."""
    lam = lam + P.lam0
    if not P.over:
        lam = adjlon(lam)
    return lam


def _stack(a: torch.Tensor, b: torch.Tensor, coo: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.stack([a, b], -1), coo[..., 2:]], -1)


# ===========================================================================
# latlong.cpp
class LatLong(Projection):
    """+proj=latlong|longlat|latlon|lonlat (latlong.cpp:39-98).
    Both sides are PJ_IO_UNITS_RADIANS (latlong.cpp:87-88): no a-scaling, x_0/y_0
    forced to 0 (latlong.cpp:79-80). PROJ still subtracts lon_0 in fwd_prepare
    and adds it back in inv_finalize, and applies +lon_wrap in fwd_finalize
    (fwd.cpp:174-184)."""

    RIGHT_UNITS = "radians"

    def __init__(self, params):
        super().__init__(params)
        self.x0 = 0.0  # latlong.cpp:79-80
        self.y0 = 0.0
        self.long_wrap = None
        if "lon_wrap" in self.params:  # init.cpp:614-623
            c = float(self.params["lon_wrap"]) * DEG_TO_RAD
            if not abs(c) < 10 * M_TWOPI:
                raise ValueError("Invalid value for lon_wrap")
            self.long_wrap = c

    def _fwd(self, lam, phi):  # latlong.cpp:39-45
        return lam, phi

    def _inv(self, x, y):  # latlong.cpp:47-53
        return x, y

    def forward(self, coo):
        lam, phi, bad = _fwd_prepare(self, coo)
        x, y = self._fwd(lam, phi)
        if self.long_wrap is not None:  # fwd.cpp:177-182
            x = self.long_wrap + adjlon(x - self.long_wrap)
        return _stack(_nan_where(bad, x), _nan_where(bad, y), coo)

    def inverse(self, coo):
        lam, phi = self._inv(coo[..., 0], coo[..., 1])
        return _stack(_inv_finalize(self, lam), phi, coo)


# ===========================================================================
# eqc.cpp
class Eqc(_ProjShapeMixin, Projection):
    """+proj=eqc (eqc.cpp:41-145)."""

    def __init__(self, params):
        super().__init__(params)
        self._set_proj_shape()
        phi1 = float(self.params.get("lat_ts", 0.0)) * DEG_TO_RAD  # eqc.cpp:110
        cos_phi1 = math.cos(phi1)
        if cos_phi1 <= 0.0:  # eqc.cpp:113-117
            raise ValueError("Invalid value for lat_ts: |lat_ts| should be <= 90")
        if self.es != 0.0:  # eqc.cpp:119-133
            sin_phi1 = math.sin(phi1)
            nu1 = 1.0 / math.sqrt(1.0 - self.es * sin_phi1 * sin_phi1)
            self.rc = nu1 * cos_phi1
            self.en = pj_enfn(self.n)
            self.M0 = pj_mlfn(self.phi0, math.sin(self.phi0), math.cos(self.phi0), self.en)
        else:  # eqc.cpp:134-142
            self.rc = cos_phi1
            self.en = None
            self.M0 = 0.0

    def _fwd(self, lam, phi):
        x = self.rc * lam
        if self.en is None:  # eqc_s_forward, eqc.cpp:41-49
            y = phi - self.phi0
        else:  # eqc_e_forward, eqc.cpp:65-76
            y = pj_mlfn(phi, torch.sin(phi), torch.cos(phi), self.en) - self.M0
        return x, y

    def _inv(self, x, y):
        lam = x / self.rc
        if self.en is None:  # eqc_s_inverse, eqc.cpp:51-59
            phi = y + self.phi0
        else:  # eqc_e_inverse, eqc.cpp:81-89
            phi = pj_inv_mlfn(y + self.M0, self.en)
        return lam, phi


# ===========================================================================
# geos.cpp
class Geos(_ProjShapeMixin, Projection):
    """+proj=geos (geos.cpp:53-246).

    NaN semantics follow PROJ exactly:
      * ellipsoidal forward fails for points not visible from the satellite
        (geos.cpp:96-100);
      * the SPHERICAL forward has no visibility test at all (geos.cpp:65 is an
        empty "Check visibility" comment): far-side points get finite (wrong)
        coordinates in PROJ, and here too;
      * both inverses fail when the view ray misses the earth, det < 0
        (geos.cpp:134-138, 176-180)."""

    def __init__(self, params):
        super().__init__(params)
        self._set_proj_shape()
        self.h = float(self.params.get("h", 0.0))  # geos.cpp:206
        sweep = self.params.get("sweep", None)  # geos.cpp:208-224
        if sweep is None:
            self.flip_axis = False
        else:
            if sweep not in ("x", "y"):
                raise ValueError("Invalid value for sweep: it should be equal to x or y.")
            self.flip_axis = sweep == "x"
        self.radius_g_1 = self.h / self.a  # geos.cpp:226-230
        if self.radius_g_1 <= 0 or self.radius_g_1 > 1e10:
            raise ValueError("Invalid value for h.")
        self.radius_g = 1.0 + self.radius_g_1
        self.C = self.radius_g * self.radius_g - 1.0
        if self.es != 0.0:  # geos.cpp:233-243
            self.radius_p = math.sqrt(self.one_es)
            self.radius_p2 = self.one_es
            self.radius_p_inv2 = self.rone_es
            self.ellipsoidal = True
        else:
            self.radius_p = self.radius_p2 = self.radius_p_inv2 = 1.0
            self.ellipsoidal = False

    def _view_angles(self, Vx, Vy, Vz):
        """geos.cpp:68-76 / 103-111."""
        tmp = self.radius_g - Vx
        if self.flip_axis:
            x = self.radius_g_1 * torch.atan(Vy / torch.hypot(Vz, tmp))
            y = self.radius_g_1 * torch.atan(Vz / tmp)
        else:
            x = self.radius_g_1 * torch.atan(Vy / tmp)
            y = self.radius_g_1 * torch.atan(Vz / torch.hypot(Vy, tmp))
        return x, y

    def _fwd(self, lam, phi):
        if not self.ellipsoidal:  # geos_s_forward, geos.cpp:53-79
            tmp = torch.cos(phi)
            Vx = torch.cos(lam) * tmp
            Vy = torch.sin(lam) * tmp
            Vz = torch.sin(phi)
            return self._view_angles(Vx, Vy, Vz)
        # geos_e_forward, geos.cpp:81-114
        phi = torch.atan(self.radius_p2 * torch.tan(phi))  # geocentric latitude
        cphi, sphi = torch.cos(phi), torch.sin(phi)
        h = torch.hypot(self.radius_p * cphi, sphi)
        # NB: `float / tensor` is computed by torch as tensor.reciprocal() * float,
        # which is not a correctly rounded division (1 ulp off vs C): divide tensors.
        r = torch.full_like(h, self.radius_p) / h
        Vx = r * torch.cos(lam) * cphi
        Vy = r * torch.sin(lam) * cphi
        Vz = r * sphi
        invisible = ((self.radius_g - Vx) * Vx - Vy * Vy - Vz * Vz * self.radius_p_inv2) < 0.0
        # tmp = radius_g - Vx >= radius_g - 1 > 0, so the masked branch is finite:
        # no double where needed, the gradient of invisible points is exactly 0.
        x, y = self._view_angles(Vx, Vy, Vz)
        return _nan_where(invisible, x), _nan_where(invisible, y)

    def _inv(self, x, y):
        # geos.cpp:121-129 / 161-170
        Vx = -1.0
        if self.flip_axis:
            Vz = torch.tan(y / self.radius_g_1)
            if self.ellipsoidal:
                Vy = torch.tan(x / self.radius_g_1) * torch.hypot(torch.ones_like(Vz), Vz)
            else:
                Vy = torch.tan(x / self.radius_g_1) * torch.sqrt(1.0 + Vz * Vz)
        else:
            Vy = torch.tan(x / self.radius_g_1)
            if self.ellipsoidal:
                Vz = torch.tan(y / self.radius_g_1) * torch.hypot(torch.ones_like(Vy), Vy)
            else:
                Vz = torch.tan(y / self.radius_g_1) * torch.sqrt(1.0 + Vy * Vy)
        # geos.cpp:131-138 / 172-180
        if self.ellipsoidal:
            a = Vz / self.radius_p
            a = Vy * Vy + a * a + Vx * Vx
        else:
            a = Vy * Vy + Vz * Vz + Vx * Vx
        b = 2 * self.radius_g * Vx
        det = (b * b) - 4 * a * self.C
        miss = det < 0.0
        det_safe = torch.where(miss, torch.ones_like(det), det)  # double where
        # geos.cpp:140-153 / 182-195
        k = (-b - torch.sqrt(det_safe)) / (2.0 * a)
        Vx = self.radius_g + k * Vx
        Vy = Vy * k
        Vz = Vz * k
        lam = torch.atan2(Vy, Vx)
        phi = torch.atan(self.radius_p_inv2 * Vz / torch.hypot(Vx, Vy))
        return _nan_where(miss, lam), _nan_where(miss, phi)


# ===========================================================================
# moll.cpp (needed by the gie ob_tran test vectors)
class Moll(Projection):
    """+proj=moll (moll.cpp:22-82); always spherical (moll.cpp:61)."""

    MAX_ITER = 30     # moll.cpp:13
    LOOP_TOL = 1e-7   # moll.cpp:14

    def __init__(self, params, p=HALFPI):
        super().__init__(params)
        self.es = 0.0  # moll.cpp:61
        sp = math.sin(p)
        p2 = p + p
        r = math.sqrt(M_TWOPI * sp / (p2 + math.sin(p2)))
        self.C_x = 2.0 * r / M_PI
        self.C_y = r / sp
        self.C_p = p2 + math.sin(p2)

    def _fwd(self, lam, phi):  # moll.cpp:22-41
        k = self.C_p * torch.sin(phi)
        active = torch.ones_like(phi, dtype=torch.bool)
        for _ in range(self.MAX_ITER):  # masked loop, same cap as PROJ
            den = 1.0 + torch.cos(phi)
            den_safe = torch.where(active, den, torch.ones_like(den))  # double where
            num = torch.where(active, phi + torch.sin(phi) - k, torch.zeros_like(phi))
            V = num / den_safe
            phi = torch.where(active, phi - V, phi)
            active = active & ~(V.abs() < self.LOOP_TOL)
        # moll.cpp:34-37: not converged after MAX_ITER -> +-pi/2, else theta/2.
        # dtype policy: LOOP_TOL=1e-7 is below float32's ulp for theta in [1, pi)
        # (2.4e-7), so in reduced precision |V| stalls at rounding noise and the
        # snap would send points at |lat| > ~40 deg to the pole (1.5e7 m errors).
        # float64 keeps PROJ's rule; other dtypes keep the last iterate.
        if phi.dtype == torch.float64:
            phi = torch.where(active, _signed_halfpi(phi), 0.5 * phi)
        else:
            phi = 0.5 * phi
        return self.C_x * lam * torch.cos(phi), self.C_y * torch.sin(phi)

    def _inv(self, x, y):  # moll.cpp:43-55
        phi, f1 = aasin(y / self.C_y)
        # cos(phi) is never exactly 0 here (cos(+-HALFPI) = 6e-17 in float64,
        # -4e-8 in float32), so the division needs no guard.
        lam = x / (self.C_x * torch.cos(phi))
        ok = lam.abs() < M_PI
        phi = phi + phi
        phi, f2 = aasin((phi + torch.sin(phi)) / self.C_p)
        bad = ~ok | f1 | f2
        return _nan_where(bad, lam), _nan_where(bad, phi)


# ===========================================================================
# ob_tran.cpp
REGISTRY: Dict[str, type] = {
    "latlong": LatLong, "longlat": LatLong, "latlon": LatLong, "lonlat": LatLong,
    "eqc": Eqc, "geos": Geos, "moll": Moll,
}


def register_projection(name: str, cls: type) -> None:
    """Make any torch Projection usable as +o_proj=<name> (and in make())."""
    REGISTRY[name] = cls


def ob_tran_target_params(params: dict) -> Optional[dict]:
    """ob_tran.cpp:139-174: all params except proj=ob_tran / inv, with
    o_proj=xxx turned into proj=xxx. Returns None for o_proj=ob_tran."""
    out = {k: v for k, v in params.items() if k not in ("proj", "inv", "o_proj")}
    o_proj = params.get("o_proj")
    if o_proj is True or o_proj is None or o_proj == "ob_tran":
        return None
    out["proj"] = o_proj
    return out


class ObTran(Projection):
    """+proj=ob_tran (ob_tran.cpp:28-301), a composition rotation o link.

    `link` may be any core.Projection instance (it is called through its raw
    _fwd/_inv, exactly like Q->link->fwd / Q->link->inv in ob_tran.cpp:43/65);
    by default it is built from REGISTRY[o_proj] with the ob_tran parameters,
    like pj_create_argv_internal does (ob_tran.cpp:196-208).

    Output units (ob_tran.cpp:297-298): if the link's right side is radians
    (o_proj=latlong & aliases) ob_tran's right side is PJ_IO_UNITS_WHATEVER:
    forward returns RADIANS, unscaled, x_0/y_0 ignored; inverse takes radians.
    Otherwise output is classic (unit sphere * a + x_0)."""

    TOL = 1e-10  # ob_tran.cpp:26

    def __init__(self, params, link: Optional[Projection] = None):
        if isinstance(params, str):
            params = parse_proj_string(params)
        super().__init__(params)
        if link is None:
            if "o_proj" not in params or params["o_proj"] is True:  # ob_tran.cpp:190-193
                raise ValueError("Missing parameter: o_proj")
            lp = ob_tran_target_params(params)
            if lp is None:  # ob_tran.cpp:197-201
                raise ValueError("Failed to find projection to be rotated")
            if lp["proj"] not in REGISTRY:  # ob_tran.cpp:205-208
                raise ValueError("Projection to be rotated is unknown: %s" % lp["proj"])
            link = REGISTRY[lp["proj"]](lp)
        self.link = link

        P = params
        r = lambda k: float(P.get(k, 0.0)) * DEG_TO_RAD  # pj_param "r..." (degrees -> radians)
        if "o_alpha" in P:  # ob_tran.cpp:227-241
            lamc, phic, alpha = r("o_lon_c"), r("o_lat_c"), r("o_alpha")
            if abs(abs(phic) - HALFPI) <= self.TOL:
                raise ValueError("Invalid value for lat_c: |lat_c| should be < 90")
            n, d = -math.cos(alpha), -math.sin(alpha) * math.sin(phic)
            self.lamp = lamc + (0.0 if (abs(n) < ATOL and abs(d) < ATOL) else math.atan2(n, d))
            v = math.cos(phic) * math.sin(alpha)
            phip = math.copysign(HALFPI, v) if abs(v) >= 1.0 else math.asin(v)
        elif "o_lat_p" in P:  # ob_tran.cpp:242-245
            self.lamp = r("o_lon_p")
            phip = r("o_lat_p")
        else:  # ob_tran.cpp:246-282
            lam1, phi1, lam2, phi2 = r("o_lon_1"), r("o_lat_1"), r("o_lon_2"), r("o_lat_2")
            con = abs(phi1)
            if abs(phi1) > HALFPI - self.TOL:
                raise ValueError("Invalid value for lat_1: |lat_1| should be < 90")
            if abs(phi2) > HALFPI - self.TOL:
                raise ValueError("Invalid value for lat_2: |lat_2| should be < 90")
            if abs(phi1 - phi2) < self.TOL:
                raise ValueError("Invalid value for lat_1 and lat_2: lat_1 should be different from lat_2")
            if con < self.TOL:
                raise ValueError("Invalid value for lat_1: lat_1 should be different from zero")
            self.lamp = math.atan2(
                math.cos(phi1) * math.sin(phi2) * math.cos(lam1)
                - math.sin(phi1) * math.cos(phi2) * math.cos(lam2),
                math.sin(phi1) * math.cos(phi2) * math.sin(lam2)
                - math.cos(phi1) * math.sin(phi2) * math.sin(lam1))
            phip = math.atan(-math.cos(self.lamp - lam1) / math.tan(phi1))
        self.phip = phip
        self.oblique = abs(phip) > self.TOL  # ob_tran.cpp:284-292
        self.cphip = math.cos(phip) if self.oblique else 0.0
        self.sphip = math.sin(phip) if self.oblique else 0.0
        self.whatever_units = getattr(link, "RIGHT_UNITS", "classic") == "radians"  # :297

    # --- rotation (unit sphere), then the link's raw fwd/inv
    def _rotate_fwd(self, lam, phi):
        coslam, sinlam = torch.cos(lam), torch.sin(lam)
        sinphi, cosphi = torch.sin(phi), torch.cos(phi)
        if self.oblique:  # o_forward, ob_tran.cpp:28-44 (Snyder 5-8b, 5-7)
            lam = adjlon(aatan2(cosphi * sinlam, self.sphip * cosphi * coslam + self.cphip * sinphi)
                         + self.lamp)
            phi, failed = aasin(self.sphip * sinphi - self.cphip * cosphi * coslam)
        else:  # t_forward, ob_tran.cpp:46-57
            lam = adjlon(aatan2(cosphi * sinlam, sinphi) + self.lamp)
            phi, failed = aasin(-cosphi * coslam)
        return lam, phi, failed

    def _rotate_inv(self, lam, phi):
        if self.oblique:  # o_inverse, ob_tran.cpp:59-78 (Snyder 5-9, 5-10b)
            lam = lam - self.lamp
            coslam, sinphi, cosphi = torch.cos(lam), torch.sin(phi), torch.cos(phi)
            phi_out, failed = aasin(self.sphip * sinphi + self.cphip * cosphi * coslam)
            lam_out = aatan2(cosphi * torch.sin(lam), self.sphip * cosphi * coslam - self.cphip * sinphi)
        else:  # t_inverse, ob_tran.cpp:80-94
            cosphi = torch.cos(phi)
            t = lam - self.lamp
            lam_out = aatan2(cosphi * torch.sin(t), -torch.sin(phi))
            phi_out, failed = aasin(cosphi * torch.cos(t))
        return lam_out, phi_out, failed

    def _fwd(self, lam, phi):
        lam, phi, failed = self._rotate_fwd(lam, phi)
        x, y = self.link._fwd(lam, phi)
        return _nan_where(failed, x), _nan_where(failed, y)

    def _inv(self, x, y):
        lam, phi = self.link._inv(x, y)
        lam, phi, failed = self._rotate_inv(lam, phi)
        return _nan_where(failed, lam), _nan_where(failed, phi)

    def forward(self, coo):
        if not self.whatever_units:
            return super().forward(coo)
        # PJ_IO_UNITS_WHATEVER output: fwd_finalize does nothing (fwd.cpp:168-169)
        lam, phi, bad = _fwd_prepare(self, coo)
        x, y = self._fwd(lam, phi)
        return _stack(_nan_where(bad, x), _nan_where(bad, y), coo)

    def inverse(self, coo):
        if not self.whatever_units:
            return super().inverse(coo)
        # PJ_IO_UNITS_WHATEVER input: inv_prepare does nothing (inv.cpp:59-60)
        lam, phi = self._inv(coo[..., 0], coo[..., 1])
        return _stack(_inv_finalize(self, lam), phi, coo)


# ===========================================================================
def make(definition: Union[str, dict]) -> Projection:
    """Build a torch projection from a single-operation PROJ string."""
    params = parse_proj_string(definition) if isinstance(definition, str) else dict(definition)
    name = params.get("proj")
    if name == "ob_tran":
        return ObTran(params)
    if name not in REGISTRY:
        raise ValueError("unsupported projection %r" % name)
    return REGISTRY[name](params)


def angular_output(P: Projection, inverse: bool) -> bool:
    """proj_angular_output(): inverse output is always radians; forward output
    is radians only for latlong (ob_tran over latlong is 'whatever')."""
    return True if inverse else getattr(P, "RIGHT_UNITS", "classic") == "radians"


def angular_input_inverse(P: Projection) -> bool:
    """proj_angular_input(P, PJ_INV): right side is PJ_IO_UNITS_RADIANS."""
    return getattr(P, "RIGHT_UNITS", "classic") == "radians"
