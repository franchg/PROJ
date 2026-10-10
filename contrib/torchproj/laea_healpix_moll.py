"""Lambert Azimuthal Equal Area, HEALPix / rHEALPix and Mollweide (+ Wagner IV/V)
ported from PROJ (9.8.x, this PROJ tree) to PyTorch.

Sources ported (file:line refer to this PROJ tree):
  * src/projections/laea.cpp      laea (N_POLE, S_POLE, EQUIT, OBLIQ; ellipsoid + sphere)
  * src/projections/healpix.cpp   healpix (+rot_xy) and rhealpix (+north_square/south_square)
  * src/projections/moll.cpp      moll, wag4, wag5 (same code, different constants)
  * src/latitudes.cpp             authalic latitude helpers (pj_authalic_lat_q,
                                  pj_authalic_lat_compute_coeffs, pj_authalic_lat,
                                  pj_authalic_lat_inverse, pj_auxlat_coeffs [phi<->xi
                                  matrices only], pj_polyval, pj_clenshaw, pj_auxlat_convert)
  * src/aasincos.cpp              aasin
  * src/ell_set.cpp:582,611       third flattening n = tan(asin(e)/2)^2
  * src/trans.cpp:352             NaN in -> all-NaN out

Conventions (see core.py): `_fwd(lam, phi)` / `_inv(x, y)` work on the unit
ellipsoid with lam relative to lon_0; failed points (PROJ: HUGE_VAL + errno) are
NaN. All PROJ `if`s on per-point values are torch.where; the inputs of an untaken
(or failing) branch are replaced by harmless values *before* the arithmetic
("double where") so NaN/inf of the untaken branch cannot leak into gradients.
Data-dependent loops (Mollweide Newton, authalic Newton for |n| >= 0.01) are
masked loops with PROJ's iteration cap; there is no .item()/Python `if` on tensor
values in forward(), so the code is GPU / torch.compile(fullgraph=True) friendly.

Mollweide offers three solvers for the auxiliary angle (see Mollweide docstring):
  method="masked"   faithful (default): 30 Newton steps, a point freezes once
                    |V| < 1e-7 exactly like the C `break`; gradients are obtained by
                    back-propagating through the unrolled iterations.
  method="fixed"    N unconditional Newton steps (no freezing), same convergence
                    test on the last step; N = n_iter (default 30).
  method="implicit" value from the masked loop under no_grad, gradient from the
                    implicit function theorem via a custom autograd.Function
                    (O(1) memory, exact derivative of the converged root).
"""
import math

import torch

from core import HALFPI, Projection, parse_proj_string

# ---------------------------------------------------------------------------
# constants
EPS10 = 1.0e-10                  # laea.cpp:27
FORTPI = math.pi / 4             # proj_internal.h:160 (M_FORTPI = M_PI_4)
HEALPIX_EPS = 1e-15              # healpix.cpp:66
ONE_TOL = 1.00000000000001       # aasincos.cpp:8
MOLL_MAX_ITER = 30               # moll.cpp:13
MOLL_LOOP_TOL = 1e-7             # moll.cpp:14

# AuxLat enum (proj_internal.h:456-466)
GEOGRAPHIC, PARAMETRIC, GEOCENTRIC, RECTIFYING, CONFORMAL, AUTHALIC = range(6)
AUX_ORDER = 6

# Only the two matrices needed for authalic latitudes, keyed (auxout, auxin)
# like C[eta, zeta] in latitudes.cpp.
_AUX_COEFFS = {
    # C[phi,xi]  (latitudes.cpp:257-264): authalic -> geographic
    (GEOGRAPHIC, AUTHALIC): [
        4.0 / 3.0, 4.0 / 45.0, -16.0 / 35.0, -2582.0 / 14175.0,
        60136.0 / 467775.0, 28112932.0 / 212837625.0, 46.0 / 45.0,
        152.0 / 945.0, -11966.0 / 14175.0, -21016.0 / 51975.0,
        251310128.0 / 638512875.0, 3044.0 / 2835.0, 3802.0 / 14175.0,
        -94388.0 / 66825.0, -8797648.0 / 10945935.0, 6059.0 / 4725.0,
        41072.0 / 93555.0, -1472637812.0 / 638512875.0, 768272.0 / 467775.0,
        455935736.0 / 638512875.0, 4210684958.0 / 1915538625.0],
    # C[xi,phi]  (latitudes.cpp:309-316): geographic -> authalic
    (AUTHALIC, GEOGRAPHIC): [
        -4.0 / 3.0, -4.0 / 45.0, 88.0 / 315.0, 538.0 / 4725.0,
        20824.0 / 467775.0, -44732.0 / 2837835.0, 34.0 / 45.0, 8.0 / 105.0,
        -2482.0 / 14175.0, -37192.0 / 467775.0, -12467764.0 / 212837625.0,
        -1532.0 / 2835.0, -898.0 / 14175.0, 54968.0 / 467775.0,
        100320856.0 / 1915538625.0, 6007.0 / 14175.0, 24496.0 / 467775.0,
        -5884124.0 / 70945875.0, -23356.0 / 66825.0, -839792.0 / 19348875.0,
        570284222.0 / 1915538625.0],
}

_DATUM_ELLPS = {"WGS84": "WGS84", "NAD83": "GRS80", "NAD27": "clrk66"}


# ---------------------------------------------------------------------------
# setup-time (Python float) helpers
def polyval(x, p, N):
    """pj_polyval (latitudes.cpp:363-368): sum p[i] x^i, i=0..N, Horner."""
    y = 0.0 if N < 0 else p[N]
    while N > 0:
        N -= 1
        y = y * x + p[N]
    return y


def auxlat_coeffs(n, auxin, auxout):
    """pj_auxlat_coeffs (latitudes.cpp:238-359), restricted to phi<->xi."""
    coeffs = _AUX_COEFFS[(auxout, auxin)]
    F = []
    o = 0
    d = n
    if auxin <= RECTIFYING and auxout <= RECTIFYING:  # latitudes.cpp:342-349
        n2 = n * n
        for l in range(AUX_ORDER):
            m = (AUX_ORDER - l - 1) // 2
            F.append(d * polyval(n2, coeffs[o:], m))
            o += m + 1
            d *= n
    else:  # latitudes.cpp:350-356
        for l in range(AUX_ORDER):
            m = AUX_ORDER - l - 1
            F.append(d * polyval(n, coeffs[o:], m))
            o += m + 1
            d *= n
    return F


# ---------------------------------------------------------------------------
# tensor helpers
def _sel(mask, a, b, like):
    """torch.where(mask, a, b) for two Python constants, in like's dtype/device.
    (torch.where(mask, 1.0, 2.0) would silently produce the *default* dtype,
    float32, and round constants like pi/2 -> caught by the rhealpix tests.)"""
    return torch.where(mask, torch.full_like(like, a), torch.full_like(like, b))


def _hypot0(x, y):
    """hypot with a zero (not NaN) gradient at the origin; same values."""
    z = (x == 0) & (y == 0)
    return torch.where(z, torch.zeros_like(x), torch.hypot(torch.where(z, 1.0, x), y))


def _atan2_0(y, x):
    """atan2(y, x) with zero gradient (not NaN) at (0, 0). The (0, 0) value keeps
    C's signed-zero semantics (atan2(+0, -0) = pi)."""
    z = (x == 0) & (y == 0)
    v0 = torch.atan2(y.detach(), x.detach())
    return torch.where(z, v0, torch.atan2(y, torch.where(z, 1.0, x)))


def _asin(v):
    """C asin() values (NaN for |v| > 1, +-pi/2 at |v| == 1) but with a zero
    gradient for |v| >= 1 instead of inf/NaN: |v| > 1 is a failed point and
    |v| == 1 a non-differentiable point (pole / domain edge). Double where."""
    edge = v.abs() >= 1.0
    inner = torch.asin(torch.where(edge, 0.0, v))
    out = torch.where(v.abs() > 1.0, float("nan"), _sel(v < 0.0, -HALFPI, HALFPI, v))
    return torch.where(edge, out, inner)


def clenshaw(szeta, czeta, F):
    """pj_clenshaw (latitudes.cpp:372-382): sum F[k] sin((2k+2) zeta)."""
    u0 = torch.zeros_like(szeta)
    u1 = torch.zeros_like(szeta)
    X = 2 * (czeta - szeta) * (czeta + szeta)
    for k in range(len(F) - 1, -1, -1):  # fixed trip count (K = 6)
        t = X * u0 - u1 + F[k]
        u1 = u0
        u0 = t
    return 2 * szeta * czeta * u0


def auxlat_convert(zeta, F):
    """pj_auxlat_convert (latitudes.cpp:387-398)."""
    return zeta + clenshaw(torch.sin(zeta), torch.cos(zeta), F)


def aasin(v):
    """aasin (aasincos.cpp:11-21). Returns (value, error_mask); |v| > ONE_TOL is an
    error (errno set -> PROJ turns the point into HUGE_VAL, inv.cpp:144-149)."""
    av = v.abs()
    big = av >= 1.0
    err = av > ONE_TOL
    inner = torch.asin(torch.where(big, 0.0, v))  # double where: asin' inf at |v|=1
    edge = _sel(v < 0.0, -HALFPI, HALFPI, v)
    return torch.where(big, edge, inner), err


class Authalic:
    """Authalic latitude machinery of latitudes.cpp:52-155 for one ellipsoid.
    Setup is pure Python floats; lat()/lat_inv() are tensor ops."""

    def __init__(self, e, es, one_es):
        self.e, self.es, self.one_es = e, es, one_es
        alpha = math.asin(e)                   # ell_set.cpp:582
        self.n = math.tan(alpha / 2) ** 2      # ell_set.cpp:611
        self.series = abs(self.n) < 0.01       # latitudes.cpp:81 PROJ_AUTHALIC_SERIES_VALID
        # pj_authalic_lat_compute_coeffs (latitudes.cpp:82-93):
        #   APA[0:6]  = xi -> phi;  APA[6:12] = phi -> xi (only if series valid)
        self.apa_inv = auxlat_coeffs(self.n, AUTHALIC, GEOGRAPHIC)
        self.apa_fwd = auxlat_coeffs(self.n, GEOGRAPHIC, AUTHALIC) if self.series else None
        self.qp = self._q_float(1.0)

    def _q_float(self, sinphi):
        """pj_authalic_lat_q (latitudes.cpp:55-71) for a Python float."""
        if self.e >= 1e-7:
            e_sinphi = self.e * sinphi
            om = 1.0 - e_sinphi * e_sinphi
            if om == 0.0:
                return math.inf
            return self.one_es * (sinphi / om + math.atanh(e_sinphi) / self.e)
        return 2 * sinphi

    def q(self, sinphi):
        """pj_authalic_lat_q (latitudes.cpp:55-71), tensor version."""
        if self.e >= 1e-7:
            e_sinphi = self.e * sinphi
            om = 1.0 - e_sinphi * e_sinphi
            zero = om == 0.0  # cannot happen for e < 1; kept for fidelity
            r = self.one_es * (sinphi / torch.where(zero, 1.0, om)
                               + torch.atanh(e_sinphi) / self.e)
            return torch.where(zero, math.inf, r)
        return 2 * sinphi

    def lat(self, phi, sinphi, cosphi):
        """pj_authalic_lat (latitudes.cpp:99-116): geographic -> authalic."""
        if self.series:
            return phi + clenshaw(sinphi, cosphi, self.apa_fwd)
        ratio = self.q(sinphi) / self.qp
        # "Rounding error" clamp (|ratio| > 1 -> +-1). asin'(+-1) = inf would give an
        # inf/NaN gradient at the poles; _asin gives 0 there (true value is finite).
        ratio = torch.where(ratio.abs() > 1, _sel(ratio > 0, 1.0, -1.0, ratio), ratio)
        return _asin(ratio)

    def lat_inv(self, beta):
        """pj_authalic_lat_inverse (latitudes.cpp:123-154): authalic -> geographic.
        Series for |n| < 0.01, else series start + Newton (masked loop, cap 10,
        break when !(|dphi| >= 1e-15), exactly like the C loop)."""
        phi = auxlat_convert(beta, self.apa_inv)
        if self.series:
            return phi
        q = torch.sin(beta) * self.qp
        q_div_one_minus_es = q / self.one_es
        active = torch.ones_like(phi, dtype=torch.bool)
        for _ in range(10):
            sinphi = torch.sin(phi)
            cosphi = torch.cos(phi)
            om = 1 - self.es * (sinphi * sinphi)
            dphi = ((om * om) / (2 * cosphi)
                    * (q_div_one_minus_es - sinphi / om - torch.atanh(self.e * sinphi) / self.e))
            upd = active & (dphi.abs() >= 1e-15)
            phi = torch.where(upd, phi + dphi, phi)
            active = upd
        return phi


# ---------------------------------------------------------------------------
class _Base(Projection):
    """Adds +datum=<ellipsoid datum> and PROJ's NaN-in/NaN-out rule (trans.cpp:352)."""

    def __init__(self, params):
        if isinstance(params, str):
            params = parse_proj_string(params)
        params = dict(params)
        if "datum" in params and not any(k in params for k in ("ellps", "a", "R", "b", "rf", "es", "e", "f")):
            params["ellps"] = _DATUM_ELLPS[params["datum"]]
        super().__init__(params)
        self.a_ellps = self.a  # semi-major axis before any projection-specific change

    def _fwd(self, lam, phi):
        x, y = self._fwd_impl(lam, phi)
        nan_in = torch.isnan(lam) | torch.isnan(phi)
        return torch.where(nan_in, float("nan"), x), torch.where(nan_in, float("nan"), y)

    def _inv(self, x, y):
        lam, phi = self._inv_impl(x, y)
        nan_in = torch.isnan(x) | torch.isnan(y)
        return torch.where(nan_in, float("nan"), lam), torch.where(nan_in, float("nan"), phi)


# ===========================================================================
# Lambert Azimuthal Equal Area (laea.cpp)
# ===========================================================================
N_POLE, S_POLE, EQUIT, OBLIQ = 0, 1, 2, 3  # laea.cpp:9


class LambertAzimuthalEqualArea(_Base):
    """+proj=laea. Setup = laea.cpp:243-309 (mode chosen from lat_0 at setup)."""

    def __init__(self, params):
        super().__init__(params)
        t = abs(self.phi0)                                       # laea.cpp:252-263
        if t > HALFPI + EPS10:
            raise ValueError("Invalid value for lat_0: |lat_0| should be <= 90deg")
        if abs(t - HALFPI) < EPS10:
            self.mode = S_POLE if self.phi0 < 0.0 else N_POLE
        elif abs(t) < EPS10:
            self.mode = EQUIT
        else:
            self.mode = OBLIQ
        self.sinb1 = self.cosb1 = self.xmf = self.ymf = self.dd = self.rq = 0.0
        self.ellipsoidal = self.es != 0.0
        if self.ellipsoidal:                                     # laea.cpp:264-298
            self.auth = Authalic(self.e, self.es, self.one_es)
            self.qp = self.auth.qp
            self.mmf = 0.5 / (1.0 - self.es)
            if self.mode in (N_POLE, S_POLE):
                self.dd = 1.0
            elif self.mode == EQUIT:
                self.rq = math.sqrt(0.5 * self.qp)
                self.dd = 1.0 / self.rq
                self.xmf = 1.0
                self.ymf = 0.5 * self.qp
            else:
                self.rq = math.sqrt(0.5 * self.qp)
                sinphi = math.sin(self.phi0)
                cosphi = math.cos(self.phi0)
                p0 = torch.tensor(self.phi0, dtype=torch.float64)
                b1 = float(self.auth.lat(p0, torch.sin(p0), torch.cos(p0)))
                self.sinb1 = math.sin(b1)
                self.cosb1 = math.cos(b1)
                self.dd = math.cos(self.phi0) / (
                    math.sqrt(1.0 - self.es * sinphi * sinphi) * self.rq * self.cosb1)
                self.xmf = self.rq
                self.ymf = self.xmf / self.dd
                self.xmf *= self.dd
        elif self.mode == OBLIQ:                                 # laea.cpp:300-303
            self.sinb1 = math.sin(self.phi0)
            self.cosb1 = math.cos(self.phi0)

    def _fwd_impl(self, lam, phi):
        return self._e_fwd(lam, phi) if self.ellipsoidal else self._s_fwd(lam, phi)

    def _inv_impl(self, x, y):
        return self._e_inv(x, y) if self.ellipsoidal else self._s_inv(x, y)

    # -- laea.cpp:29-90 ----------------------------------------------------
    def _e_fwd(self, lam, phi):
        mode = self.mode
        coslam = torch.cos(lam)
        sinlam = torch.sin(lam)
        sinphi = torch.sin(phi)
        cosphi = torch.cos(phi)
        xi = self.auth.lat(phi, sinphi, cosphi)
        q = torch.sin(xi) * self.qp
        if mode in (OBLIQ, EQUIT):
            sinb = torch.sin(xi)
            cosb = torch.cos(xi)
        if mode == OBLIQ:
            b = 1.0 + self.sinb1 * sinb + self.cosb1 * cosb * coslam
        elif mode == EQUIT:
            b = 1.0 + cosb * coslam
        elif mode == N_POLE:
            b = HALFPI + phi
            q = self.qp - q
        else:
            b = phi - HALFPI
            q = self.qp + q
        bad = b.abs() < EPS10  # -> PROJ_ERR_COORD_TRANSFM_OUTSIDE_PROJECTION_DOMAIN
        if mode == OBLIQ:
            bb = torch.sqrt(2.0 / torch.where(bad, 1.0, b))
            y = self.ymf * bb * (self.cosb1 * sinb - self.sinb1 * cosb * coslam)
            x = self.xmf * bb * cosb * sinlam
        elif mode == EQUIT:
            den = 1.0 + cosb * coslam
            bb = torch.sqrt(2.0 / torch.where(bad, 1.0, den))
            y = bb * sinb * self.ymf
            x = self.xmf * bb * cosb * sinlam
        else:
            ok = q >= 1e-15
            bb = torch.sqrt(torch.where(ok, q, 1.0))   # sqrt' = inf at q = 0
            x = torch.where(ok, bb * sinlam, 0.0)
            y = torch.where(ok, coslam * (bb if mode == S_POLE else -bb), 0.0)
        return torch.where(bad, float("nan"), x), torch.where(bad, float("nan"), y)

    # -- laea.cpp:92-132 ---------------------------------------------------
    def _s_fwd(self, lam, phi):
        mode = self.mode
        sinphi = torch.sin(phi)
        cosphi = torch.cos(phi)
        coslam = torch.cos(lam)
        if mode in (EQUIT, OBLIQ):
            if mode == EQUIT:
                yy = 1.0 + cosphi * coslam
            else:
                yy = 1.0 + self.sinb1 * sinphi + self.cosb1 * cosphi * coslam
            bad = yy <= EPS10
            yy = torch.sqrt(2.0 / torch.where(bad, 1.0, yy))
            x = yy * cosphi * torch.sin(lam)
            if mode == EQUIT:
                y = yy * sinphi
            else:
                y = yy * (self.cosb1 * sinphi - self.sinb1 * cosphi * coslam)
        else:
            if mode == N_POLE:
                coslam = -coslam
            bad = (phi + self.phi0).abs() < EPS10
            t = FORTPI - phi * 0.5
            t = 2.0 * (torch.cos(t) if mode == S_POLE else torch.sin(t))
            x = t * torch.sin(lam)
            y = t * coslam
        return torch.where(bad, float("nan"), x), torch.where(bad, float("nan"), y)

    # -- laea.cpp:134-186 --------------------------------------------------
    def _e_inv(self, x, y):
        mode = self.mode
        if mode in (EQUIT, OBLIQ):
            x = x / self.dd
            y = y * self.dd
            rho0 = torch.hypot(x, y)
            centre = rho0 < EPS10                       # -> (0, phi0)
            xs = torch.where(centre, 1.0, x)            # double where (hypot/atan2/1/rho)
            ys = torch.where(centre, 1.0, y)
            rho = torch.hypot(xs, ys)
            arg = 0.5 * rho / self.rq
            bad = (~centre) & (arg > 1)                 # outside projection domain
            sCe = 2.0 * _asin(torch.where(bad, 0.5, arg))
            cCe = torch.cos(sCe)
            sCe = torch.sin(sCe)
            xs = xs * sCe
            if mode == OBLIQ:
                ab = cCe * self.sinb1 + ys * sCe * self.cosb1 / rho
                ys = rho * self.cosb1 * cCe - ys * self.sinb1 * sCe
            else:
                ab = ys * sCe / rho
                ys = rho * cCe
        else:
            if mode == N_POLE:
                y = -y
            q = x * x + y * y
            centre = q == 0.0                           # -> (0, phi0)
            bad = torch.zeros_like(centre)
            xs = torch.where(centre, 1.0, x)
            ys = torch.where(centre, 1.0, y)
            ab = 1.0 - q / self.qp
            if mode == S_POLE:
                ab = -ab
        lam = _atan2_0(xs, ys)
        # asin(ab) is NaN for |ab| > 1 (rounding / outside), as in C
        phi = self.auth.lat_inv(_asin(torch.where(centre, 0.0, ab)))
        lam = torch.where(centre, 0.0, lam)
        phi = torch.where(centre, self.phi0, phi)
        return torch.where(bad, float("nan"), lam), torch.where(bad, float("nan"), phi)

    # -- laea.cpp:188-229 --------------------------------------------------
    def _s_inv(self, x, y):
        mode = self.mode
        rh = _hypot0(x, y)
        half = rh * 0.5
        bad = half > 1.0
        phi = 2.0 * _asin(torch.where(bad, 0.0, half))
        if mode in (OBLIQ, EQUIT):
            sinz = torch.sin(phi)
            cosz = torch.cos(phi)
            small = rh.abs() <= EPS10
            rh_s = torch.where(small, 1.0, rh)
            if mode == EQUIT:
                phi = torch.where(small, 0.0, _asin(y * sinz / rh_s))
                x = x * sinz
                y = cosz * rh
            else:
                phi = torch.where(small, self.phi0,
                                  _asin(cosz * self.sinb1 + y * sinz * self.cosb1 / rh_s))
                x = x * (sinz * self.cosb1)
                y = (cosz - torch.sin(phi) * self.sinb1) * rh
            yzero = y == 0.0
            lam = torch.where(yzero, 0.0, _atan2_0(x, torch.where(yzero, 1.0, y)))
        else:
            if mode == N_POLE:
                y = -y
                phi = HALFPI - phi
            else:
                phi = phi - HALFPI
            lam = _atan2_0(x, y)
        return torch.where(bad, float("nan"), lam), torch.where(bad, float("nan"), phi)


# ===========================================================================
# HEALPix / rHEALPix (healpix.cpp)
# ===========================================================================
def _healpix_verts():
    """healpixVertsJit, healpix.cpp:177-195."""
    E = HEALPIX_EPS
    return [(-math.pi - E, FORTPI), (-3 * FORTPI, HALFPI + E), (-HALFPI, FORTPI + E),
            (-FORTPI, HALFPI + E), (0.0, FORTPI + E), (FORTPI, HALFPI + E),
            (HALFPI, FORTPI + E), (3 * FORTPI, HALFPI + E), (math.pi + E, FORTPI),
            (math.pi + E, -FORTPI), (3 * FORTPI, -HALFPI - E), (HALFPI, -FORTPI - E),
            (FORTPI, -HALFPI - E), (0.0, -FORTPI - E), (-FORTPI, -HALFPI - E),
            (-HALFPI, -FORTPI - E), (-3 * FORTPI, -HALFPI - E), (-math.pi - E, -FORTPI),
            (-math.pi - E, FORTPI)]


def _rhealpix_verts(ns, ss):
    """rhealpixVertsJit, healpix.cpp:204-217."""
    E = HEALPIX_EPS
    return [(-math.pi - E, FORTPI + E),
            (-math.pi + ns * HALFPI - E, FORTPI + E),
            (-math.pi + ns * HALFPI - E, 3 * FORTPI + E),
            (-math.pi + (ns + 1.0) * HALFPI + E, 3 * FORTPI + E),
            (-math.pi + (ns + 1.0) * HALFPI + E, FORTPI + E),
            (math.pi + E, FORTPI + E),
            (math.pi + E, -FORTPI - E),
            (-math.pi + (ss + 1.0) * HALFPI + E, -FORTPI - E),
            (-math.pi + (ss + 1.0) * HALFPI + E, -3 * FORTPI - E),
            (-math.pi + ss * HALFPI - E, -3 * FORTPI - E),
            (-math.pi + ss * HALFPI - E, -FORTPI - E),
            (-math.pi - E, -FORTPI - E)]


def pnpoly(vert, tx, ty):
    """pnpoly (healpix.cpp:132-165), vectorised over points; the vertex list is a
    setup-time constant so the Python loop is a fixed unrolled sequence.
    Faithful detail: the C loop runs i = 1..nvert-1 and never closes the polygon
    (last -> first edge). For HEALPix first == last vertex so this is harmless; for
    rHEALPix the left edge x = -pi-EPS is missing (see report)."""
    hit = torch.zeros_like(tx, dtype=torch.bool)
    for vx, vy in vert:                                   # boundary vertices
        hit = hit | ((tx == vx) & (ty == vy))
    counter = torch.zeros_like(tx, dtype=torch.int32)
    p1x, p1y = vert[0]
    for i in range(1, len(vert)):
        p2x, p2y = vert[i % len(vert)]
        if p1y != p2y:                                    # constant part of the test
            c = (ty > min(p1y, p2y)) & (ty <= max(p1y, p2y)) & (tx <= max(p1x, p2x))
            if p1x != p2x:
                xinters = (ty - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                c = c & (tx <= xinters)
            counter = counter + c.to(torch.int32)
        p1x, p1y = p2x, p2y
    return hit | (counter % 2 == 1)


def healpix_sphere(lam, phi):
    """healpix_sphere (healpix.cpp:246-268)."""
    phi0 = math.asin(2.0 / 3.0)
    eq = phi.abs() <= phi0
    x_eq = lam
    y_eq = 3 * math.pi / 8 * torch.sin(phi)
    u = 3 * (1 - torch.sin(phi).abs())
    pos = u > 0
    sigma = torch.where(pos, torch.sqrt(torch.where(pos, u, 1.0)), 0.0)  # sqrt' inf at pole
    cn = torch.floor(2 * lam / math.pi + 2)
    cn = torch.where(cn >= 4, 3.0, cn)
    lamc = -3 * FORTPI + HALFPI * cn
    x_p = lamc + (lam - lamc) * sigma
    y_p = torch.sign(phi) * FORTPI * (2 - sigma)
    return torch.where(eq, x_eq, x_p), torch.where(eq, y_eq, y_p)


def healpix_sphere_inverse(x, y):
    """healpix_spherhealpix_e_inverse (healpix.cpp:273-298)."""
    ay = y.abs()
    eq = ay <= FORTPI
    mid = (~eq) & (ay < HALFPI)
    lam_e = x
    phi_e = torch.asin(8 * torch.where(eq, y, 0.0) / (3 * math.pi))
    cn = torch.floor(2 * x / math.pi + 2)
    cn = torch.where(cn >= 4, 3.0, cn)
    xc = -3 * FORTPI + HALFPI * cn
    tau = torch.where(mid, 2.0 - 4 * ay / math.pi, 1.0)
    lam_m = xc + (x - xc) / tau
    phi_m = torch.sign(y) * _asin(1.0 - tau ** 2 / 3.0)
    lam_o = torch.full_like(x, -math.pi)
    phi_o = torch.sign(y) * HALFPI
    lam = torch.where(eq, lam_e, torch.where(mid, lam_m, lam_o))
    phi = torch.where(eq, phi_e, torch.where(mid, phi_m, phi_o))
    return lam, phi


def _rot_quarter(k, vx, vy):
    """ROT[get_rotate_index(k)] . (vx, vy) (healpix.cpp:42-64,105-123,327-336):
    counter-clockwise rotation by k*pi/2, k integer-valued float tensor in -3..3.
    Entries are 0/+-1 so the result is exact, like the C dot_product."""
    k4 = torch.remainder(k, 4)
    c = torch.where(k4 == 0, 1.0, _sel(k4 == 2, -1.0, 0.0, vx))
    s = torch.where(k4 == 1, 1.0, _sel(k4 == 3, -1.0, 0.0, vx))
    return c * vx - s * vy, s * vx + c * vy


def combine_caps_fwd(x, y, ns, ss):
    """combine_caps(..., inverse=0) with get_cap(..., inverse=0)
    (healpix.cpp:346-382, 435-491)."""
    north = y > FORTPI
    south = y < -FORTPI
    c_y = _sel(north, HALFPI, -HALFPI, x)
    cn = torch.where(x < -HALFPI, 0.0, torch.where(x < 0, 1.0, _sel(x < HALFPI, 2.0, 3.0, x)))
    c_x = torch.where(cn == 0, -3 * FORTPI, torch.where(cn == 1, -FORTPI,
                      _sel(cn == 2, FORTPI, 3 * FORTPI, x)))
    pole = _sel(north, float(ns), float(ss), x)
    k = torch.where(north, cn - pole, -1 * (cn - pole))
    rx, ry = _rot_quarter(k, x - c_x, y - c_y)
    ax = -3 * FORTPI + pole * HALFPI
    ay = _sel(north, 1.0, -1.0, x) * HALFPI
    polar = north | south
    return torch.where(polar, rx + ax, x), torch.where(polar, ry + ay, y)


def combine_caps_inv(x, y, ns, ss):
    """combine_caps(..., inverse=1) with get_cap(..., inverse=1)
    (healpix.cpp:383-424, 435-491)."""
    E = HEALPIX_EPS
    north = y > FORTPI
    south = y < -FORTPI
    c_x = _sel(north, -3 * FORTPI + ns * HALFPI, -3 * FORTPI + ss * HALFPI, x)
    c_y = _sel(north, HALFPI, -HALFPI, x)
    xx = torch.where(north, x - ns * HALFPI, x - ss * HALFPI)
    # north polar square (healpix.cpp:401-410)
    n1 = (y >= -xx - FORTPI - E) & (y < xx + 5 * FORTPI - E)
    n2 = (y > -xx - FORTPI + E) & (y >= xx + 5 * FORTPI - E)
    n3 = (y <= -xx - FORTPI + E) & (y > xx + 5 * FORTPI + E)
    cn_n = torch.where(n1, float((ns + 1) % 4), torch.where(
        n2, float((ns + 2) % 4), _sel(n3, float((ns + 3) % 4), float(ns), x)))
    # south polar square (healpix.cpp:411-420)
    s1 = (y <= xx + FORTPI + E) & (y > -xx - 5 * FORTPI + E)
    s2 = (y < xx + FORTPI - E) & (y <= -xx - 5 * FORTPI + E)
    s3 = (y >= xx + FORTPI - E) & (y < -xx - 5 * FORTPI - E)
    cn_s = torch.where(s1, float((ss + 1) % 4), torch.where(
        s2, float((ss + 2) % 4), _sel(s3, float((ss + 3) % 4), float(ss), x)))
    cn = torch.where(north, cn_n, cn_s)
    pole = _sel(north, float(ns), float(ss), x)
    k = torch.where(north, -1 * (cn - pole), cn - pole)
    rx, ry = _rot_quarter(k, x - c_x, y - c_y)
    ax = -3 * FORTPI + cn * HALFPI
    ay = _sel(north, 1.0, -1.0, x) * HALFPI
    polar = north | south
    return torch.where(polar, rx + ax, x), torch.where(polar, ry + ay, y)


class HEALPix(_Base):
    """+proj=healpix (healpix.cpp:607-634) and +proj=rhealpix (healpix.cpp:636-677)."""

    def __init__(self, params):
        super().__init__(params)
        proj = self.params.get("proj", "healpix")
        self.rhealpix = proj == "rhealpix"
        self.rot_xy = 0.0
        self.north_square = self.south_square = 0
        if self.rhealpix:
            self.north_square = int(self.params.get("north_square", 0))
            self.south_square = int(self.params.get("south_square", 0))
            if not 0 <= self.north_square <= 3:
                raise ValueError("Invalid value for north_square: it should be in [0,3] range.")
            if not 0 <= self.south_square <= 3:
                raise ValueError("Invalid value for south_square: it should be in [0,3] range.")
            self.verts = _rhealpix_verts(self.north_square, self.south_square)
        else:
            self.rot_xy = math.radians(float(self.params.get("rot_xy", 0.0)))
            self.verts = _healpix_verts()
        self.ellipsoidal = self.es != 0.0
        if self.ellipsoidal:
            self.auth = Authalic(self.e, self.es, self.one_es)
            self.qp = self.auth.qp
            # P->a = a * sqrt(0.5 qp)  (authalic radius), healpix.cpp:623-625 / 667-668
            self.a = self.a * math.sqrt(0.5 * self.qp)
            self.ell.a = self.a
            self.ell.ra = 1.0 / self.a

    def _rotate(self, x, y, sign):
        """rotate(p, angle) (healpix.cpp:94-99) with angle = sign*rot_xy."""
        ca = math.cos(sign * self.rot_xy)
        sa = math.sin(sign * self.rot_xy)
        return x * ca - y * sa, y * ca + x * sa

    def _fwd_impl(self, lam, phi):
        if self.ellipsoidal:                              # e_healpix_forward 500-505
            phi = self.auth.lat(phi, torch.sin(phi), torch.cos(phi))
        x, y = healpix_sphere(lam, phi)
        if self.rhealpix:                                 # s/e_rhealpix_forward 543-558
            return combine_caps_fwd(x, y, self.north_square, self.south_square)
        return self._rotate(x, y, -1.0)                   # s_healpix_forward 493-498

    def _inv_impl(self, x, y):
        if self.rhealpix:                                 # s/e_rhealpix_inverse 560-594
            inside = pnpoly(self.verts, x, y)
            x, y = combine_caps_inv(x, y, self.north_square, self.south_square)
        else:                                             # s/e_healpix_inverse 507-541
            x, y = self._rotate(x, y, 1.0)
            inside = pnpoly(self.verts, x, y)
        lam, phi = healpix_sphere_inverse(x, y)
        if self.ellipsoidal:
            phi = self.auth.lat_inv(phi)
        return torch.where(inside, lam, float("nan")), torch.where(inside, phi, float("nan"))


# ===========================================================================
# Mollweide / Wagner IV / Wagner V (moll.cpp)
# ===========================================================================
def moll_newton_masked(phi, k, n_iter=MOLL_MAX_ITER, tol=MOLL_LOOP_TOL):
    """moll.cpp:27-33 as a masked loop: every point runs the loop body at most
    n_iter times and freezes after the step where |V| < LOOP_TOL (the C `break`).
    Returns (theta2, converged) with theta2 the iterate (= 2*theta)."""
    th = phi
    active = torch.ones_like(phi, dtype=torch.bool)
    for _ in range(n_iter):
        den = 1.0 + torch.cos(th)
        V = (th + torch.sin(th) - k) / torch.where(active, den, 1.0)
        th = torch.where(active, th - V, th)
        active = active & ~(V.abs() < tol)
    return th, ~active


def moll_newton_fixed(phi, k, n_iter=MOLL_MAX_ITER, tol=MOLL_LOOP_TOL):
    """Fixed-N Newton: n_iter unconditional steps, no per-point state. The
    convergence test of moll.cpp:31/34 is applied to the last step only."""
    th = phi
    V = torch.zeros_like(phi)
    for _ in range(n_iter):
        den = 1.0 + torch.cos(th)
        ok = den != 0.0
        V = torch.where(ok, (th + torch.sin(th) - k) / torch.where(ok, den, 1.0), 0.0)
        th = th - V
    return th, V.abs() < tol


class MollThetaImplicit(torch.autograd.Function):
    """theta2 = root of F(theta2, phi) = theta2 + sin(theta2) - C_p sin(phi) = 0.
    Value: masked Newton loop (faithful). Gradient: implicit function theorem,
    d theta2 / d phi = C_p cos(phi) / (1 + cos(theta2)); zero for non-converged
    points (PROJ replaces them by the constant +-pi/2). First derivatives only."""

    @staticmethod
    def forward(ctx, phi, C_p, n_iter, tol):
        with torch.no_grad():
            th, conv = moll_newton_masked(phi, C_p * torch.sin(phi), n_iter, tol)
        ctx.save_for_backward(phi, th, conv)
        ctx.C_p = C_p
        ctx.mark_non_differentiable(conv)
        return th, conv

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, g_th, g_conv):
        phi, th, conv = ctx.saved_tensors
        den = 1.0 + torch.cos(th)
        ok = conv & (den != 0.0)
        dth = torch.where(ok, ctx.C_p * torch.cos(phi) / torch.where(ok, den, 1.0), 0.0)
        return g_th * dth, None, None, None


class Mollweide(_Base):
    """+proj=moll / wag4 / wag5 (moll.cpp:57-110). Spherical only: PROJ sets
    P->es = 0 but keeps P->a (so +ellps=GRS80 means a sphere of radius 6378137).

    Gradient notes for the forward Newton iteration:
      * masked/fixed: autograd differentiates the unrolled iterations (30 steps of
        saved activations per point). Away from the poles Newton converges
        quadratically and the derivative of the iterate converges to the
        implicit-function derivative with an error O(error of the previous
        iterate) <= ~1e-7 relative for the PROJ stopping rule.
      * implicit: exact derivative of the root, evaluated once (O(1) memory).
    """

    def __init__(self, params, method="masked", n_iter=MOLL_MAX_ITER, loop_tol=MOLL_LOOP_TOL):
        super().__init__(params)
        proj = self.params.get("proj", "moll")
        self.es = 0.0                                     # moll.cpp:61 / 101
        if proj == "wag5":                                # moll.cpp:94-110
            self.C_x, self.C_y, self.C_p = 0.90977, 1.65014, 3.00896
        else:                                             # moll.cpp:57-72
            p = HALFPI if proj == "moll" else math.pi / 3.0
            p2 = p + p
            sp = math.sin(p)
            r = math.sqrt(2 * math.pi * sp / (p2 + math.sin(p2)))
            self.C_x = 2.0 * r / math.pi
            self.C_y = r / sp
            self.C_p = p2 + math.sin(p2)
        if method not in ("masked", "fixed", "implicit"):
            raise ValueError(method)
        self.method = method
        self.n_iter = n_iter
        # PROJ's LOOP_TOL = 1e-7 is below float32 resolution of theta (ulp(pi) =
        # 2.4e-7): in float32 most points never "converge" and fall back to +-pi/2.
        # loop_tol is exposed only so that this can be demonstrated/worked around.
        self.loop_tol = loop_tol

    def theta2(self, phi):
        """The Newton solve of moll.cpp:27-33 -> (iterate, converged)."""
        if self.method == "implicit":
            return MollThetaImplicit.apply(phi, self.C_p, self.n_iter, self.loop_tol)
        k = self.C_p * torch.sin(phi)
        if self.method == "fixed":
            return moll_newton_fixed(phi, k, self.n_iter, self.loop_tol)
        return moll_newton_masked(phi, k, self.n_iter, self.loop_tol)

    def _fwd_impl(self, lam, phi):                         # moll.cpp:22-41
        th, conv = self.theta2(phi)
        th = torch.where(conv, th * 0.5, _sel(th < 0.0, -HALFPI, HALFPI, th))
        x = self.C_x * lam * torch.cos(th)
        y = self.C_y * torch.sin(th)
        return x, y

    def _inv_impl(self, x, y):                             # moll.cpp:43-55
        phi, err1 = aasin(y / self.C_y)
        lam = x / (self.C_x * torch.cos(phi))
        ok = lam.abs() < math.pi
        phi = phi + phi
        phi, err2 = aasin((phi + torch.sin(phi)) / self.C_p)
        bad = (~ok) | err1 | err2
        return torch.where(bad, float("nan"), lam), torch.where(bad, float("nan"), phi)


# ---------------------------------------------------------------------------
def make(definition, **kw):
    """Build the right Projection from a PROJ string or dict."""
    params = parse_proj_string(definition) if isinstance(definition, str) else dict(definition)
    proj = params.get("proj")
    if proj == "laea":
        return LambertAzimuthalEqualArea(params)
    if proj in ("healpix", "rhealpix"):
        return HEALPix(params)
    if proj in ("moll", "wag4", "wag5"):
        return Mollweide(params, **kw)
    raise ValueError(f"unsupported projection {proj!r}")
