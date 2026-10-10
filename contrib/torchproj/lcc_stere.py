"""PyTorch port of PROJ's Lambert Conformal Conic (lcc), Stereographic (stere)
and Universal Polar Stereographic (ups), plus the helpers pj_tsfn, pj_msfn and
pj_phi2 / pj_sinhpsi2tanphi.

Sources ported (this PROJ tree):
  src/tsfn.cpp:6-29            pj_tsfn            -> tsfn() / _tsfn_scalar()
  src/msfn.cpp:5-7             pj_msfn            -> _msfn_scalar()
  src/phi2.cpp:39-150          pj_sinhpsi2tanphi  -> sinhpsi2tanphi()
  src/phi2.cpp:153-176         pj_phi2            -> phi2()
  src/projections/lcc.cpp      lcc fwd/inv/setup  -> LambertConformalConic
  src/projections/stere.cpp    stere/ups          -> Stereographic, UniversalPolarStereographic
  src/dmstor.cpp:26-114        dmstor (subset)    -> dmstor()
  src/ell_set.cpp:274-339      shape -> es        -> _proj_es()
  src/init.cpp:680-716         units/to_meter     -> _BaseProj

Conventions follow core.py (unit ellipsoid _fwd/_inv, NaN for failed points).

Iterative parts (both have a convergence test in PROJ):
  * pj_sinhpsi2tanphi: Newton, cap numit=5, tol sqrt(eps)/10 (phi2.cpp:122-149)
  * stere_e_inverse:   fixed point, cap NITER=8, |dphi| < CONV=1e-10 (stere.cpp:173-187)
  iter_mode="masked" (default, faithful): always runs the PROJ cap number of
      iterations; a point is frozen (torch.where) once PROJ would have broken
      out; points that never converge are failures (NaN), exactly like PROJ
      (errno set -> pj_fwd/pj_inv return HUGE_VAL).
  iter_mode="fixed": n_iter plain unrolled iterations, no convergence test, no
      failure flag (cheaper, compile-friendly, see report for accuracy).
  For dtypes other than float64 the hard-coded double tolerances are replaced
  by dtype-aware ones (GeographicLib's tauf is templated on the type).  With
  PROJ's double tolerances, float32 Newton steps that oscillate by one ulp never
  satisfy |dtau| < sqrt(DBL_EPSILON)/10 and ~0.7% of points would be flagged as
  failures in pj_phi2.  For float64 the tolerances are exactly PROJ's.

Gradient hygiene: every branch whose untaken side could produce inf/NaN gets
"double where" safe inputs.  At removable singularities where PROJ uses a
constant special case (stere pole in polar mode, projection centre of the
oblique/equatorial inverse) the value is PROJ's, but the gradient comes from an
algebraically equivalent smooth formula (straight-through), so autograd gives
the true finite Jacobian there instead of 0.
"""
import math
from typing import Dict, Optional, Tuple, Union

import torch

from core import HALFPI, PJ_EPS_LAT, Ellipsoid, Projection, adjlon, parse_proj_string

M_FORTPI = math.pi / 4
M_PI_2 = math.pi / 2
DEG_TO_RAD = 0.017453292519943296  # proj_internal.h
EPS10 = 1.0e-10


class ProjSetupError(ValueError):
    """Raised where PROJ's setup returns pj_default_destructor(P, err)."""


# ----------------------------------------------------------------------------
# C-like scalar maths for setup (Python's math raises where libm returns
# inf/nan; PROJ's setup checks such as `n == 0` rely on the libm behaviour).
# ----------------------------------------------------------------------------
def _clog(x: float) -> float:
    if x == 0.0:
        return -math.inf
    if x < 0.0 or math.isnan(x):
        return math.nan
    return math.log(x)


def _cpow(x: float, y: float) -> float:
    try:
        return math.pow(x, y)
    except ZeroDivisionError:  # pow(0, negative)
        return math.inf
    except OverflowError:
        return math.inf
    except ValueError:  # negative base, non-integer exponent
        return math.nan


def _catanh(x: float) -> float:
    if x == 1.0:
        return math.inf
    if x == -1.0:
        return -math.inf
    if abs(x) > 1.0:
        return math.nan
    return math.atanh(x)


def _ctan(x: float) -> float:
    return math.tan(x)


def _cdiv(a: float, b: float) -> float:
    if b == 0.0:
        if a == 0.0 or math.isnan(a):
            return math.nan
        return math.copysign(math.inf, a) * math.copysign(1.0, b)
    return a / b


def _tsfn_scalar(phi: float, sinphi: float, e: float) -> float:
    """src/tsfn.cpp:6-29 (Python floats, for setup)."""
    cosphi = math.cos(phi)
    try:
        f = math.exp(e * _catanh(e * sinphi))
    except OverflowError:
        f = math.inf
    return f * (_cdiv(cosphi, 1 + sinphi) if sinphi > 0 else _cdiv(1 - sinphi, cosphi))


def _msfn_scalar(sinphi: float, cosphi: float, es: float) -> float:
    """src/msfn.cpp:5-7."""
    return _cdiv(cosphi, math.sqrt(1.0 - es * sinphi * sinphi))


# ----------------------------------------------------------------------------
# Parameter parsing helpers
# ----------------------------------------------------------------------------
_VM = (DEG_TO_RAD, 0.0002908882086657216, 0.0000048481368110953599)  # dmstor.cpp:16


def _strtod(s: str, i: int) -> Tuple[Optional[float], int]:
    """strtod on s[i:], with dmstor.cpp's proj_strtod quirk ('d'/'D' end a number)."""
    j = i
    n = len(s)
    while j < n and (s[j].isdigit() or s[j] == "."):
        j += 1
    if j < n and s[j] in "eE":
        k = j + 1
        if k < n and s[k] in "+-":
            k += 1
        if k < n and s[k].isdigit():
            while k < n and s[k].isdigit():
                k += 1
            j = k
    if j == i:
        return None, i
    return float(s[i:j]), j


def dmstor(value) -> float:
    """src/dmstor.cpp:26-114: DMS string (or plain number of degrees) -> radians."""
    if isinstance(value, (int, float)):
        return float(value) * DEG_TO_RAD
    s = str(value).strip()
    sign = "+"
    i = 0
    if s[:1] in "+-":
        sign = s[0]
        i = 1
    v = 0.0
    nl = 0
    while nl < 3:
        if i >= len(s) or not (s[i].isdigit() or s[i] == "."):
            break
        tv, i = _strtod(s, i)
        if tv is None:
            break
        c = s[i] if i < len(s) else ""
        if c in ("D", "d", "°"):
            n = 0
        elif c == "'":
            n = 1
        elif c == '"':
            n = 2
        elif c in ("r", "R"):
            if nl:
                raise ProjSetupError("invalid DMS value %r" % value)
            i += 1
            v = tv
            nl = 5  # n = 4 -> nl = n + 1
            continue
        else:
            v += tv * _VM[nl]
            nl = 5
            continue
        if n < nl:
            raise ProjSetupError("invalid DMS value %r" % value)
        v += tv * _VM[n]
        i += 1
        nl = n + 1
    if i < len(s) and s[i] in "NnEeSsWw":
        sign = "-" if s[i] in "SsWw" else "+"
    return -v if sign == "-" else v


# src/units.cpp:27 etc. (to_meter strings, as parsed by init.cpp:693-713)
_UNITS = {"m": "1", "km": "1000", "dm": "1/10", "cm": "1/100", "mm": "1/1000",
          "ft": "0.3048", "us-ft": "0.304800609601219", "us-mi": "1609.347218694437",
          "mi": "1609.344", "yd": "0.9144", "in": "0.0254", "fath": "1.8288",
          "kmi": "1852"}


def _to_meter(params) -> float:
    s = None
    if "units" in params:
        if params["units"] not in _UNITS:
            raise ProjSetupError("Invalid value for units")
        s = _UNITS[params["units"]]
    elif "to_meter" in params:
        s = str(params["to_meter"])
    if s is None:
        return 1.0
    if "/" in s:
        num, den = s.split("/", 1)
        if float(den) == 0.0:
            raise ProjSetupError("Invalid value for to_meter denominator")
        to_meter = float(num) / float(den)
    else:
        to_meter = float(s)
    if to_meter <= 0.0:
        raise ProjSetupError("Invalid value for to_meter")
    return to_meter


def _proj_es(params, ell: Ellipsoid) -> float:
    """Squared eccentricity exactly as PROJ computes it (ell_set.cpp:274-339).
    core.Ellipsoid uses es = 1 - b^2/a^2 for b-defined shapes, PROJ uses
    f = (a - b)/a, es = 2f - f^2; this matters for degenerate gie cases."""
    if "R" in params:
        return 0.0
    for key in ("rf", "f", "es", "e", "b"):
        if key in params:
            if key == "b":
                b = float(params["b"])
                if b <= 0:
                    raise ProjSetupError("Invalid value for b. Should be > 0")
                if b == ell.a:
                    return 0.0
                f = (ell.a - b) / ell.a
                return 2 * f - f * f
            return ell.es
    ellps = params.get("ellps")
    if ellps == "clrk66":  # builtin definition is a=6378206.4 b=6356583.8
        f = (ell.a - 6356583.8) / ell.a
        return 2 * f - f * f
    return ell.es


class _BaseProj(Projection):
    """core.Projection + the PROJ details the spike base class leaves out:
    DMS angles (dmstor), PROJ's es for b-defined shapes, and +units/+to_meter
    (fwd.cpp:163-164, inv.cpp:79-80)."""

    _ANGLE_KEYS = ("lon_0", "lat_0", "lat_1", "lat_2", "lat_ts")

    def __init__(self, params: Union[str, dict]):
        if isinstance(params, str):
            params = parse_proj_string(params)
        params = dict(params)
        self.rad = {k: dmstor(params[k]) for k in self._ANGLE_KEYS if k in params}
        clean = dict(params)
        for k, v in self.rad.items():
            clean[k] = repr(v / DEG_TO_RAD)
        super().__init__(clean)
        self.params = params
        self.lam0 = self.rad.get("lon_0", 0.0)
        self.phi0 = self.rad.get("lat_0", 0.0)
        es = _proj_es(params, self.ell)
        if not (es >= 0):  # ell_set.cpp:336-339 (written that way to catch NaN)
            raise ProjSetupError("Invalid eccentricity")
        self.es = es
        self.e = math.sqrt(es)
        self.one_es = 1.0 - es
        self.to_meter = _to_meter(params)
        self.fr_meter = 1.0 / self.to_meter

    def forward(self, coo: torch.Tensor) -> torch.Tensor:
        # Same steps as core.Projection.forward (fwd.cpp:40-125, 138-166), except
        # for the latitude clamp: torch.clamp's backward is 0 *at* the bounds, so
        # core's phi.clamp(-HALFPI, HALFPI) zeroes d/dphi exactly at the poles.
        # Here: identical value, identity gradient.
        lam, phi = coo[..., 0], coo[..., 1]
        bad = phi.abs() > HALFPI + PJ_EPS_LAT
        phi = phi - (phi - phi.clamp(-HALFPI, HALFPI)).detach()
        lam = lam - self.lam0
        if not self.over:
            lam = adjlon(lam)
        x, y = self._fwd(lam, phi)
        x = self.fr_meter * (self.a * x + self.x0) if self.fr_meter != 1.0 else self.a * x + self.x0
        y = self.fr_meter * (self.a * y + self.y0) if self.fr_meter != 1.0 else self.a * y + self.y0
        x = _nan_where(bad, x)
        y = _nan_where(bad, y)
        return torch.cat([torch.stack([x, y], -1), coo[..., 2:]], -1)

    def inverse(self, coo: torch.Tensor) -> torch.Tensor:
        if self.to_meter != 1.0:
            coo = torch.cat([coo[..., :2] * self.to_meter, coo[..., 2:]], -1)
        return super().inverse(coo)


# ----------------------------------------------------------------------------
# Tensor helpers
# ----------------------------------------------------------------------------
def _nan_where(bad: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.where(bad, torch.full_like(v, float("nan")), v)


def _safe_hypot(x: torch.Tensor, y: torch.Tensor):
    """hypot(x, y) with a zero (not NaN) gradient at the origin.
    Returns (rho, origin_mask, x_safe, y_safe)."""
    origin = (x == 0) & (y == 0)
    xs = torch.where(origin, torch.ones_like(x), x)
    ys = torch.where(origin, torch.zeros_like(y), y)
    rho = torch.where(origin, torch.zeros_like(x), torch.hypot(xs, ys))
    return rho, origin, xs, ys


def _safe_atan2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """atan2(a, b); value of C's atan2 at (+-0, +-0), zero gradient there."""
    zero = (a == 0) & (b == 0)
    val0 = torch.atan2(a.detach(), b.detach())
    a_s = torch.where(zero, torch.zeros_like(a), a)
    b_s = torch.where(zero, torch.ones_like(b), b)
    return torch.where(zero, val0, torch.atan2(a_s, b_s))


def _atan2_or_zero(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C idiom `(a == 0. && b == 0.) ? 0. : atan2(a, b)` with safe gradients."""
    zero = (a == 0) & (b == 0)
    a_s = torch.where(zero, torch.zeros_like(a), a)
    b_s = torch.where(zero, torch.ones_like(b), b)
    return torch.where(zero, torch.zeros_like(a), torch.atan2(a_s, b_s))


def _asin_dw(arg: torch.Tensor) -> torch.Tensor:
    """asin with PROJ's values (NaN for |arg| > 1, +-pi/2 at +-1) but a zero
    instead of inf/NaN gradient where |arg| >= 1 (points mapping to a pole)."""
    edge = arg.abs() >= 1
    safe = torch.where(edge, torch.zeros_like(arg), arg)
    return torch.where(edge, torch.asin(arg.detach()), torch.asin(safe))


def _straight_through(value: torch.Tensor, smooth: torch.Tensor) -> torch.Tensor:
    """Forward value `value`, gradient of `smooth`."""
    return value.detach() + (smooth - smooth.detach())


def tsfn(phi: torch.Tensor, sinphi: torch.Tensor, e: float) -> torch.Tensor:
    """src/tsfn.cpp:6-29: ts = exp(-psi)."""
    cosphi = torch.cos(phi)
    pos = sinphi > 0
    one = torch.ones_like(sinphi)
    zero = torch.zeros_like(sinphi)
    # double where: each branch only ever sees arguments with a safe denominator
    s_pos = torch.where(pos, sinphi, zero)
    c_pos = torch.where(pos, cosphi, one)
    s_neg = torch.where(pos, zero, sinphi)
    c_neg = torch.where(pos, one, cosphi)
    t = torch.where(pos, c_pos / (1 + s_pos), (1 - s_neg) / c_neg)
    if e == 0.0:  # exp(0 * atanh(0)) == 1 exactly
        return t
    return torch.exp(e * torch.atanh(e * sinphi)) * t


def _eps(dtype) -> float:
    return torch.finfo(dtype).eps


def sinhpsi2tanphi(taup: torch.Tensor, e: float, mode: str = "masked",
                   n_iter: Optional[int] = None,
                   stats: Optional[dict] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """src/phi2.cpp:39-150 (GeographicLib's tauf). tau' = sinh(psi) -> tau = tan(phi).
    Returns (tau, failed) where failed marks points where PROJ sets errno
    (Newton did not satisfy the tolerance within numit=5 iterations)."""
    numit = 5  # phi2.cpp:122
    rooteps = math.sqrt(_eps(taup.dtype))  # == sqrt(DBL_EPSILON) for float64
    tol = rooteps / 10  # phi2.cpp:125
    tmax = 2 / rooteps  # phi2.cpp:126
    e2m = 1 - e * e  # phi2.cpp:128
    stol = tol * torch.clamp(taup.abs(), min=1.0)  # phi2.cpp:129
    # phi2.cpp:131 initial guess
    tau0 = torch.where(taup.abs() > 70, taup * math.exp(e * _catanh(e)), taup / e2m)
    # phi2.cpp:132-133: |tau| >= tmax (or inf/nan) returns the initial guess
    skip = ~(tau0.abs() < tmax)
    zero = torch.zeros_like(taup)
    tau = torch.where(skip, zero, tau0)  # double where: safe values for the loop
    taup_s = torch.where(skip, zero, taup)
    active = ~skip
    niter = numit if mode == "masked" else int(n_iter if n_iter is not None else 2)
    count = torch.zeros_like(taup, dtype=torch.int32) if stats is not None else None
    for _ in range(niter):  # phi2.cpp:137-146
        if count is not None:  # diagnostics only: iterations PROJ would run
            count = count + active.to(torch.int32)
        tau1 = torch.sqrt(1 + tau * tau)
        sig = torch.sinh(e * torch.atanh(e * tau / tau1))
        taupa = torch.sqrt(1 + sig * sig) * tau - sig * tau1
        dtau = ((taup_s - taupa) * (1 + e2m * (tau * tau))
                / (e2m * tau1 * torch.sqrt(1 + taupa * taupa)))
        if mode == "masked":
            tau = torch.where(active, tau + dtau, tau)
            # "if (!(fabs(dtau) >= stol)) break;"
            active = active & (dtau.abs() >= stol)
        else:
            tau = tau + dtau
    if count is not None:
        stats["iters"] = count
    if mode == "masked":
        failed = active  # loop ran out (i == 0): phi2.cpp:147-148
    else:
        failed = torch.zeros_like(skip)
    tau = torch.where(skip, tau0, tau)
    return tau, failed


def phi2(ts0: torch.Tensor, e: float, mode: str = "masked",
         n_iter: Optional[int] = None,
         stats: Optional[dict] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """src/phi2.cpp:153-176: latitude from ts = exp(-psi). Returns (phi, failed)."""
    taup = (1 / ts0 - ts0) / 2
    tau, failed = sinhpsi2tanphi(taup, e, mode, n_iter, stats)
    return torch.atan(tau), failed


# ----------------------------------------------------------------------------
# Lambert Conformal Conic  (src/projections/lcc.cpp)
# ----------------------------------------------------------------------------
class LambertConformalConic(_BaseProj):
    def __init__(self, params, iter_mode: str = "masked", n_iter: Optional[int] = None):
        super().__init__(params)
        self.iter_mode = iter_mode
        self.n_iter = n_iter
        self.stats = None  # set to a dict to collect iteration counts (diagnostics)
        p = self.params
        # lcc.cpp:88-95
        phi1 = self.rad.get("lat_1", 0.0)
        if "lat_2" in p:
            phi2_ = self.rad["lat_2"]
        else:
            phi2_ = phi1
            if "lat_0" not in p:
                self.phi0 = phi1
        self.phi1, self.phi2 = phi1, phi2_
        # lcc.cpp:97-101
        if abs(phi1 + phi2_) < EPS10:
            raise ProjSetupError("Invalid value for lat_1 and lat_2: |lat_1 + lat_2| should be > 0")
        # lcc.cpp:103-115
        n = sinphi = math.sin(phi1)
        cosphi = math.cos(phi1)
        if abs(cosphi) < EPS10 or abs(phi1) >= M_PI_2:
            raise ProjSetupError("Invalid value for lat_1: |lat_1| should be < 90")
        if abs(math.cos(phi2_)) < EPS10 or abs(phi2_) >= M_PI_2:
            raise ProjSetupError("Invalid value for lat_2: |lat_2| should be < 90")
        secant = abs(phi1 - phi2_) >= EPS10  # lcc.cpp:117
        es, e, phi0 = self.es, self.e, self.phi0
        if es != 0.0:  # lcc.cpp:118-146
            m1 = _msfn_scalar(sinphi, cosphi, es)
            ml1 = _tsfn_scalar(phi1, sinphi, e)
            if secant:
                sinphi = math.sin(phi2_)
                n = _clog(_cdiv(m1, _msfn_scalar(sinphi, math.cos(phi2_), es)))
                if n == 0:
                    raise ProjSetupError("Invalid value for eccentricity")
                ml2 = _tsfn_scalar(phi2_, sinphi, e)
                denom = _clog(_cdiv(ml1, ml2))
                if denom == 0:
                    raise ProjSetupError("Invalid value for eccentricity")
                n = _cdiv(n, denom)
            rho0 = _cdiv(m1 * _cpow(ml1, -n), n)
            c = rho0
            rho0 *= (0.0 if abs(abs(phi0) - M_PI_2) < EPS10
                     else _cpow(_tsfn_scalar(phi0, math.sin(phi0), e), n))
        else:  # lcc.cpp:147-165
            if secant:
                n = _cdiv(_clog(cosphi / math.cos(phi2_)),
                          _clog(_cdiv(_ctan(M_FORTPI + .5 * phi2_), _ctan(M_FORTPI + .5 * phi1))))
            if n == 0:
                raise ProjSetupError("Invalid value for lat_1 and lat_2: |lat_1 + lat_2| should be > 0")
            c = _cdiv(cosphi * _cpow(_ctan(M_FORTPI + .5 * phi1), n), n)
            rho0 = (0.0 if abs(abs(phi0) - M_PI_2) < EPS10
                    else c * _cpow(_ctan(M_FORTPI + .5 * phi0), -n))
        self.n, self.c, self.rho0 = n, c, rho0

    def _fwd(self, lam, phi):  # lcc.cpp:22-42
        n = self.n
        at_pole = (phi.abs() - M_PI_2).abs() < EPS10
        fail = at_pole & (phi * n <= 0)  # lcc.cpp:28-31
        phi_s = torch.where(at_pole, torch.zeros_like(phi), phi)  # double where
        if self.es != 0.0:
            rho = self.c * torch.pow(tsfn(phi_s, torch.sin(phi_s), self.e), n)
        else:
            rho = self.c * torch.pow(torch.tan(M_FORTPI + .5 * phi_s), -n)
        rho = torch.where(at_pole, torch.zeros_like(rho), rho)  # lcc.cpp:32
        lam = lam * n
        x = self.k0 * (rho * torch.sin(lam))
        y = self.k0 * (self.rho0 - rho * torch.cos(lam))
        return _nan_where(fail, x), _nan_where(fail, y)

    def _inv(self, x, y):  # lcc.cpp:44-76
        n = self.n
        x = x / self.k0
        y = y / self.k0
        y = self.rho0 - y
        rho, origin, xs, ys = _safe_hypot(x, y)
        rho_s = torch.where(origin, torch.ones_like(rho), rho)
        if n < 0.0:  # lcc.cpp:55-59
            rho_s = -rho_s
            xs = -xs
            ys = -ys
        if self.es != 0.0:
            phi, failed = phi2(torch.pow(rho_s / self.c, 1.0 / n), self.e,
                               self.iter_mode, self.n_iter, self.stats)
        else:
            phi = 2.0 * torch.atan(torch.pow(self.c / rho_s, 1.0 / n)) - M_PI_2
            failed = torch.zeros_like(origin)
        lam = torch.atan2(xs, ys) / n
        # lcc.cpp:71-74
        lam = torch.where(origin, torch.zeros_like(lam), lam)
        phi = torch.where(origin, torch.full_like(phi, M_PI_2 if n > 0 else -M_PI_2), phi)
        bad = failed & ~origin
        return _nan_where(bad, lam), _nan_where(bad, phi)


# ----------------------------------------------------------------------------
# Stereographic  (src/projections/stere.cpp)
# ----------------------------------------------------------------------------
S_POLE, N_POLE, OBLIQ, EQUIT = 0, 1, 2, 3
_TOL = 1.e-8  # stere.cpp:27
_NITER = 8  # stere.cpp:28
_CONV = 1.e-10  # stere.cpp:29


def _ssfn(phit, sinphi, eccen: float):
    """stere.cpp:31-35."""
    sinphi = sinphi * eccen
    return torch.tan(.5 * (M_PI_2 + phit)) * torch.pow((1. - sinphi) / (1. + sinphi), .5 * eccen)


def _ssfn_scalar(phit: float, sinphi: float, eccen: float) -> float:
    sinphi *= eccen
    return math.tan(.5 * (M_PI_2 + phit)) * _cpow((1. - sinphi) / (1. + sinphi), .5 * eccen)


class Stereographic(_BaseProj):
    def __init__(self, params, iter_mode: str = "masked", n_iter: Optional[int] = None,
                 smooth_grad: bool = True):
        super().__init__(params)
        self.iter_mode = iter_mode
        self.n_iter = n_iter
        self.smooth_grad = smooth_grad
        self.stats = None  # set to a dict to collect iteration counts (diagnostics)
        self._entry()
        self._setup()

    def _entry(self):  # stere.cpp:295-307
        self.phits = self.rad["lat_ts"] if "lat_ts" in self.params else M_PI_2

    def _setup(self):  # stere.cpp:233-293
        phi0 = self.phi0
        t = abs(phi0)
        if abs(t - M_PI_2) < EPS10:
            self.mode = S_POLE if phi0 < 0. else N_POLE
        else:
            self.mode = OBLIQ if t > EPS10 else EQUIT
        self.phits = abs(self.phits)
        self.sinX1 = 0.0
        self.cosX1 = 0.0
        e, k0 = self.e, self.k0
        if self.es != 0.0:
            if self.mode in (N_POLE, S_POLE):
                if abs(self.phits - M_PI_2) < EPS10:
                    self.akm1 = 2. * k0 / math.sqrt(_cpow(1 + e, 1 + e) * _cpow(1 - e, 1 - e))
                else:
                    t = math.sin(self.phits)
                    self.akm1 = _cdiv(math.cos(self.phits), _tsfn_scalar(self.phits, t, e))
                    t *= e
                    self.akm1 /= math.sqrt(1. - t * t)
            else:
                t = math.sin(phi0)
                X = 2. * math.atan(_ssfn_scalar(phi0, t, e)) - M_PI_2
                t *= e
                self.akm1 = 2. * k0 * math.cos(phi0) / math.sqrt(1. - t * t)
                self.sinX1 = math.sin(X)
                self.cosX1 = math.cos(X)
        else:
            if self.mode == OBLIQ:
                self.sinX1 = math.sin(phi0)  # sinph0
                self.cosX1 = math.cos(phi0)  # cosph0
            if self.mode in (OBLIQ, EQUIT):
                self.akm1 = 2. * k0
            else:
                self.akm1 = (math.cos(self.phits) / math.tan(M_FORTPI - .5 * self.phits)
                             if abs(self.phits - M_PI_2) >= EPS10 else 2. * k0)

    # ---------------------------------------------------------------- forward
    def _fwd(self, lam, phi):
        return self._e_fwd(lam, phi) if self.es != 0.0 else self._s_fwd(lam, phi)

    def _e_fwd(self, lam, phi):  # stere.cpp:37-92
        coslam = torch.cos(lam)
        sinlam = torch.sin(lam)
        sinphi = torch.sin(phi)
        e, akm1 = self.e, self.akm1
        mode = self.mode
        if mode in (OBLIQ, EQUIT):
            X = 2. * torch.atan(_ssfn(phi, sinphi, e)) - M_PI_2
            sinX = torch.sin(X)
            cosX = torch.cos(X)
        if mode == OBLIQ:
            denom = self.cosX1 * (1. + self.sinX1 * sinX + self.cosX1 * cosX * coslam)
            bad = denom == 0
            A = akm1 / torch.where(bad, torch.ones_like(denom), denom)
            y = A * (self.cosX1 * sinX - self.sinX1 * cosX * coslam)
            x = A * cosX
        elif mode == EQUIT:
            d = 1. + cosX * coslam
            # PROJ returns x = 0, y = HUGE_VAL here (stere.cpp:67-68); HUGE_VAL is
            # PROJ's error marker, so the port reports the point as failed (NaN).
            bad = d == 0.0
            A = akm1 / torch.where(bad, torch.ones_like(d), d)
            y = A * sinX
            x = A * cosX
        else:
            if mode == S_POLE:  # stere.cpp:76-80
                phi = -phi
                coslam = -coslam
                sinphi = -sinphi
            at_pole = (phi - M_PI_2).abs() < 1e-15  # stere.cpp:82
            xt = akm1 * tsfn(phi, sinphi, e)  # finite at the pole (cos(pi/2) > 0)
            if self.smooth_grad:
                # value 0 as in PROJ, gradient of the general formula (finite)
                x = torch.where(at_pole, xt - xt.detach(), xt)
            else:
                x = torch.where(at_pole, torch.zeros_like(xt), xt)
            y = -x * coslam
            bad = torch.zeros_like(at_pole)
        x = x * sinlam
        return _nan_where(bad, x), _nan_where(bad, y)

    def _s_fwd(self, lam, phi):  # stere.cpp:94-135
        sinphi = torch.sin(phi)
        cosphi = torch.cos(phi)
        coslam = torch.cos(lam)
        sinlam = torch.sin(lam)
        mode, akm1 = self.mode, self.akm1
        if mode in (EQUIT, OBLIQ):
            if mode == EQUIT:
                d = 1. + cosphi * coslam
            else:
                d = 1. + self.sinX1 * sinphi + self.cosX1 * cosphi * coslam
            bad = d <= EPS10  # stere.cpp:111
            yy = akm1 / torch.where(bad, torch.ones_like(d), d)
            x = yy * cosphi * sinlam
            y = yy * (sinphi if mode == EQUIT
                      else self.cosX1 * sinphi - self.sinX1 * cosphi * coslam)
        else:
            if mode == N_POLE:  # stere.cpp:120-123
                coslam = -coslam
                phi = -phi
            bad = (phi - M_PI_2).abs() < _TOL  # stere.cpp:125
            phi_s = torch.where(bad, torch.zeros_like(phi), phi)
            yy = akm1 * torch.tan(M_FORTPI + .5 * phi_s)
            x = sinlam * yy
            y = yy * coslam
        return _nan_where(bad, x), _nan_where(bad, y)

    # ---------------------------------------------------------------- inverse
    def _inv(self, x, y):
        return self._e_inv(x, y) if self.es != 0.0 else self._s_inv(x, y)

    def _conv_tol(self, dtype) -> float:
        return _CONV if dtype == torch.float64 else max(_CONV, 4 * _eps(dtype))

    def _e_inv(self, x, y):  # stere.cpp:137-188
        e, akm1, mode = self.e, self.akm1, self.mode
        sinX1, cosX1 = self.sinX1, self.cosX1
        rho, origin, xs, ys = _safe_hypot(x, y)
        if mode in (OBLIQ, EQUIT):
            tp = 2. * torch.atan2(rho * cosX1, torch.full_like(rho, akm1))
            cosphi = torch.cos(tp)
            sinphi = torch.sin(tp)
            rho_s = torch.where(origin, torch.ones_like(rho), rho)
            phi_gen = _asin_dw(cosphi * sinX1 + (y * sinphi * cosX1 / rho_s))
            x_gen = x * sinphi
            y_gen = rho * cosX1 * cosphi - y * sinX1 * sinphi
            lam_gen = _atan2_or_zero(x_gen, y_gen)  # stere.cpp:180
            # Centre (rho == 0): PROJ uses asin(cosphi*sinX1) and lam = 0.  Same
            # values from the smooth forms below (sin(tp)/rho = S has no 0/0),
            # which also give the true Jacobian at the centre.
            S = 2. * cosX1 * akm1 / (akm1 * akm1 + (rho * cosX1) ** 2)
            phi_c = _asin_dw(cosphi * sinX1 + y * cosX1 * S)
            lam_c = torch.atan2(x * S, cosX1 * cosphi - y * sinX1 * S)
            if not self.smooth_grad:
                phi_c = _asin_dw(cosphi * sinX1)
                lam_c = torch.zeros_like(lam_c)
            phi_l = torch.where(origin, phi_c, phi_gen)
            lam = torch.where(origin, lam_c, lam_gen)
            tp = torch.tan(.5 * (M_PI_2 + phi_l))
            halfpi = M_PI_2
            halfe = .5 * e
        else:
            if mode == N_POLE:
                ys = -ys
            tp = -rho / akm1
            phi_l = M_PI_2 - 2. * torch.atan(tp)
            halfpi = -M_PI_2
            halfe = -.5 * e
            # "(xy.x == 0. && xy.y == 0.) ? 0. : atan2(xy.x, xy.y)"
            lam = torch.where(origin, torch.zeros_like(rho), torch.atan2(xs, ys))
        # stere.cpp:173-187: fixed-point iteration
        conv = self._conv_tol(x.dtype)
        if self.iter_mode == "masked":
            done = torch.zeros_like(origin)
            phi = phi_l
            count = torch.zeros_like(rho, dtype=torch.int32) if self.stats is not None else None
            for _ in range(_NITER):
                if count is not None:  # diagnostics only: iterations PROJ would run
                    count = count + (~done).to(torch.int32)
                sp = e * torch.sin(phi_l)
                phi_new = 2. * torch.atan(tp * torch.pow((1. + sp) / (1. - sp), halfe)) - halfpi
                phi = torch.where(done, phi, phi_new)
                done = done | ((phi_l - phi_new).abs() < conv)
                phi_l = torch.where(done, phi_l, phi_new)
            failed = ~done
            if count is not None:
                self.stats["iters"] = count
        else:
            for _ in range(int(self.n_iter if self.n_iter is not None else 5)):
                sp = e * torch.sin(phi_l)
                phi_l = 2. * torch.atan(tp * torch.pow((1. + sp) / (1. - sp), halfe)) - halfpi
            phi = phi_l
            failed = torch.zeros_like(origin)
        if mode == S_POLE:
            phi = -phi
        return _nan_where(failed, lam), _nan_where(failed, phi)

    def _s_inv(self, x, y):  # stere.cpp:190-231
        akm1, mode = self.akm1, self.mode
        sinph0, cosph0 = self.sinX1, self.cosX1
        rh, origin, xs, ys = _safe_hypot(x, y)
        c = 2. * torch.atan(rh / akm1)
        sinc = torch.sin(c)
        cosc = torch.cos(c)
        small = rh.abs() <= EPS10
        rh_s = torch.where(small, torch.ones_like(rh), rh)
        # sin(c)/rh without the 0/0, for gradients at the centre
        S = 2. * akm1 / (akm1 * akm1 + rh * rh)
        zero = torch.zeros_like(rh)
        if mode == EQUIT:
            phi_gen = _asin_dw(y * sinc / rh_s)
            phi_c = zero
            if self.smooth_grad:
                phi_c = _straight_through(phi_c, _asin_dw(y * S))
            phi = torch.where(small, phi_c, phi_gen)
            # lam = atan2(x*sinc, cosc*rh) if (cosc != 0 || x != 0) else 0
            lam = _safe_atan2(x * sinc, cosc * rh)
            if self.smooth_grad:
                lam = torch.where(origin, torch.atan2(x * S, cosc), lam)
            lam = torch.where((cosc != 0.) | (x != 0.), lam, zero)
        elif mode == OBLIQ:
            phi_gen = _asin_dw(cosc * sinph0 + y * sinc * cosph0 / rh_s)
            phi_c = torch.full_like(rh, self.phi0)
            if self.smooth_grad:
                phi_c = _straight_through(phi_c, _asin_dw(cosc * sinph0 + y * S * cosph0))
            phi = torch.where(small, phi_c, phi_gen)
            cc = cosc - sinph0 * torch.sin(phi)
            lam = _safe_atan2(x * sinc * cosph0, cc * rh)
            if self.smooth_grad:
                lam = torch.where(origin, torch.atan2(x * S * cosph0, cc), lam)
            lam = torch.where((cc != 0.) | (x != 0.), lam, zero)
        else:
            if mode == N_POLE:
                ys = -ys
            phi_gen = _asin_dw(-cosc if mode == S_POLE else cosc)
            if self.smooth_grad:
                # asin(cos c) == pi/2 - c for c in [0, pi): PROJ's value (which
                # loses ~sqrt(eps) near the pole), gradient of the exact identity
                phi_gen = _straight_through(phi_gen, (c - M_PI_2) if mode == S_POLE else (M_PI_2 - c))
            phi = torch.where(small, torch.full_like(rh, self.phi0), phi_gen)
            lam = torch.where(origin, zero, torch.atan2(xs, ys))
        return lam, phi


class UniversalPolarStereographic(Stereographic):
    def _entry(self):  # stere.cpp:309-331
        self.phi0 = -M_PI_2 if self.params.get("south", False) else M_PI_2
        if self.es == 0.0:
            raise ProjSetupError("Invalid value for es: only ellipsoidal formulation supported")
        self.k0 = .994
        self.x0 = 2000000.
        self.y0 = 2000000.
        self.phits = M_PI_2
        self.lam0 = 0.


_REGISTRY = {"lcc": LambertConformalConic, "stere": Stereographic,
             "ups": UniversalPolarStereographic}


def make(definition: Union[str, Dict], **kw) -> Projection:
    """Build a projection from a '+proj=lcc ...' / '+proj=stere ...' / '+proj=ups ...' string."""
    params = parse_proj_string(definition) if isinstance(definition, str) else dict(definition)
    name = params.get("proj")
    if name not in _REGISTRY:
        raise ProjSetupError("unsupported projection %r" % name)
    return _REGISTRY[name](params, **kw)
