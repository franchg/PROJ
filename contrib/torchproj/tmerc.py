"""Transverse Mercator family (tmerc, etmerc, utm) ported from PROJ to PyTorch.

Source: src/projections/tmerc.cpp (PROJ 9.8.x), plus the
helpers it relies on: src/latitudes.cpp (pj_auxlat_coeffs, pj_polyval,
pj_clenshaw, pj_auxlat_convert, pj_rectifying_radius), src/mlfn.cpp
(pj_enfn, pj_mlfn, pj_inv_mlfn), src/adjlon.cpp, the shape checks of
src/ell_set.cpp and the to_meter handling of src/init.cpp / fwd.cpp / inv.cpp.

Algorithms (selected at setup time, exactly like tmerc.cpp:510-597):
  * "poder_engsager" (default, "exact"): Gauss/Krueger series of order 6 with
    a complex Clenshaw summation (clenS).
  * "evenden_snyder" (+approx): Snyder's power series around the central
    meridian, meridian arc from mlfn.cpp.
  * spherical: closed form, used whenever es == 0 (PROJ forces evenden_snyder).
  * "auto": per point choice between the two ellipsoidal algorithms
    (tmerc.cpp:491-508). Both branches are evaluated and blended with
    torch.where, so the code stays branch-free.

Complex arithmetic: clenS is written with explicit real/imaginary float
tensors, exactly like the C++ (which already works on precomputed sin, cos,
sinh, cosh of the real and imaginary parts). Reasons: (1) 1:1 auditable
against tmerc.cpp, including PROJ's algebraic shortcuts for sin(2Cn),
sinh(2Ce); (2) works identically for float32 and float64 inputs with no
complex64/complex128 promotion; (3) autograd stays in the real domain (no
Wirtinger/conjugate conventions to reason about); (4) TorchInductor fuses
the real version, while for the complex version (torch 2.14) it warns
"Torchinductor does not support code generation for complex operators" and
falls back to eager kernels. tmerc_test.py checks that a torch.complex
variant agrees to 2e-18 and is differentiable too (gradcheck passes).

Branch handling: every PROJ "if" is a torch.where. Inputs of untaken or
failing branches are replaced by harmless values *before* the arithmetic
("double where") so that NaN/inf cannot reach the gradients of other points.
Failed points (PROJ's HUGE_VAL + errno) are NaN.
"""
import math

import torch

from core import HALFPI, Projection, parse_proj_string

# --------------------------------------------------------------------------
# constants (tmerc.cpp:51-63)
EPS10 = 1.0e-10
FC1 = 1.0
FC2 = 0.5
FC3 = 0.16666666666666666666
FC4 = 0.08333333333333333333
FC5 = 0.05
FC6 = 0.03333333333333333333
FC7 = 0.02380952380952380952
FC8 = 0.01785714285714285714
PROJ_ETMERC_ORDER = 6
DEG_TO_RAD = math.pi / 180.0
EXACT_CE_MAX = 2.623395162778  # tmerc.cpp:375,394 ("150 degrees")
# proj_internal.h:840 + data/proj.ini "tmerc_default_algo = poder_engsager"
DEFAULT_TMERC_ALGO = "poder_engsager"

# --------------------------------------------------------------------------
# Auxiliary latitudes, setup-time part (latitudes.cpp:157-420), plain floats.
# AuxLat enum, proj_internal.h:456-466
GEOGRAPHIC, PARAMETRIC, GEOCENTRIC, RECTIFYING, CONFORMAL, AUTHALIC = range(6)
AUX_ORDER = 6

# Only the matrices tmerc/mlfn need, copied from latitudes.cpp:243-326 and
# keyed by (auxout, auxin) like C[eta, zeta] in the C++ comments.
_AUX_COEFFS = {
    # C[phi,mu]; even coeffs only  (latitudes.cpp:249-252)
    (GEOGRAPHIC, RECTIFYING): [
        3.0 / 2.0, -27.0 / 32.0, 269.0 / 512.0, 21.0 / 16.0, -55.0 / 32.0,
        6759.0 / 4096.0, 151.0 / 96.0, -417.0 / 128.0, 1097.0 / 512.0,
        -15543.0 / 2560.0, 8011.0 / 2560.0, 293393.0 / 61440.0],
    # C[phi,chi]  (latitudes.cpp:253-259)
    (GEOGRAPHIC, CONFORMAL): [
        2.0, -2.0 / 3.0, -2.0, 116.0 / 45.0, 26.0 / 45.0, -2854.0 / 675.0,
        7.0 / 3.0, -8.0 / 5.0, -227.0 / 45.0, 2704.0 / 315.0, 2323.0 / 945.0,
        56.0 / 15.0, -136.0 / 35.0, -1262.0 / 105.0, 73814.0 / 2835.0,
        4279.0 / 630.0, -332.0 / 35.0, -399572.0 / 14175.0, 4174.0 / 315.0,
        -144838.0 / 6237.0, 601676.0 / 22275.0],
    # C[mu,phi]; even coeffs only  (latitudes.cpp:284-287)
    (RECTIFYING, GEOGRAPHIC): [
        -3.0 / 2.0, 9.0 / 16.0, -3.0 / 32.0, 15.0 / 16.0, -15.0 / 32.0,
        135.0 / 2048.0, -35.0 / 48.0, 105.0 / 256.0, 315.0 / 512.0,
        -189.0 / 512.0, -693.0 / 1280.0, 1001.0 / 2048.0],
    # C[mu,chi]  (latitudes.cpp:291-297)
    (RECTIFYING, CONFORMAL): [
        1.0 / 2.0, -2.0 / 3.0, 5.0 / 16.0, 41.0 / 180.0, -127.0 / 288.0,
        7891.0 / 37800.0, 13.0 / 48.0, -3.0 / 5.0, 557.0 / 1440.0,
        281.0 / 630.0, -1983433.0 / 1935360.0, 61.0 / 240.0, -103.0 / 140.0,
        15061.0 / 26880.0, 167603.0 / 181440.0, 49561.0 / 161280.0,
        -179.0 / 168.0, 6601661.0 / 7257600.0, 34729.0 / 80640.0,
        -3418889.0 / 1995840.0, 212378941.0 / 319334400.0],
    # C[chi,phi]  (latitudes.cpp:299-304)
    (CONFORMAL, GEOGRAPHIC): [
        -2.0, 2.0 / 3.0, 4.0 / 3.0, -82.0 / 45.0, 32.0 / 45.0, 4642.0 / 4725.0,
        5.0 / 3.0, -16.0 / 15.0, -13.0 / 9.0, 904.0 / 315.0, -1522.0 / 945.0,
        -26.0 / 15.0, 34.0 / 21.0, 8.0 / 5.0, -12686.0 / 2835.0, 1237.0 / 630.0,
        -12.0 / 5.0, -24832.0 / 14175.0, -734.0 / 315.0, 109598.0 / 31185.0,
        444337.0 / 155925.0],
    # C[chi,mu]  (latitudes.cpp:307-313)
    (CONFORMAL, RECTIFYING): [
        -1.0 / 2.0, 2.0 / 3.0, -37.0 / 96.0, 1.0 / 360.0, 81.0 / 512.0,
        -96199.0 / 604800.0, -1.0 / 48.0, -1.0 / 15.0, 437.0 / 1440.0,
        -46.0 / 105.0, 1118711.0 / 3870720.0, -17.0 / 480.0, 37.0 / 840.0,
        209.0 / 4480.0, -5569.0 / 90720.0, -4397.0 / 161280.0, 11.0 / 504.0,
        830251.0 / 7257600.0, -4583.0 / 161280.0, 108847.0 / 3991680.0,
        -20648693.0 / 638668800.0],
}


def pj_polyval(x, p, N):
    """latitudes.cpp:363-368: sum(p[i] * x^i, i=0..N) by Horner."""
    y = 0.0 if N < 0 else p[N]
    while N > 0:
        N -= 1
        y = y * x + p[N]
    return y


def pj_auxlat_coeffs(n, auxin, auxout):
    """latitudes.cpp:238-360: Fourier coefficients F[0..5] for auxin -> auxout."""
    coeffs = _AUX_COEFFS[(auxout, auxin)]
    o, d, n2, F = 0, n, n * n, []
    even = auxin <= RECTIFYING and auxout <= RECTIFYING
    for l in range(AUX_ORDER):
        m = (AUX_ORDER - l - 1) // 2 if even else (AUX_ORDER - l - 1)
        F.append(d * pj_polyval(n2 if even else n, coeffs[o:o + m + 1], m))
        o += m + 1
        d *= n
    assert o == len(coeffs)
    return F


def pj_rectifying_radius(n):
    """latitudes.cpp:415-420."""
    return pj_polyval(n * n, [1.0, 1.0 / 4, 1.0 / 64, 1.0 / 256], 3) / (1 + n)


def pj_clenshaw(szeta, czeta, F):
    """latitudes.cpp:372-383: sum(F[k] sin((2k+2) zeta)) by Clenshaw.
    Pure arithmetic: works on Python floats (setup) and tensors (runtime).
    Fixed trip count len(F) (= 6), unrolled by Python at trace time."""
    u0, u1 = 0.0, 0.0
    X = 2 * (czeta - szeta) * (czeta + szeta)  # 2 cos(2 zeta)
    for k in range(len(F) - 1, -1, -1):
        t = X * u0 - u1 + F[k]
        u1 = u0
        u0 = t
    return 2 * szeta * czeta * u0  # sin(2 zeta) * u0


def pj_auxlat_convert(zeta, F, szeta=None, czeta=None):
    """latitudes.cpp:387-398 (both 'zeta' signatures)."""
    if szeta is None:
        if isinstance(zeta, torch.Tensor):
            szeta, czeta = torch.sin(zeta), torch.cos(zeta)
        else:
            szeta, czeta = math.sin(zeta), math.cos(zeta)
    return zeta + pj_clenshaw(szeta, czeta, F)


def pj_enfn(n):
    """mlfn.cpp:9-22: [rectifying radius, phi->mu coeffs(6), mu->phi coeffs(6)]."""
    return ([pj_rectifying_radius(n)]
            + pj_auxlat_coeffs(n, GEOGRAPHIC, RECTIFYING)
            + pj_auxlat_coeffs(n, RECTIFYING, GEOGRAPHIC))


def pj_mlfn(phi, sphi, cphi, en):
    """mlfn.cpp:24-26: meridian distance (unit a)."""
    return en[0] * pj_auxlat_convert(phi, en[1:1 + AUX_ORDER], sphi, cphi)


def pj_inv_mlfn(mu, en):
    """mlfn.cpp:28-31."""
    return pj_auxlat_convert(mu / en[0], en[1 + AUX_ORDER:1 + 2 * AUX_ORDER])


def adjlon_f(lon):
    """adjlon.cpp:6-22 for setup-time floats."""
    if abs(lon) < math.pi + 1e-12:
        return lon
    lon += math.pi
    lon -= 2 * math.pi * math.floor(lon / (2 * math.pi))
    return lon - math.pi


# --------------------------------------------------------------------------
# Complex Clenshaw summation (tmerc.cpp:269-297), explicit real/imag parts.
def clenS(a, sin_arg_r, cos_arg_r, sinh_arg_i, cosh_arg_i):
    """Returns (R, I) = sum a[k] sin((2k+2) z), z = arg_r + i arg_i, given
    sin/cos of the real and sinh/cosh of the imaginary part. 5 fixed steps."""
    r = 2 * cos_arg_r * cosh_arg_i
    i = -2 * sin_arg_r * sinh_arg_i
    hr, hr1, hi, hi1 = a[-1], 0.0, 0.0, 0.0
    for k in range(len(a) - 2, -1, -1):
        hr2, hi2 = hr1, hi1
        hr1, hi1 = hr, hi
        hr = -hr2 + r * hr1 - i * hi1 + a[k]
        hi = -hi2 + i * hr1 + r * hi1
    r = sin_arg_r * cosh_arg_i
    i = cos_arg_r * sinh_arg_i
    return r * hr - i * hi, r * hi + i * hr


def _nan_where(bad, v):
    return torch.where(bad, torch.full_like(v, float("nan")), v)


# --------------------------------------------------------------------------
# Setup-time parameter checks that core.Ellipsoid does not do.
def _check_shape(params):
    """ell_set.cpp:243-343 (+ :596-604): validate the first shape key in
    PROJ's precedence order rf, f, es, e, b, and the size key."""
    for key in ("rf", "f", "es", "e", "b"):
        if key in params:
            v = float(params[key])
            ok = {"rf": v > 0, "f": 0 <= v < 1, "es": 0 <= v < 1,
                  "e": 0 <= v < 1, "b": v > 0}[key]
            if not ok:
                raise ValueError(f"invalid_op_illegal_arg_value: {key}={params[key]}")
            break
    for key in ("R", "a"):
        if key in params and not float(params[key]) > 0:
            raise ValueError(f"invalid_op_illegal_arg_value: {key}")


def _parse_to_meter(params):
    """init.cpp:693-716: to_meter=<num> or <num>/<den>."""
    if "units" in params and params["units"] != "m":
        raise NotImplementedError("only +units=m is supported by the spike")
    s = params.get("to_meter")
    if s is None:
        return 1.0
    num, _, den = str(s).partition("/")
    to_meter = float(num)
    if den:
        d = float(den)
        if d == 0.0:
            raise ValueError("invalid_op_illegal_arg_value: to_meter denominator")
        to_meter /= d
    if to_meter <= 0.0:
        raise ValueError("invalid_op_illegal_arg_value: to_meter")
    return to_meter


def get_algo_from_params(params, es, phi0, k0):
    """tmerc.cpp:557-597 (getAlgoFromParams)."""
    if params.get("approx", False):
        return "evenden_snyder"
    alg = params.get("algo")
    if alg is not None:
        if alg in ("evenden_snyder", "poder_engsager"):
            return alg
        if alg != "auto":
            raise ValueError("invalid_op_illegal_arg_value: unknown value for +algo")
        algo = "auto"
    else:
        algo = DEFAULT_TMERC_ALGO
    # Auto criterion only validated for phi0 == 0, k0 ~ 1, es <= 0.1
    if algo == "auto" and (es > 0.1 or phi0 != 0 or abs(k0 - 1) > 0.01):
        algo = "poder_engsager"
    return algo


# --------------------------------------------------------------------------
class TransverseMercator(Projection):
    """+proj=tmerc (tmerc.cpp:605-615)."""

    forced_algo = None

    def __init__(self, params):
        if isinstance(params, str):
            params = parse_proj_string(params)
        _check_shape(params)
        super().__init__(params)
        self.to_meter = _parse_to_meter(params)
        self.fr_meter = 1.0 / self.to_meter
        alpha = math.asin(self.e)  # ell_set.cpp:611 third flattening
        self.n = math.tan(alpha / 2) ** 2
        self._projection_setup(params)

    def _projection_setup(self, params):
        algo = self.forced_algo or get_algo_from_params(params, self.es, self.phi0, self.k0)
        self._setup(algo)

    def _setup(self, algo):
        """tmerc.cpp:510-555."""
        if self.es == 0:
            algo = "evenden_snyder"
        self.algo = algo
        if algo in ("evenden_snyder", "auto"):
            self._setup_approx()
        if algo in ("poder_engsager", "auto"):
            self._setup_exact()
        self.kind = "sphere" if (algo == "evenden_snyder" and self.es == 0) else algo

    def _setup_approx(self):
        """tmerc.cpp:235-249."""
        if self.es != 0.0:
            self.en = pj_enfn(self.n)
            self.ml0 = pj_mlfn(self.phi0, math.sin(self.phi0), math.cos(self.phi0), self.en)
            self.esp = self.es / (1.0 - self.es)
        else:
            self.esp = self.k0
            self.ml0 = 0.5 * self.esp

    def _setup_exact(self):
        """tmerc.cpp:452-489."""
        assert self.es > 0
        n = self.n
        self.cgb = pj_auxlat_coeffs(n, CONFORMAL, GEOGRAPHIC)
        self.cbg = pj_auxlat_coeffs(n, GEOGRAPHIC, CONFORMAL)
        self.Qn = self.k0 * pj_rectifying_radius(n)
        self.utg = pj_auxlat_coeffs(n, RECTIFYING, CONFORMAL)
        self.gtu = pj_auxlat_coeffs(n, CONFORMAL, RECTIFYING)
        Z = pj_auxlat_convert(self.phi0, self.cbg)
        self.Zb = -self.Qn * pj_auxlat_convert(Z, self.gtu)

    # ---------------- dispatch (setup-time constant, no data dependence)
    def _fwd(self, lam, phi):
        if self.kind == "sphere":
            return self._sph_fwd(lam, phi)
        if self.kind == "evenden_snyder":
            return self._approx_fwd(lam, phi)
        if self.kind == "poder_engsager":
            return self._exact_fwd(lam, phi)
        # auto_e_fwd, tmerc.cpp:491-496. Both branches are evaluated; the
        # approx branch gets a harmless input where it is not selected (double
        # where). The exact branch is regular for |lam| <= 3 deg, needs no guard.
        use_exact = lam.abs() > 3 * DEG_TO_RAD
        xe, ye = self._exact_fwd(lam, phi)
        xa, ya = self._approx_fwd(torch.where(use_exact, torch.zeros_like(lam), lam), phi)
        return torch.where(use_exact, xe, xa), torch.where(use_exact, ye, ya)

    def _inv(self, x, y):
        if self.kind == "sphere":
            return self._sph_inv(x, y)
        if self.kind == "evenden_snyder":
            return self._approx_inv(x, y)
        if self.kind == "poder_engsager":
            return self._exact_inv(x, y)
        # auto_e_inv, tmerc.cpp:498-508
        use_exact = x.abs() > 0.053 - 0.022 * y * y
        le, pe = self._exact_inv(x, y)
        zero = torch.zeros_like(x)  # double where: approx sees (0,0) where unused
        la, pa = self._approx_inv(torch.where(use_exact, zero, x), torch.where(use_exact, zero, y))
        return torch.where(use_exact, le, la), torch.where(use_exact, pe, pa)

    # ---------------- Evenden/Snyder ellipsoidal
    def _approx_fwd(self, lam, phi):
        """approx_e_fwd, tmerc.cpp:71-116."""
        bad = (lam < -HALFPI) | (lam > HALFPI)
        sinphi = torch.sin(phi)
        cosphi = torch.cos(phi)
        big = cosphi.abs() > 1e-10
        t = torch.where(big, sinphi / torch.where(big, cosphi, torch.ones_like(cosphi)), 0.0)
        t = t * t
        al = cosphi * lam
        als = al * al
        al = al / torch.sqrt(1.0 - self.es * sinphi * sinphi)
        n = self.esp * cosphi * cosphi
        x = self.k0 * al * (
            FC1 + FC3 * als * (
                1.0 - t + n + FC5 * als * (
                    5.0 + t * (t - 18.0) + n * (14.0 - 58.0 * t)
                    + FC7 * als * (61.0 + t * (t * (179.0 - t) - 479.0)))))
        y = self.k0 * (
            pj_mlfn(phi, sinphi, cosphi, self.en) - self.ml0
            + sinphi * al * lam * FC2 * (
                1.0 + FC4 * als * (
                    5.0 - t + n * (9.0 + 4.0 * n) + FC6 * als * (
                        61.0 + t * (t - 58.0) + n * (270.0 - 330 * t)
                        + FC8 * als * (1385.0 + t * (t * (543.0 - t) - 3111.0))))))
        return _nan_where(bad, x), _nan_where(bad, y)

    def _approx_inv(self, x, y):
        """approx_e_inv, tmerc.cpp:160-197."""
        phi = pj_inv_mlfn(self.ml0 + y / self.k0, self.en)
        pole = phi.abs() >= HALFPI
        phi_s = torch.where(pole, torch.zeros_like(phi), phi)  # safe input
        sinphi = torch.sin(phi_s)
        cosphi = torch.cos(phi_s)
        big = cosphi.abs() > 1e-10
        t = torch.where(big, sinphi / torch.where(big, cosphi, torch.ones_like(cosphi)), 0.0)
        n = self.esp * cosphi * cosphi
        con = 1.0 - self.es * sinphi * sinphi
        d = x * torch.sqrt(con) / self.k0
        con = con * t
        t = t * t
        ds = d * d
        phi_e = phi_s - (con * ds / (1.0 - self.es)) * FC2 * (
            1.0 - ds * FC4 * (
                5.0 + t * (3.0 - 9.0 * n) + n * (1.0 - 4 * n) - ds * FC6 * (
                    61.0 + t * (90.0 - 252.0 * n + 45.0 * t) + 46.0 * n - ds * FC8 * (
                        1385.0 + t * (3633.0 + t * (4095.0 + 1575.0 * t))))))
        lam_e = d * (
            FC1 - ds * FC3 * (
                1.0 + 2.0 * t + n - ds * FC5 * (
                    5.0 + t * (28.0 + 24.0 * t + 8.0 * n) + 6.0 * n - ds * FC7 * (
                        61.0 + t * (662.0 + t * (1320.0 + 720.0 * t)))))) / cosphi
        pole_phi = torch.where(y < 0.0, torch.full_like(phi, -HALFPI), torch.full_like(phi, HALFPI))
        return torch.where(pole, torch.zeros_like(lam_e), lam_e), torch.where(pole, pole_phi, phi_e)

    # ---------------- spherical (es == 0)
    def _sph_fwd(self, lam, phi):
        """tmerc_spherical_fwd, tmerc.cpp:118-158."""
        cosphi = torch.cos(phi)
        b = cosphi * torch.sin(lam)
        fail1 = ((b.abs() - 1.0).abs() <= EPS10)
        b = torch.where(fail1, torch.zeros_like(b), b)
        x = self.ml0 * torch.log((1.0 + b) / (1.0 - b))
        on_eq = cosphi == 1.0
        y_eq = torch.where((lam < -HALFPI) | (lam > HALFPI),
                           torch.full_like(lam, math.pi), torch.zeros_like(lam))
        yy = cosphi * torch.cos(lam) / torch.sqrt(1.0 - b * b)
        ayy = yy.abs()
        ge1 = ayy >= 1.0
        fail2 = (~on_eq) & ge1 & ((ayy - 1.0) > EPS10)
        y_ac = torch.acos(torch.where(ge1, torch.zeros_like(yy), yy))  # safe acos input
        y = torch.where(on_eq, y_eq, torch.where(ge1, torch.zeros_like(yy), y_ac))
        y = torch.where(phi < 0.0, -y, y)
        y = self.esp * (y - self.phi0)
        bad = fail1 | fail2
        return _nan_where(bad, x), _nan_where(bad, y)

    def _sph_inv(self, x, y):
        """tmerc_spherical_inv, tmerc.cpp:199-222."""
        h = torch.exp(x / self.esp)
        bad = h == 0
        h = torch.where(bad, torch.ones_like(h), h)
        g = 0.5 * (h - 1.0 / h)
        D = self.phi0 + y / self.esp  # Snyder eq. 8-8
        h = torch.cos(D)
        phi = torch.asin(torch.sqrt((1.0 - h * h) / (1.0 + g * g)))
        phi = torch.copysign(phi, D)
        both0 = (g == 0.0) & (h == 0.0)
        lam = torch.atan2(torch.where(both0, torch.zeros_like(g), g),
                          torch.where(both0, torch.ones_like(h), h))
        lam = torch.where(both0, torch.zeros_like(lam), lam)
        return _nan_where(bad, lam), _nan_where(bad, phi)

    # ---------------- Poder/Engsager ("exact")
    def _exact_fwd(self, lam, phi):
        """exact_e_fwd, tmerc.cpp:300-383."""
        # ell. LAT, LNG -> Gaussian LAT, LNG
        Cn = pj_auxlat_convert(phi, self.cbg)
        # Gaussian LAT, LNG -> compl. sph. LAT
        sin_Cn = torch.sin(Cn)
        cos_Cn = torch.cos(Cn)
        sin_Ce = torch.sin(lam)
        cos_Ce = torch.cos(lam)
        cos_Cn_cos_Ce = cos_Cn * cos_Ce
        Cn = torch.atan2(sin_Cn, cos_Cn_cos_Ce)
        denom = torch.hypot(sin_Cn, cos_Cn_cos_Ce)
        sing = denom == 0  # equator at |lam| = 90: PROJ gets tan_Ce = inf -> error
        inv_denom_tan_Ce = 1.0 / torch.where(sing, torch.ones_like(denom), denom)
        tan_Ce = sin_Ce * cos_Cn * inv_denom_tan_Ce
        # compl. sph. N, E -> ell. norm. N, E
        Ce = torch.asinh(tan_Ce)
        two_inv_denom_tan_Ce = 2 * inv_denom_tan_Ce
        two_inv_denom_tan_Ce_square = two_inv_denom_tan_Ce * inv_denom_tan_Ce
        tmp_r = cos_Cn_cos_Ce * two_inv_denom_tan_Ce_square
        sin_arg_r = sin_Cn * tmp_r
        cos_arg_r = cos_Cn_cos_Ce * tmp_r - 1
        sinh_arg_i = tan_Ce * two_inv_denom_tan_Ce
        cosh_arg_i = two_inv_denom_tan_Ce_square - 1
        dCn, dCe = clenS(self.gtu, sin_arg_r, cos_arg_r, sinh_arg_i, cosh_arg_i)
        Cn = Cn + dCn
        Ce = Ce + dCe
        ok = (Ce.abs() <= EXACT_CE_MAX) & ~sing
        y = self.Qn * Cn + self.Zb  # Northing
        x = self.Qn * Ce  # Easting
        return _nan_where(~ok, x), _nan_where(~ok, y)

    def _exact_inv(self, x, y):
        """exact_e_inv, tmerc.cpp:386-450."""
        # normalize N, E
        Cn = (y - self.Zb) / self.Qn
        Ce = x / self.Qn
        ok = Ce.abs() <= EXACT_CE_MAX  # 150 degrees
        Ce = torch.where(ok, Ce, torch.zeros_like(Ce))  # safe input (exp overflow)
        # norm. N, E -> compl. sph. LAT, LNG
        sin_arg_r = torch.sin(2 * Cn)
        cos_arg_r = torch.cos(2 * Cn)
        exp_2_Ce = torch.exp(2 * Ce)
        half_inv_exp_2_Ce = 0.5 / exp_2_Ce
        sinh_arg_i = 0.5 * exp_2_Ce - half_inv_exp_2_Ce
        cosh_arg_i = 0.5 * exp_2_Ce + half_inv_exp_2_Ce
        dCn, dCe = clenS(self.utg, sin_arg_r, cos_arg_r, sinh_arg_i, cosh_arg_i)
        Cn = Cn + dCn
        Ce = Ce + dCe
        # compl. sph. LAT -> Gaussian LAT, LNG
        sin_Cn = torch.sin(Cn)
        cos_Cn = torch.cos(Cn)
        sinhCe = torch.sinh(Ce)
        Ce = torch.atan2(sinhCe, cos_Cn)
        modulus_Ce = torch.hypot(sinhCe, cos_Cn)
        rr = torch.hypot(sin_Cn, modulus_Ce)
        Cn = torch.atan2(sin_Cn, modulus_Ce)
        # Gaussian LAT, LNG -> ell. LAT, LNG
        phi = pj_auxlat_convert(Cn, self.cgb, sin_Cn / rr, modulus_Ce / rr)
        return _nan_where(~ok, Ce), _nan_where(~ok, phi)

    # ---------------- to_meter (fwd.cpp:162-165, inv.cpp:75-79)
    def forward(self, coo):
        out = super().forward(coo)
        if self.fr_meter != 1.0:  # setup-time constant
            k = min(out.shape[-1], 3)  # x, y and z (vto_meter defaults to to_meter)
            out = torch.cat([out[..., :k] * self.fr_meter, out[..., k:]], -1)
        return out

    def inverse(self, coo):
        if self.to_meter != 1.0:
            k = min(coo.shape[-1], 3)
            coo = torch.cat([coo[..., :k] * self.to_meter, coo[..., k:]], -1)
        return super().inverse(coo)


class ExtendedTransverseMercator(TransverseMercator):
    """+proj=etmerc (tmerc.cpp:617-625): always Poder/Engsager, ellipsoid only."""

    forced_algo = "poder_engsager"

    def _projection_setup(self, params):
        if self.es == 0.0:
            raise ValueError("invalid_op_illegal_arg_value: eccentricity should not be zero")
        self._setup("poder_engsager")


class UTM(TransverseMercator):
    """+proj=utm (tmerc.cpp:630-672)."""

    def _projection_setup(self, params):
        if self.es == 0.0:
            raise ValueError("invalid_op_illegal_arg_value: eccentricity should not be zero")
        if self.lam0 < -1000.0 or self.lam0 > 1000.0:
            raise ValueError("invalid_op_illegal_arg_value: lon_0")
        self.y0 = 10000000.0 if params.get("south", False) else 0.0
        self.x0 = 500000.0
        if "zone" in params:
            s = str(params["zone"])
            zone = int(s) if (s.isdigit()) else 0  # param.cpp:180-189 digits only
            if 0 < zone <= 60:
                zone -= 1
            else:
                raise ValueError("invalid_op_illegal_arg_value: zone")
        else:  # nearest central meridian
            zone = int(round(math.floor((adjlon_f(self.lam0) + math.pi) * 30.0 / math.pi)))
            zone = min(max(zone, 0), 59)
        self.zone = zone + 1
        self.lam0 = (zone + 0.5) * math.pi / 30.0 - math.pi
        self.k0 = 0.9996
        self.phi0 = 0.0
        self._setup(get_algo_from_params(params, self.es, self.phi0, self.k0))


def make(definition):
    """Build the torch projection for a '+proj=tmerc|etmerc|utm ...' string."""
    params = parse_proj_string(definition) if isinstance(definition, str) else dict(definition)
    cls = {"tmerc": TransverseMercator, "etmerc": ExtendedTransverseMercator,
           "utm": UTM}[params.get("proj")]
    return cls(params)
