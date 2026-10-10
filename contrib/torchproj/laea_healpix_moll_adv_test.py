"""ADVERSARIAL verification of laea_healpix_moll.py (laea / healpix / rhealpix / moll /
wag4 / wag5) against PROJ 9.8.1 (pyproj 3.8) and the C++ source in this PROJ tree.

Independent of the implementer's test: own gie runner, own pyproj comparison (different
seed, radians fed straight to PROJ with radians=True, no unitconvert step), edge-case
attacks, finite-difference Jacobians, NaN-gradient scans, host-sync / compile checks.

Run:  python laea_healpix_moll_adv_test.py [--n 20000] [--no-compile]
Prints findings; exit code is always 0 (this is a report, not a gate).
"""
import ast
import inspect
import math
import sys
import time

import numpy as np
import torch
from pyproj import Transformer
from pyproj.enums import TransformDirection as TD

import testing
import laea_healpix_moll as M

torch.set_num_threads(1)
N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 20000
DO_COMPILE = "--no-compile" not in sys.argv
H = math.pi / 2
FINDINGS = []
NANIN = [0]
SUSPECT = []


def finding(sev, msg):
    FINDINGS.append((sev, msg))
    print(f"  !! [{sev}] {msg}")


def hdr(s):
    print("=" * 100)
    print(s)


# ----------------------------------------------------------------------------- refs
_TC = {}


def tr(defn):
    if defn not in _TC:
        _TC[defn] = Transformer.from_pipeline(defn)
    return _TC[defn]


def _nanify(a, b):
    a = np.array(a, dtype=float, copy=True)
    b = np.array(b, dtype=float, copy=True)
    bad = ~(np.isfinite(a) & np.isfinite(b))
    a[bad] = np.nan
    b[bad] = np.nan
    return a, b


def ref_fwd(defn, lam, phi):
    """PROJ forward, RADIANS in (no unitconvert step), metres out; failure -> NaN."""
    return _nanify(*tr(defn).transform(np.asarray(lam, float), np.asarray(phi, float), radians=True))


def ref_inv(defn, x, y):
    """PROJ inverse, metres in, RADIANS out; failure -> NaN."""
    return _nanify(*tr(defn).transform(np.asarray(x, float), np.asarray(y, float), radians=True,
                                       direction=TD.INVERSE))


def port_fwd(P, lam, phi, dtype=torch.float64):
    inp = torch.tensor(np.stack([np.asarray(lam, float), np.asarray(phi, float)], -1),
                       dtype=torch.float64).to(dtype)
    with torch.no_grad():
        o = P.forward(inp).double().numpy()
    return o[..., 0], o[..., 1]


def port_inv(P, x, y, dtype=torch.float64):
    inp = torch.tensor(np.stack([np.asarray(x, float), np.asarray(y, float)], -1),
                       dtype=torch.float64).to(dtype)
    with torch.no_grad():
        o = P.inverse(inp).double().numpy()
    return o[..., 0], o[..., 1]


def ang_err(P, l1, p1, l2, p2):
    return testing.angular_error_m(np.degrees(l1), np.degrees(p1), np.degrees(l2), np.degrees(p2),
                                   a=P.a_ellps)


def at_adjlon_threshold(P, lam):
    """lam - lon_0 lands exactly on +-(pi + 1e-12) or outside [-pi, pi]: core.adjlon ('<=', single pass)
    and PROJ (fwd.cpp:72-82 + adjlon.cpp:10 '<') wrap differently there (see 3b/3h)."""
    d = np.abs(np.asarray(lam) - P.lam0)
    return (np.abs(np.asarray(lam)) > math.pi) | (d >= math.pi + 1e-12 - 1e-15)


def fwd_stats(P, defn, lam, phi, dtype=torch.float64):
    X, Y = ref_fwd(defn, lam, phi)
    x, y = port_fwd(P, lam, phi, dtype)
    pf, tf = np.isnan(X), np.isnan(x) | np.isnan(y)
    both = ~pf & ~tf
    e = np.hypot(x[both] - X[both], y[both] - Y[both])
    return dict(max=float(e.max()) if e.size else 0.0, p50=float(np.median(e)) if e.size else 0.0,
                mism=int((pf != tf).sum()), mism_idx=np.nonzero(pf != tf)[0],
                worst=int(np.nonzero(both)[0][np.argmax(e)]) if e.size else -1, n=len(lam))


def inv_stats(P, defn, x, y, dtype=torch.float64, zone_deg=1e-2, centre=None):
    L, Ph = ref_inv(defn, x, y)
    l, ph = port_inv(P, x, y, dtype)
    pf, tf = np.isnan(L), np.isnan(l) | np.isnan(ph)
    both = ~pf & ~tf
    e = np.full(len(x), np.nan)
    e[both] = ang_err(P, l[both], ph[both], L[both], Ph[both])
    # "singular zone": within zone_deg of a pole or of the antipode of an azimuthal centre
    zone = np.abs(np.degrees(np.abs(Ph)) - 90) < zone_deg
    if centre is not None:
        lc, pc = centre
        cosd = (np.sin(Ph) * np.sin(-pc) + np.cos(Ph) * np.cos(-pc) * np.cos(L - (lc + math.pi)))
        zone |= np.degrees(np.arccos(np.clip(cosd, -1, 1))) < zone_deg
    zone &= both
    reg = both & ~zone
    return dict(max=float(np.nanmax(e)) if both.any() else 0.0,
                reg=float(e[reg].max()) if reg.any() else 0.0,
                sing=float(e[zone].max()) if zone.any() else 0.0,
                mism=int((pf != tf).sum()), mism_idx=np.nonzero(pf != tf)[0], n=len(x))


# ============================================================================= 1. gie
def run_gie():
    hdr("1. Independent gie runner (testing.gie_cases) + every gie input re-checked against PROJ")
    nb = nc = npass = 0
    worst_vs_proj = 0.0
    fails = []
    for name in ("laea", "healpix", "rhealpix", "moll", "wag4", "wag5"):
        for blk in testing.gie_cases(name):
            nb += 1
            op = blk["operation"]
            defn = " ".join(t if t.startswith("+") else "+" + t for t in op.split())
            try:
                P = M.make(op)
            except Exception as err:   # setup failure: must match PROJ's
                try:
                    tr(defn)
                    proj_ok = True
                except Exception:
                    proj_ok = False
                nc += 1
                ok = not proj_ok
                npass += ok
                print(f"  {blk['file']:<18} {op:<60} setup error ({err}); PROJ setup ok={proj_ok}")
                continue
            k = 0
            for c in blk["cases"]:
                nc += 1
                acc = c["accept"]
                if c["direction"] == "fwd":
                    lam, phi = math.radians(acc[0]), math.radians(acc[1])
                    x, y = port_fwd(P, [lam], [phi])
                    X, Y = ref_fwd(defn, [lam], [phi])
                    if c["expect"] == "failure":
                        ok = bool(np.isnan(x[0]))
                    elif np.isnan(c["expect"][:2]).all():
                        ok = bool(np.isnan(x[0]) and np.isnan(y[0]))
                    else:
                        d = math.hypot(x[0] - c["expect"][0], y[0] - c["expect"][1])
                        ok = d <= c["tolerance_m"]
                    if np.isfinite(X[0]) and np.isfinite(x[0]):
                        worst_vs_proj = max(worst_vs_proj, math.hypot(x[0] - X[0], y[0] - Y[0]))
                    elif np.isfinite(X[0]) != np.isfinite(x[0]):
                        fails.append(f"{op}: fwd {acc} failure mismatch vs PROJ")
                else:
                    l, p = port_inv(P, [acc[0]], [acc[1]])
                    L, Pp = ref_inv(defn, [acc[0]], [acc[1]])
                    if c["expect"] == "failure":
                        ok = bool(np.isnan(l[0]))
                    elif np.isnan(c["expect"][:2]).all():
                        ok = bool(np.isnan(l[0]) and np.isnan(p[0]))
                    else:
                        d = float(testing.angular_error_m(math.degrees(l[0]), math.degrees(p[0]),
                                                          c["expect"][0], c["expect"][1], a=P.a_ellps))
                        ok = d <= c["tolerance_m"]
                    if np.isfinite(L[0]) and np.isfinite(l[0]):
                        worst_vs_proj = max(worst_vs_proj, float(ang_err(P, l, p, L, Pp)[0]))
                    elif np.isfinite(L[0]) != np.isfinite(l[0]):
                        fails.append(f"{op}: inv {acc} failure mismatch vs PROJ")
                npass += ok
                k += ok
                if not ok:
                    fails.append(f"{op}: {c['direction']} {acc} expect {c['expect']}")
            for r in blk["roundtrips"]:
                nc += 1
                acc = r["accept"]
                lp = torch.tensor([[math.radians(acc[0]), math.radians(acc[1])]], dtype=torch.float64)
                xy = P.forward(lp)
                for _ in range(r["n"] - 1):
                    xy = P.forward(P.inverse(xy))
                back = P.inverse(xy)[0].numpy()
                if np.isnan(acc[:2]).all():
                    ok = bool(np.isnan(back[:2]).all())
                else:
                    d = float(testing.angular_error_m(acc[0], acc[1], math.degrees(back[0]),
                                                      math.degrees(back[1]), a=P.a_ellps))
                    ok = d <= r["tolerance_m"]
                npass += ok
                k += ok
                if not ok:
                    fails.append(f"{op}: roundtrip {acc}")
            print(f"  {blk['file']:<18} {op:<60} {k}/{len(blk['cases']) + len(blk['roundtrips'])}")
    print(f"  gie blocks={nb} cases={nc} passed={npass}; max |port - PROJ| on gie inputs = {worst_vs_proj:.2e} m")
    for f in fails:
        finding("gie", f)
    return dict(blocks=nb, cases=nc, passed=npass, worst_vs_proj=worst_vs_proj)


# ============================================================================= 2. headline
CLAIMS = {  # definition: (fwd64, inv64 overall, fwd32, inv32) from the implementer report
    "+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80": (1.6e-7, 4e-6, 6100, 2300),
    "+proj=laea +ellps=GRS80 +lon_0=-30": (7e-8, 2.1e-7, 42000, 2200),
    "+proj=laea +lat_0=90 +lon_0=-150 +datum=WGS84": (8.4e-9, 5.7e-9, 1400, 4600),
    "+proj=laea +lat_0=-90 +lon_0=0 +ellps=GRS80": (1e-8, 5.7e-9, 650, 4000),
    "+proj=laea +a=6378137 +rf=30 +lat_0=40": (7e-6, 0.011, 12000, 2300),
    "+proj=laea +R=6371000 +lat_0=45 +lon_0=10": (8e-7, 0.095, 7300, 2200),
    "+proj=laea +R=6371000": (1.1e-7, 5.4e-8, 19000, 1800),
    "+proj=laea +R=6371000 +lat_0=90": (2.6e-9, 1.6e-4, 2.6, 5300),
    "+proj=laea +R=6371000 +lat_0=-90 +lon_0=20": (2.6e-9, 6.5e-7, 7.7, 4500),
    "+proj=healpix +ellps=GRS80": (1.7e-8, 5.7e-9, 1800, 1500),
    "+proj=healpix +R=6400000 +rot_xy=42 +lon_0=30": (4.2e-8, 5.7e-9, 2000, 1500),
    "+proj=healpix +ellps=WGS84 +rot_xy=-15 +x_0=1000 +y_0=-2000": (3.4e-8, 5.7e-9, 1500, 1400),
    "+proj=healpix +a=6378137 +rf=30": (6.7e-6, 1.2e-5, 1900, 1300),
    "+proj=rhealpix +ellps=GRS80": (2.8e-8, 5.7e-9, 2000, 1500),
    "+proj=rhealpix +south_square=2 +north_square=3 +ellps=WGS84": (2.2e-8, 5.7e-9, 1900, 1500),
    "+proj=rhealpix +R=6400000 +north_square=1 +south_square=0 +lon_0=-40": (1.5e-8, 5.7e-9, 2000, 1500),
    "+proj=moll +ellps=GRS80": (3e-7, 2e-8, 1.5e7, 1200),
    "+proj=moll +R=6371000 +lon_0=10 +x_0=500": (1.2e-7, 3.7e-8, 1.5e7, 1200),
    "+proj=wag4 +a=6400000": (1.1e-8, 5.2e-7, 1.5e7, 2700),
    "+proj=wag5 +a=6400000": (1e-8, 6.1e-8, 1.5e7, 2100),
}


def centre_of(P):
    if isinstance(P, M.LambertAzimuthalEqualArea):
        return (P.lam0, P.phi0)
    return None


def special_points(P, wrap=False):
    lam0 = P.lam0
    lats = [0.0, H, -H, H - 1e-12, -H + 1e-12, H - 1e-9, -H + 1e-9, math.asin(2 / 3),
            -math.asin(2 / 3), math.radians(41.8103148958), 0.7297276562269663]
    lons = [-math.pi, math.pi, -math.pi + 1e-12, math.pi - 1e-12, 0.0, -H, H, lam0 + math.pi,
            lam0 - math.pi, lam0 + math.pi - 1e-12, lam0 - math.pi + 1e-12, 1e-300, -1e-300]
    lo, la = np.meshgrid(lons, lats)
    lo = lo.ravel()
    if wrap:   # keep lon_0+-180 inside [-pi, pi] (values > pi are attacked separately in 3b)
        lo = np.where(lo > math.pi, lo - 2 * math.pi, np.where(lo < -math.pi, lo + 2 * math.pi, lo))
    return lo, la.ravel()


def run_headline(rng):
    hdr(f"2. Headline fp64/fp32 numbers re-derived ({N} random pts + special pts, seed differs from implementer)")
    print(f"  {'definition':<70} {'fwd64':>8} {'claim':>8} {'inv64':>8} {'reg':>8} {'claim':>8} "
          f"{'mism f/i':>8} {'fwd32':>8} {'claim':>8} {'inv32':>8} {'claim':>8}")
    out = {}
    for defn, (cf64, ci64, cf32, ci32) in CLAIMS.items():
        P = M.make(defn)
        sl, sp = special_points(P, wrap=True)
        lam = np.concatenate([rng.uniform(-math.pi, math.pi, N), sl])
        phi = np.concatenate([np.arcsin(rng.uniform(-1, 1, N)), sp])   # area-uniform this time
        f64 = fwd_stats(P, defn, lam, phi)
        f32 = fwd_stats(P, defn, lam[:N], phi[:N], torch.float32)
        # inverse: PROJ images of fresh points + a box 15% bigger than the image (failure masks)
        l2 = rng.uniform(-math.pi, math.pi, N)
        p2 = np.arcsin(rng.uniform(-1, 1, N))
        X, Y = ref_fwd(defn, np.concatenate([l2, sl]), np.concatenate([p2, sp]))
        ok = np.isfinite(X)
        X, Y = X[ok], Y[ok]
        cx, cy = 0.5 * (X.max() + X.min()), 0.5 * (Y.max() + Y.min())
        wx, wy = 0.575 * (X.max() - X.min()), 0.575 * (Y.max() - Y.min())
        bx = rng.uniform(cx - wx, cx + wx, N // 2)
        by = rng.uniform(cy - wy, cy + wy, N // 2)
        i64 = inv_stats(P, defn, np.concatenate([X, bx]), np.concatenate([Y, by]), centre=centre_of(P))
        i32 = inv_stats(P, defn, X[:N], Y[:N], torch.float32)
        out[defn] = dict(f64=f64, i64=i64, f32=f32, i32=i32)
        print(f"  {defn:<70} {f64['max']:8.1e} {cf64:8.1e} {i64['max']:8.1e} {i64['reg']:8.1e} {ci64:8.1e} "
              f"{f64['mism']:>4}/{i64['mism']:<4}{f32['max']:8.1e} {cf32:8.1e} {i32['max']:8.1e} {ci32:8.1e}")
        if f64["mism"] or i64["mism"]:
            idx = f64["mism_idx"][:3]
            finding("mask", f"{defn}: fp64 failure-mask mismatches fwd={f64['mism']} inv={i64['mism']} "
                    f"(fwd examples lam,phi={[(lam[i], phi[i]) for i in idx]})")
        if f64["max"] > max(10 * cf64, 2e-5):
            w = f64["worst"]
            keep = ~at_adjlon_threshold(P, lam)
            f64b = fwd_stats(P, defn, lam[keep], phi[keep])
            print(f"      worst at lam={lam[w]!r} (lam-lon_0={lam[w] - P.lam0!r}): adjlon-threshold point; "
                  f"max excluding such points {f64b['max']:.1e} m")
            if f64b["max"] > max(10 * cf64, 2e-5):
                finding("accuracy", f"{defn}: fwd fp64 max {f64b['max']:.2e} m >> claimed {cf64:.1e}")
        if i64["reg"] > 2e-5:
            finding("accuracy", f"{defn}: inv fp64 max outside pole/antipode zone {i64['reg']:.2e} m")
    return out


# ============================================================================= 3. attacks
def cmp_points(defn, lam, phi, label, P=None, tol=1e-6, quiet=False):
    """Forward at given radian points: value error + failure-mask agreement."""
    P = P or M.make(defn)
    s = fwd_stats(P, defn, np.asarray(lam, float), np.asarray(phi, float))
    bad = s["mism"] > 0 or s["max"] > tol
    if not quiet or bad:
        print(f"  {'BAD ' if bad else 'ok  '}{label:<48} {defn:<55} max {s['max']:.2e} m  mask mismatches {s['mism']}/{s['n']}")
    return s


def run_attacks(rng):
    hdr("3. Edge-case attacks vs PROJ (radian inputs passed unchanged to PROJ)")
    defs = list(CLAIMS)
    # ---- 3a poles / latitude clamp threshold
    print(" 3a. poles: exactly +-pi/2, +-(pi/2-1e-12), +-(pi/2+k*1e-13) (PROJ: clamp if phi-pi/2 <= 1e-12 else error)")
    lat_c = [H, -H, H - 1e-12, -H + 1e-12, math.radians(90 - 1e-12), math.radians(-90 + 1e-12),
             H + 5e-13, -H - 5e-13, H + 1e-12, -(H + 1e-12), H + 1.0000001e-12, H + 2e-12, -H - 2e-12]
    lon_c = [0.0, 1.0, -2.5, math.pi, -math.pi, 0.3]
    lo, la = np.meshgrid(lon_c, lat_c)
    for defn in defs:
        s = cmp_points(defn, lo.ravel(), la.ravel(), "poles / lat clamp", tol=1e-6, quiet=True)
        if s["mism"]:
            ex = [(float(lo.ravel()[i]), float(la.ravel()[i])) for i in s["mism_idx"][:4]]
            finding("core/lat-check", f"{defn}: {s['mism']} pole-threshold points where PROJ fails and port "
                    f"does not (or v.v.), e.g. {ex}")
            break   # same root cause for all definitions; report once
    else:
        print("   all definitions agree at the poles and on the 1e-12 clamp threshold")
    for defn in defs:
        s = fwd_stats(M.make(defn), defn, lo.ravel()[:36], la.ravel()[:36])
        if s["max"] > 1e-6:
            finding("pole", f"{defn}: forward at/near poles differs by {s['max']:.2e} m")
    # ---- 3b longitudes outside [-180,180], lon_0 +- 180, +over, |lon| > 10 rad
    print(" 3b. longitudes outside [-pi,pi] (0..360 grids), lon_0+-180 written as lon>180, |lon|>10 rad, +over")
    lond = np.array([-540, -360, -190, -180.0000001, 180.0000001, 190, 200, 270, 359.9, 360, 540,
                     572.9, 573.0, 600, 700.0])
    latd = np.array([0.0, 10.0, -35.0, 60.0])
    lo, la = np.meshgrid(np.radians(lond), np.radians(latd))
    for defn in ["+proj=moll +R=6371000 +lon_0=10", "+proj=healpix +R=6371000 +lon_0=10",
                 "+proj=laea +R=6371000 +lat_0=30 +lon_0=10", "+proj=wag4 +R=6371000",
                 "+proj=moll +R=6371000 +lon_0=10 +over", "+proj=laea +R=6371000 +over"]:
        s = cmp_points(defn, lo.ravel(), la.ravel(), "lon outside [-180,180]", tol=1e-6)
        if s["mism"] or s["max"] > 1e-6:
            w = s["worst"]
            finding("core/lon", f"{defn}: lon outside [-180,180]: max err {s['max']:.3e} m "
                    f"(at lon={np.degrees(lo.ravel()[w]):.4f} deg), mask mismatches {s['mism']} "
                    f"(PROJ errors for |lon| > 10 rad; port returns values)")
    # lon_0 + 180 written as lon > 180: interrupted projections flip side
    d = "+proj=moll +R=6371000 +lon_0=10"
    X, _ = ref_fwd(d, [math.radians(190.0)], [0.0])
    x, _ = port_fwd(M.make(d), [math.radians(190.0)], [0.0])
    print(f"   moll lon_0=10, lon=190 deg, lat=0: PROJ x={X[0]:.1f}  port x={x[0]:.1f}")
    if abs(X[0] - x[0]) > 1:
        finding("core/lon", f"lon = lon_0+180 given as a value > 180 deg (0..360 grids) lands on the "
                f"OPPOSITE edge of interrupted maps: moll lon_0=10 lon=190: PROJ x={X[0]:.0f}, port x={x[0]:.0f} "
                "(core.forward does one adjlon after subtracting lon_0; fwd.cpp:80-82 adjlon()s the raw "
                "longitude first, then again after -lam0)")
    # ---- 3c k_0, x_0/y_0, unusual ellipsoids, EPS thresholds
    print(" 3c. k_0, x_0/y_0, unusual ellipsoids, parameters at PROJ's EPS thresholds")
    extra = [
        "+proj=laea +lat_0=52 +lon_0=10 +k_0=0.9996 +ellps=GRS80",
        "+proj=laea +R=6371000 +lat_0=-33 +k_0=2",
        "+proj=healpix +k_0=3 +ellps=WGS84 +x_0=-5e5 +y_0=7e5",
        "+proj=moll +k_0=2 +R=1000000 +x_0=10 +y_0=-20",
        "+proj=laea +R=1 +lat_0=30",
        "+proj=laea +a=6378137 +b=6356752.314245 +lat_0=30",
        "+proj=laea +ellps=intl +lat_0=-30 +lon_0=170",
        "+proj=laea +ellps=clrk66 +lat_0=60 +lon_0=-100",
        "+proj=laea +ellps=bessel +lat_0=-89.999999999",      # 1.7e-11 rad from pole -> S_POLE mode
        "+proj=laea +ellps=GRS80 +lat_0=89.99999999",          # 1.7e-10 rad from pole -> OBLIQ (nearly polar)
        "+proj=laea +R=6371000 +lat_0=89.99999999",            # sphere, OBLIQ at the EPS10 edge
        "+proj=laea +R=6371000 +lat_0=89.999999999",           # sphere, N_POLE by EPS10
        "+proj=laea +R=6371000 +lat_0=1e-9",                   # EQUIT by EPS10
        "+proj=laea +ellps=GRS80 +lat_0=-1e-8",                # OBLIQ, nearly equatorial
        "+proj=laea +a=6378137 +rf=50.49",                     # n just above 0.01 -> asin(q/qp) + Newton
        "+proj=laea +a=6378137 +rf=50.51 +lat_0=20",           # n just below 0.01 -> series
        "+proj=laea +a=6378137 +rf=3 +lat_0=-60",              # very flat: Newton authalic
        "+proj=laea +a=6378137 +es=1e-14 +lat_0=45",           # e = 1e-7 (pj_authalic_lat_q threshold)
        "+proj=laea +a=6378137 +es=1e-15 +lat_0=45",           # e < 1e-7 -> q = 2 sin(phi)
        "+proj=laea +a=6378137 +f=0.003 +lat_0=-90",
        "+proj=laea +ellps=sphere +lat_0=10",
        "+proj=healpix +ellps=intl +lon_0=180",
        "+proj=healpix +a=6378137 +rf=3",
        "+proj=healpix +R=1 +rot_xy=180 +x_0=3",
        "+proj=rhealpix +ellps=clrk66 +south_square=3 +north_square=0",
        "+proj=rhealpix +a=6378137 +rf=50.49 +north_square=2",
        "+proj=moll +ellps=WGS84 +lon_0=-170 +x_0=1e6 +y_0=-1e6",
        "+proj=wag4 +R=1",
        "+proj=wag5 +ellps=intl +lon_0=33",
    ]
    for defn in extra:
        try:
            P = M.make(defn)
            tr(defn)
        except Exception as err:
            print(f"  skip {defn}: setup error {err}")
            continue
        sl, sp = special_points(P)
        lam = np.concatenate([rng.uniform(-math.pi, math.pi, N // 4), sl])
        phi = np.concatenate([np.arcsin(rng.uniform(-1, 1, N // 4)), sp])
        keep = ~at_adjlon_threshold(P, lam)          # those are attacked in 3b/3h
        lam, phi = lam[keep], phi[keep]
        f = fwd_stats(P, defn, lam, phi)
        X, Y = ref_fwd(defn, lam, phi)
        ok = np.isfinite(X)
        i = inv_stats(P, defn, X[ok], Y[ok], centre=centre_of(P))
        scale = P.a_ellps
        flag = f["mism"] or i["mism"] or f["max"] > 1e-6 * max(scale / 6.4e6, 1e-6) * 10 or i["reg"] > 2e-5 * max(scale / 6.4e6, 1e-6)
        print(f"  {'BAD ' if flag else 'ok  '}{defn:<64} fwd {f['max']:.1e} m (mism {f['mism']}) | "
              f"inv {i['reg']:.1e} m reg, {i['sing']:.1e} m pole/antipode zone (mism {i['mism']})")
        if flag and "89.99999999" not in defn:
            w = f["worst"]
            finding("accuracy", f"{defn}: fwd max {f['max']:.2e} m at lam={lam[w]!r} phi={phi[w]!r}, "
                    f"mask mism fwd {f['mism']} inv {i['mism']}, inv reg {i['reg']:.2e}")
        elif flag:
            print("      -> near-polar OBLIQ sphere: PROJ's inverse itself is ill-conditioned here, see 5f")
    # setup failures
    print(" 3d. setup failures must match PROJ")
    for defn in ["+proj=laea +lat_0=90.00000001 +R=1", "+proj=laea +lat_0=-91 +ellps=GRS80",
                 "+proj=laea +lat_0=90.000000001 +R=1", "+proj=rhealpix +north_square=4 +R=1",
                 "+proj=rhealpix +south_square=-1 +R=1", "+proj=healpix +rot_xy=1000 +R=1",
                 "+proj=laea +a=6378137 +rf=0.5", "+proj=laea +a=1 +es=1",
                 "+proj=laea +ellps=GRS80 +lat_0=90.000000001", "+proj=laea +R=1 +lat_0=-90.0000000001",
                 "+proj=moll +R=1 +lat_0=95", "+proj=healpix +R=1 +lat_0=-100"]:
        try:
            tr(defn)
            pok = True
        except Exception:
            pok = False
        try:
            M.make(defn)
            mok = True
        except Exception:
            mok = False
        print(f"  {'ok  ' if pok == mok else 'BAD '}{defn:<50} PROJ setup ok={pok}  port setup ok={mok}")
        if pok != mok:
            finding("setup", f"{defn}: PROJ setup ok={pok} but port ok={mok}")
    # ---- 3e out-of-domain: dense grids over a box larger than the image, exact failure masks
    print(" 3e. out-of-domain: failure masks on dense inverse grids (incl. exact image-boundary values)")
    for defn in defs + ["+proj=laea +R=1 +lat_0=30", "+proj=healpix +R=1", "+proj=rhealpix +R=1",
                        "+proj=moll +R=1", "+proj=wag5 +R=1"]:
        P = M.make(defn)
        a = P.a  # authalic a for healpix
        # grid with exact multiples of pi/4 * a (polygon vertices / edges) and around 2R (laea rim)
        g = np.concatenate([np.linspace(-4.2, 4.2, 211), np.arange(-5, 6) * math.pi / 4,
                            [-2.0, 2.0, -math.pi, math.pi, np.nextafter(2.0, 3), np.nextafter(-2, -3)]])
        gx, gy = np.meshgrid(g * a + P.x0, g * a + P.y0)
        s = inv_stats(P, defn, gx.ravel(), gy.ravel(), centre=centre_of(P))
        flag = s["mism"] > 0
        print(f"  {'BAD ' if flag else 'ok  '}{defn:<72} grid {s['n']} pts: mask mismatches {s['mism']}, "
              f"max err {s['max']:.1e} m (reg {s['reg']:.1e})")
        if flag:
            idx = s["mism_idx"][:3]
            finding("mask", f"{defn}: inverse failure mask differs at {s['mism']} grid points, e.g. "
                    f"{[(gx.ravel()[i], gy.ravel()[i]) for i in idx]}")
    # forward around the laea antipode ring where |b| ~ EPS10 (decision depends on last-ulp rounding)
    for defn in ["+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80",
                 "+proj=laea +R=6371000 +lat_0=45 +lon_0=10", "+proj=laea +R=6371000",
                 "+proj=laea +ellps=GRS80 +lon_0=-30", "+proj=laea +R=6371000 +lat_0=90",
                 "+proj=laea +ellps=GRS80 +lat_0=-90"]:
        P = M.make(defn)
        alon, alat = P.lam0 + math.pi, -P.phi0
        rr = np.concatenate([np.geomspace(1e-9, 1e-3, 400), [0.0]])
        az = rng.uniform(0, 2 * math.pi, len(rr))
        lam = alon + rr * np.cos(az) / max(math.cos(alat), 1e-3)
        phi = np.clip(alat + rr * np.sin(az), -H, H)
        lam = np.where(lam > math.pi, lam - 2 * math.pi, lam)
        s = cmp_points(defn, lam, phi, "ring around antipode (|b|~EPS10)", tol=1e9)
        if s["mism"]:
            finding("mask", f"{defn}: {s['mism']} forward failure-mask mismatches near the antipode")
    # ---- 3f NaN / inf / huge inputs; z,t passthrough semantics vs proj_trans
    print(" 3f. NaN / inf inputs, NaN in z/t (trans.cpp:352 makes ALL components NaN), z/t on failed points")
    defn = "+proj=laea +R=6371000 +lat_0=45"
    P = M.make(defn)
    cases = [[float("nan"), 0.5, 1.0, 2.0], [0.3, float("nan"), 1.0, 2.0], [0.3, 0.5, float("nan"), 2.0],
             [0.3, 0.5, 1.0, float("nan")], [float("inf"), 0.5, 1.0, 2.0], [0.3, float("inf"), 1.0, 2.0],
             [0.3, 3.0, 1.0, 2.0]]
    t = tr(defn)
    for c in cases:
        ref = np.array(t.transform(*[np.array([v]) for v in c], radians=True)).ravel()
        got = P.forward(torch.tensor([c], dtype=torch.float64))[0].numpy()
        same = np.array_equal(np.isfinite(ref), np.isfinite(got))
        print(f"  {'ok  ' if same else 'DIFF'} in={c}  PROJ={ref}  port={got}")
        if not same:
            finding("nan-rule", f"proj_trans semantics differ for input {c}: PROJ {ref} vs port {got} "
                    "(module's NaN rule only looks at x/y; z/t are passed through even when NaN or failed)")


# ============================================================================= 4. shapes
def run_shapes():
    hdr("4. Batched shapes, z/t passthrough, float32 inputs, requires_grad inputs, empty/1-D inputs")
    g = torch.Generator().manual_seed(7)
    for defn, kw in [("+proj=laea +ellps=GRS80 +lat_0=52 +lon_0=10", {}), ("+proj=healpix +ellps=WGS84 +rot_xy=7", {}),
                     ("+proj=rhealpix +R=1 +north_square=1", {}), ("+proj=moll +R=1", {"method": "implicit"}),
                     ("+proj=laea +a=6378137 +rf=30 +lat_0=-90", {})]:
        P = M.make(defn, **kw)
        B, Hh, W = 2, 3, 5
        lam = (torch.rand(B, Hh, W, generator=g, dtype=torch.float64) - 0.5) * 2 * math.pi
        phi = (torch.rand(B, Hh, W, generator=g, dtype=torch.float64) - 0.5) * math.pi
        coo = torch.stack([lam, phi], -1)
        o4 = P.forward(coo)
        o1 = P.forward(coo.reshape(-1, 2)).reshape(B, Hh, W, 2)
        same = torch.equal(torch.nan_to_num(o4, 1.0), torch.nan_to_num(o1, 1.0))
        i4 = P.inverse(o4)
        z = torch.randn(B, Hh, W, 2, generator=g, dtype=torch.float64)
        o5 = P.forward(torch.cat([coo, z], -1))
        zpass = torch.equal(o5[..., 2:], z) and torch.equal(torch.nan_to_num(o5[..., :2], 1.0), torch.nan_to_num(o4, 1.0))
        o3 = P.forward(torch.cat([coo, z[..., :1]], -1))
        zpass3 = o3.shape[-1] == 3 and torch.equal(o3[..., 2], z[..., 0])
        # float32
        o32 = P.forward(coo.float())
        i32 = P.inverse(o32)
        dt_ok = o32.dtype == torch.float32 and i32.dtype == torch.float32
        # requires_grad + backward with z passthrough
        c = torch.cat([coo, z], -1).clone().requires_grad_(True)
        o = P.forward(c)
        loss = torch.where(torch.isfinite(o), o, torch.zeros_like(o)).sum()
        loss.backward()
        gz_ok = torch.equal(c.grad[..., 2:], torch.ones_like(z))
        gfin = bool(torch.isfinite(c.grad).all())
        # 1-D input [2] and empty [0,2]
        try:
            e1 = P.forward(coo[0, 0, 0]).shape == (2,)
            e0 = P.forward(coo.reshape(-1, 2)[:0]).shape == (0, 2)
            ie0 = P.inverse(o4.reshape(-1, 2)[:0]).shape == (0, 2)
            small = e1 and e0 and ie0
        except Exception as err:
            small = f"error {err}"
        rt = float(torch.nan_to_num(i4 - coo, 0.0).abs().max())
        print(f"  {defn:<48} {kw}: [B,H,W,2] == flat {same}; z/t passthrough [..,4] {zpass} [..,3] {zpass3}; "
              f"fp32 dtype kept {dt_ok}; grad z==1 {gz_ok}; grads finite {gfin}; 1-D/empty {small}; roundtrip {rt:.1e} rad")
        for nm, v in [("batched", same), ("zpass", zpass and zpass3), ("fp32 dtype", dt_ok), ("grad z", gz_ok),
                      ("grad finite", gfin), ("1-D/empty", small is True)]:
            if not v:
                finding("shapes", f"{defn} {kw}: {nm} check failed")


# ============================================================================= 5. gradients
def interior_points(P, n, rng):
    lam = rng.uniform(-math.pi + 0.02, math.pi - 0.02, n)
    phi = rng.uniform(-H + 0.02, H - 0.02, n)
    keep = np.ones(n, bool)
    if isinstance(P, M.HEALPix):
        p0 = math.asin(2 / 3)
        keep &= np.abs(np.abs(phi) - p0) > 2e-3          # eq/polar kink
        keep &= np.abs(((lam + math.pi) / H) - np.round((lam + math.pi) / H)) > 2e-3  # cell edges
    if isinstance(P, M.LambertAzimuthalEqualArea):
        cosd = (np.sin(phi) * np.sin(-P.phi0) + np.cos(phi) * np.cos(-P.phi0) * np.cos(lam - math.pi))
        keep &= np.arccos(np.clip(cosd, -1, 1)) > 0.05     # away from the antipode
        cosc = (np.sin(phi) * np.sin(P.phi0) + np.cos(phi) * np.cos(P.phi0) * np.cos(lam))
        keep &= np.arccos(np.clip(cosc, -1, 1)) > 1e-3     # away from the centre (polar modes: lam undefined)
    return lam[keep], phi[keep]


def jac_autograd(fn, inp):
    inp = inp.clone().requires_grad_(True)
    out = fn(inp)
    J = torch.zeros(inp.shape[0], 2, 2, dtype=inp.dtype)
    for k in range(2):
        (g,) = torch.autograd.grad(out[:, k].sum(), inp, retain_graph=True)
        J[:, k, :] = g[:, :2]
    return J


def jac_fd(fn, inp, h):
    J = torch.zeros(inp.shape[0], 2, 2, dtype=inp.dtype)
    with torch.no_grad():
        for j in range(2):
            e = torch.zeros_like(inp)
            e[:, j] = h
            J[:, :, j] = (fn(inp + e)[:, :2] - fn(inp - e)[:, :2]) / (2 * h)
    return J


GRAD_DEFS = [(d, {}) for d in CLAIMS] + [
    ("+proj=moll +R=6371000", {"method": "fixed"}), ("+proj=moll +R=6371000", {"method": "implicit"}),
    ("+proj=laea +a=6378137 +rf=3 +lat_0=-60", {}), ("+proj=healpix +a=6378137 +rf=3", {}),
    ("+proj=laea +R=6371000 +lat_0=89.99999999", {}), ("+proj=laea +ellps=GRS80 +lat_0=89.99999999", {}),
]


def run_gradients(rng):
    hdr("5a. Autograd Jacobian vs central finite differences (float64, interior points, per-point rel. Frobenius error)")
    n = max(N // 10, 500)
    for defn, kw in GRAD_DEFS:
        P = M.make(defn, **kw)
        lam, phi = interior_points(P, n, rng)
        inp = torch.tensor(np.stack([lam, phi], -1), dtype=torch.float64)
        Ja = jac_autograd(P.forward, inp)
        Jf = jac_fd(P.forward, inp, 1e-6)
        with torch.no_grad():
            xy = P.forward(inp)
        okp = torch.isfinite(xy).all(-1)
        rel_f = ((Ja - Jf).flatten(1).norm(dim=1) / Jf.flatten(1).norm(dim=1))[okp]
        # inverse at the images (step 1 m scaled by a/6.4e6)
        xy = xy[okp]
        h = max(P.a / 6.4e6, 1e-9)
        Jai = jac_autograd(P.inverse, xy)
        Jfi = jac_fd(P.inverse, xy, h)
        # lam jumps by 2pi across the antimeridian under adjlon: drop stencils that straddle it
        with torch.no_grad():
            lp = P.inverse(xy)
        oki = torch.isfinite(Jfi.flatten(1)).all(1) & (lp[:, 0].abs() < math.pi - 0.01) & (Jfi.flatten(1).abs().max(1).values < 1e3 / P.a)
        rel_i = ((Jai - Jfi).flatten(1).norm(dim=1) / Jfi.flatten(1).norm(dim=1))[oki]
        nan_a = int((~torch.isfinite(Ja[okp])).any(-1).any(-1).sum() + (~torch.isfinite(Jai)).any(-1).any(-1).sum())
        mf, mi = float(rel_f.max()), float(rel_i.max())
        bad = mf > 1e-5 or mi > 1e-5 or nan_a
        print(f"  {'BAD ' if bad else 'ok  '}{defn:<68} {str(kw):<22} fwd max {mf:.1e} p99 {float(rel_f.quantile(0.99)):.1e} "
              f"| inv max {mi:.1e} p99 {float(rel_i.quantile(0.99)):.1e} | n={int(okp.sum())}/{int(oki.sum())} nonfinite={nan_a}")
        if bad:
            print("      -> FD disagrees; checked against mpmath in 5f (FD is meaningless when the float64 "
                  "function itself is noisy, e.g. PROJ's cancelling formula for a near-polar OBLIQ lat_0)")
            SUSPECT.append((defn, kw))

    hdr("5b. NaN/inf gradient scan at special points (loss masks non-finite outputs -> grads must stay finite)")
    for defn, kw in GRAD_DEFS + [("+proj=wag4 +R=1", {}), ("+proj=moll +R=1 +lon_0=170", {})]:
        P = M.make(defn, **kw)
        sl, sp = special_points(P)
        lam = np.concatenate([sl, [float("nan"), 0.1, float("inf")]])
        phi = np.concatenate([sp, [0.1, float("nan"), 0.2]])
        inp = torch.tensor(np.stack([lam, phi], -1), dtype=torch.float64).requires_grad_(True)
        out = P.forward(inp)
        torch.where(torch.isfinite(out), out, torch.zeros_like(out)).sum().backward()
        badf = ~torch.isfinite(inp.grad).all(-1)
        # inverse specials: centre, origin, image of poles, laea rim, outside, cap tips, polygon vertices
        a = P.a
        sx = [0.0, 1e-12, 2.0, -2.0, math.sqrt(2.0), 3 * math.pi / 4, -3 * math.pi / 4, math.pi, -math.pi,
              math.pi / 4, 2.0 * 0.90031631615710606 * math.sqrt(2), 5.0, float("nan")]
        sy = [0.0, 1e-12, 0.0, 2.0, math.sqrt(2.0), H, -H, math.pi / 4, -math.pi / 4, H, 0.0, 5.0, 0.0]
        X, Y = ref_fwd(defn, sl, sp)
        xs = np.concatenate([np.array(sx) * a + P.x0, X[np.isfinite(X)]])
        ys = np.concatenate([np.array(sy) * a + P.y0, Y[np.isfinite(Y)]])
        xs = np.concatenate([xs, xs[:-1], xs])
        ys = np.concatenate([ys, -ys[:-1], ys + 1e-9 * a])
        inp2 = torch.tensor(np.stack([xs, ys], -1), dtype=torch.float64).requires_grad_(True)
        out2 = P.inverse(inp2)
        torch.where(torch.isfinite(out2), out2, torch.zeros_like(out2)).sum().backward()
        badi = ~torch.isfinite(inp2.grad).all(-1)
        fin_f = torch.from_numpy(np.isfinite(lam) & np.isfinite(phi))
        fin_i = torch.from_numpy(np.isfinite(xs) & np.isfinite(ys))
        nf_in, ni_in = int((badf & ~fin_f).sum()), int((badi & ~fin_i).sum())
        badf, badi = badf & fin_f, badi & fin_i      # only FINITE inputs are a defect
        nf, ni = int(badf.sum()), int(badi.sum())
        print(f"  {'BAD ' if nf or ni else 'ok  '}{defn:<68} {str(kw):<22} non-finite grads at finite inputs: fwd {nf}/{len(lam)} inv {ni}/{len(xs)}"
              f"  (at NaN/inf inputs: fwd {nf_in} inv {ni_in})")
        NANIN[0] += nf_in + ni_in
        if nf or ni:
            ef = [(float(lam[i]), float(phi[i]), bool(torch.isfinite(out[i]).all())) for i in np.nonzero(badf.numpy())[0][:4]]
            ei = [(float(xs[i]), float(ys[i]), bool(torch.isfinite(out2[i]).all())) for i in np.nonzero(badi.numpy())[0][:4]]
            finding("nan-grad", f"{defn} {kw}: NaN gradient at finite inputs (value is NaN there too, as in PROJ) although the "
                    f"loss masks the output: fwd (lam,phi,out_finite)={ef} inv (x,y,out_finite)={ei}")
    if NANIN[0]:
        finding("nan-grad", f"NaN/inf INPUT coordinates always get NaN input-gradients ({NANIN[0]} cases over all "
                "definitions) even when the loss masks their NaN outputs: the NaN rule masks values only "
                "(sin/cos of NaN in the backward). Harmless for data, poisonous if coordinates come from a network.")

    hdr("5c. gradcheck on configurations the implementer did not gradcheck")
    for defn, kw in [("+proj=laea +R=6371000 +lat_0=-90 +lon_0=20", {}), ("+proj=laea +ellps=GRS80 +lat_0=89.99999999", {}),
                     ("+proj=healpix +ellps=WGS84 +rot_xy=-15 +x_0=1000 +y_0=-2000", {}),
                     ("+proj=rhealpix +R=6400000 +north_square=1 +south_square=0 +lon_0=-40", {}),
                     ("+proj=laea +a=6378137 +rf=3 +lat_0=-60", {}), ("+proj=wag5 +ellps=intl +lon_0=33", {}),
                     ("+proj=moll +R=6371000", {"method": "fixed", "n_iter": 12})]:
        P = M.make(defn, **kw)
        lam, phi = interior_points(P, 12, rng)
        inp = torch.tensor(np.stack([lam, phi], -1), dtype=torch.float64)
        a = P.a
        f = lambda t: P.forward(t) / a
        xy = P.forward(inp).detach()
        okm = torch.isfinite(xy).all(-1)
        g = lambda t: P.inverse(t * a)
        try:
            r1 = torch.autograd.gradcheck(f, (inp.clone().requires_grad_(True),), eps=1e-6, atol=1e-6, rtol=1e-4)
        except Exception as err:
            r1 = f"FAIL {str(err)[:100]}"
        try:
            r2 = torch.autograd.gradcheck(g, ((xy[okm] / a).clone().requires_grad_(True),), eps=1e-7, atol=1e-6, rtol=1e-4)
        except Exception as err:
            r2 = f"FAIL {str(err)[:100]}"
        print(f"  {defn:<68} {str(kw):<34} fwd {r1}  inv {r2}")
        if r1 is not True or r2 is not True:
            finding("gradcheck", f"{defn} {kw}: fwd {r1} inv {r2}")

    hdr("5d. gradients that are zero/huge where the true derivative is finite (documented? quantify)")
    # laea inverse near the pole image: asin edge -> d(phi)/d(x,y) = 0 within ~0.1 m
    for defn in ["+proj=laea +R=6371000 +lat_0=90", "+proj=laea +ellps=GRS80 +lat_0=90", "+proj=laea +R=6371000"]:
        P = M.make(defn)
        X, Y = ref_fwd(defn, [0.0], [H])
        d = np.array([0.0, 0.01, 0.05, 0.1, 0.2, 1.0, 10.0])
        inp = torch.tensor(np.stack([X[0] + 0 * d, Y[0] - d], -1), dtype=torch.float64).requires_grad_(True)
        out = P.inverse(inp)
        (gphi,) = torch.autograd.grad(out[:, 1].sum(), inp)
        print(f"  {defn:<40} d(phi)/dy at {d} m from the N-pole image: {gphi[:, 1].numpy()} (true |.| ~ 1/R = {1 / P.a:.2e})")
    # laea polar forward within 4.5e-8 rad of the centre pole -> zero jacobian (q<1e-15 branch)
    P = M.make("+proj=laea +ellps=GRS80 +lat_0=90")
    dd = np.array([1e-9, 1e-8, 3e-8, 5e-8, 1e-7])
    inp = torch.tensor(np.stack([0.3 + 0 * dd, H - dd], -1), dtype=torch.float64).requires_grad_(True)
    out = P.forward(inp)
    (gy,) = torch.autograd.grad(out[:, 1].sum(), inp)
    print(f"  laea ell N_POLE fwd dy/dphi at delta={dd} rad from pole: {gy[:, 1].numpy()} (true ~ a*cos(lam) ~ {P.a * math.cos(0.3):.2e})")


# ============================================================================= 5e. mpmath truth
def run_grad_truth():
    """Autograd derivatives near singular points vs mpmath (50 digits) derivatives of the
    exact mathematical map. FD in float64 is useless there (the float64 function itself is
    noisy), so this is the only reliable reference."""
    import mpmath as mp
    mp.mp.dps = 50
    hdr("5e. Gradient accuracy near poles / pole images / antipodes vs mpmath ground truth (relative error)")

    def ad_grad(fn, inp, k, j):
        inp = torch.tensor(inp, dtype=torch.float64).requires_grad_(True)
        out = fn(inp)
        (g,) = torch.autograd.grad(out[..., k].sum(), inp)
        return g[..., j].numpy(), out.detach().numpy()

    rows = []
    # (a) laea sphere inverse, d(phi)/dy approaching the N-pole image along the y axis (x = 0)
    for defn in ["+proj=laea +R=6371000", "+proj=laea +R=6371000 +lat_0=45"]:
        P = M.make(defn)
        a = P.a
        sb, cb = mp.sin(mp.mpf(P.phi0)), mp.cos(mp.mpf(P.phi0))

        def phi_true(y):
            y = mp.mpf(y)
            rh = abs(y)
            z = 2 * mp.asin(rh / 2)
            return mp.asin(mp.cos(z) * sb + y * mp.sin(z) * cb / rh)
        yp = float(2 * mp.sin((mp.pi / 2 - mp.mpf(P.phi0)) / 2))  # N-pole image on the unit sphere (x = 0)
        for d in [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]:
            y = yp * a - d
            g, _ = ad_grad(P.inverse, [[0.0, y]], 1, 1)
            gt = float(mp.diff(phi_true, mp.mpf(y) / a)) / a
            rows.append((f"{defn} inv dphi/dy, {d:g} m from pole image", float(g[0]), gt))
    # (b) laea ellipsoid N_POLE forward, dy/dphi approaching the pole
    P = M.make("+proj=laea +ellps=GRS80 +lat_0=90")
    e = mp.sqrt(mp.mpf(P.es))
    one_es = 1 - e * e

    def qf(phi):
        s = mp.sin(phi)
        return one_es * (s / (1 - e * e * s * s) + mp.atanh(e * s) / e)
    qp = qf(mp.pi / 2)
    for d in [5e-8, 1e-7, 1e-6, 1e-5, 1e-4]:
        phi = H - d
        g, _ = ad_grad(P.forward, [[0.0, phi]], 1, 1)
        gt = float(mp.diff(lambda p: -mp.sqrt(qp - qf(p)), mp.mpf(phi))) * P.a
        rows.append((f"laea GRS80 N_POLE fwd dy/dphi, {d:g} rad from pole", float(g[0]), gt))
    # (c) healpix sphere forward near the pole: dx/dlam = sigma, dy/dphi
    P = M.make("+proj=healpix +R=1")
    for d in [1e-8, 1e-7, 1e-6, 1e-5]:
        phi = H - d
        g, _ = ad_grad(P.forward, [[0.3, phi]], 0, 0)
        gt = float(mp.sqrt(3 * (1 - mp.sin(mp.mpf(phi)))))
        rows.append((f"healpix R=1 fwd dx/dlam (=sigma), {d:g} rad from pole", float(g[0]), gt))
        g, _ = ad_grad(P.forward, [[0.3, phi]], 1, 1)
        gt = float(mp.diff(lambda p: mp.pi / 4 * (2 - mp.sqrt(3 * (1 - mp.sin(p)))), mp.mpf(phi)))
        rows.append((f"healpix R=1 fwd dy/dphi, {d:g} rad from pole", float(g[0]), gt))
    # (d) Mollweide forward near the pole: dy/dphi (masked/implicit)
    for kw in [{"method": "masked"}, {"method": "implicit"}]:
        P = M.make("+proj=moll +R=1", **kw)

        def th2_true(p):  # root of t + sin t = C_p sin p (C_p = pi for moll), bracketed
            k = mp.mpf(P.C_p) * mp.sin(p)
            lo, hi = mp.mpf(0), mp.mpf(4)          # t + sin t is non-decreasing: plain bisection
            for _ in range(200):
                mid = (lo + hi) / 2
                if mid + mp.sin(mid) - k < 0:
                    lo = mid
                else:
                    hi = mid
            return (lo + hi) / 2
        for d in [1e-6, 1e-4, 1e-2]:
            phi = H - d
            g, _ = ad_grad(P.forward, [[0.3, phi]], 1, 1)
            t2 = th2_true(mp.mpf(phi))   # y = C_y sin(t2/2); dt2/dphi = C_p cos(phi) / (1 + cos t2)
            gt = float(mp.mpf(P.C_y) * mp.cos(t2 / 2) / 2 * mp.mpf(P.C_p) * mp.cos(mp.mpf(phi)) / (1 + mp.cos(t2)))
            rows.append((f"moll {kw['method']} fwd dy/dphi, {d:g} rad from pole", float(g[0]), gt))
    # (e) laea sphere OBLIQ forward approaching the antipode (b -> 0)
    P = M.make("+proj=laea +R=1 +lat_0=45")
    sb, cb = mp.sin(mp.mpf(P.phi0)), mp.cos(mp.mpf(P.phi0))

    def y_true(p, lam):
        p = mp.mpf(p)
        b = 1 + sb * mp.sin(p) + cb * mp.cos(p) * mp.cos(lam)
        return mp.sqrt(2 / b) * (cb * mp.sin(p) - sb * mp.cos(p) * mp.cos(lam))
    for d in [1e-4, 1e-3, 1e-2]:
        phi = -P.phi0 + d
        g, _ = ad_grad(P.forward, [[math.pi, phi]], 1, 1)
        gt = float(mp.diff(lambda p: y_true(p, mp.pi), mp.mpf(phi)))
        rows.append((f"laea R=1 lat_0=45 fwd dy/dphi, {d:g} rad from antipode", float(g[0]), gt))
    worst = {}
    for lab, g, gt in rows:
        rel = abs(g - gt) / abs(gt) if gt != 0 else abs(g)
        print(f"  {lab:<62} autograd {g: .6e}  truth {gt: .6e}  rel err {rel:.1e}")
        key = lab.split(",")[0]
        worst[key] = max(worst.get(key, 0.0), rel)
    for k, v in worst.items():
        if v > 1e-3:
            finding("grad-accuracy", f"{k}: autograd derivative off by up to {v:.0%} vs mpmath near the singular point "
                    "(PROJ's cancelling formula differentiated literally; value error is sub-metre but the derivative "
                    "of the rounding noise is O(1))")


# ============================================================================= 5f/5g/3g
def run_extra():
    import mpmath as mp
    mp.mp.dps = 50
    hdr("5f. Near-polar OBLIQ sphere (lat_0=89.99999999): values and Jacobians vs mpmath (resolves the 5a FD flag)")
    for defn in ["+proj=laea +R=6371000 +lat_0=89.99999999", "+proj=laea +R=6371000 +lat_0=89.9999"]:
        P = M.make(defn)
        sb, cb = mp.sin(mp.mpf(P.phi0)), mp.cos(mp.mpf(P.phi0))

        def inv_true(x, y):  # Snyder's stable form of the spherical laea inverse
            x, y = mp.mpf(x), mp.mpf(y)
            rh = mp.sqrt(x * x + y * y)
            z = 2 * mp.asin(rh / 2)
            sz, cz = mp.sin(z), mp.cos(z)
            return mp.atan2(x * sz, rh * cb * cz - y * sb * sz), mp.asin(cz * sb + y * sz * cb / rh)
        rng = np.random.default_rng(5)
        lam = rng.uniform(-3, 3, 12)
        phi = rng.uniform(-1.4, 1.4, 12)
        X, Y = ref_fwd(defn, lam, phi)
        L, _ = ref_inv(defn, X, Y)
        ep, eo, je = [], [], []
        for i in range(len(X)):
            xs, ys = X[i] / P.a, Y[i] / P.a
            lt, pt = inv_true(xs, ys)
            J = np.array([[float(mp.diff(lambda u: inv_true(u, ys)[k], xs)) / P.a,
                           float(mp.diff(lambda v: inv_true(xs, v)[k], ys)) / P.a] for k in range(2)])
            inp = torch.tensor([[X[i], Y[i]]], dtype=torch.float64, requires_grad=True)
            o = P.inverse(inp)
            Ja = np.stack([torch.autograd.grad(o[0, k], inp, retain_graph=True)[0][0].numpy() for k in range(2)])
            c = math.cos(phi[i]) * P.a
            ep.append(abs(L[i] - float(lt)) * c)
            eo.append(abs(o[0, 0].item() - float(lt)) * c)
            je.append(np.linalg.norm(Ja - J) / np.linalg.norm(J))
        print(f"  {defn:<45} PROJ lon error vs truth max {max(ep):.2e} m | port {max(eo):.2e} m | port autograd Jacobian rel err {max(je):.1e}")
        if max(ep) > 1e-3:
            finding("proj-conditioning", f"{defn}: PROJ's own spherical OBLIQ inverse (laea.cpp:213-214, cosz - sin(phi)*sinb1) "
                    f"loses {max(ep):.1f} m in longitude for a near-polar lat_0; the faithful port inherits it "
                    f"(port {max(eo):.1f} m; port-vs-PROJ differences of ~8 m are noise of this cancellation). Autograd is fine "
                    f"(rel err {max(je):.0e}); the 5a FD flag is an FD artefact.")

    hdr("5g. torch.func transforms / forward-mode AD / second derivatives")
    from torch.func import jacrev, jacfwd, jvp
    x = torch.tensor([0.3, 0.6], dtype=torch.float64)
    for meth in ["masked", "fixed", "implicit"]:
        P = M.make("+proj=moll +R=1", method=meth)
        res = []
        for nm, f in [("jacrev", lambda: jacrev(P.forward)(x)), ("jacfwd", lambda: jacfwd(P.forward)(x)),
                      ("jvp", lambda: jvp(P.forward, (x,), (torch.ones_like(x),)))]:
            try:
                f()
                res.append(f"{nm} ok")
            except Exception as err:
                res.append(f"{nm} FAILS ({str(err).splitlines()[0][:60]}...)")
        xr = x.clone().requires_grad_(True)
        (g,) = torch.autograd.grad(P.forward(xr)[1], xr, create_graph=True)
        try:
            (h,) = torch.autograd.grad(g[1], xr)
            d2 = float(h[1])
        except Exception as err:
            d2 = f"error {err}"
        f1 = lambda p: P.forward(torch.tensor([0.3, p], dtype=torch.float64))[1].item()
        fd = (f1(0.6 + 1e-4) - 2 * f1(0.6) + f1(0.6 - 1e-4)) / 1e-8
        print(f"  moll {meth:<9} {'; '.join(res)} | d2y/dphi2 double-backward {d2} vs FD {fd:.5f}")
        if meth == "implicit" and isinstance(d2, float) and abs(d2 - fd) > 1e-3:
            finding("autograd", f"moll method='implicit': plain double-backward SILENTLY returns d2y/dphi2 = {d2} "
                    f"(true {fd:.4f}) instead of raising (once_differentiable does not trigger because the backward is "
                    "computed from no-grad saved tensors); torch.func jacrev/jacfwd/jvp/hessian refuse the Function "
                    "(no setup_context/vmap rule). Losses that contain the projection Jacobian get wrong gradients.")
        if meth != "implicit" and any("FAILS" in r for r in res):
            finding("autograd", f"moll {meth}: torch.func transform failure {res}")

    hdr("5h. float32 inputs with requires_grad: non-finite gradients at finite inputs (random + special points)")
    rng = np.random.default_rng(11)
    for defn, kw in GRAD_DEFS:
        P = M.make(defn, **kw)
        sl, sp = special_points(P, wrap=True)
        lam = np.concatenate([rng.uniform(-math.pi, math.pi, 2000), sl])
        phi = np.concatenate([np.arcsin(rng.uniform(-1, 1, 2000)), sp])
        inp = torch.tensor(np.stack([lam, phi], -1), dtype=torch.float32).requires_grad_(True)
        out = P.forward(inp)
        torch.where(torch.isfinite(out), out, torch.zeros_like(out)).sum().backward()
        nf = int((~torch.isfinite(inp.grad)).any(-1).sum())
        xy = out.detach()
        xy = xy[torch.isfinite(xy).all(-1)].clone().requires_grad_(True)
        o2 = P.inverse(xy)
        torch.where(torch.isfinite(o2), o2, torch.zeros_like(o2)).sum().backward()
        ni = int((~torch.isfinite(xy.grad)).any(-1).sum())
        nnan_out = int((~torch.isfinite(o2)).any(-1).sum())
        print(f"  {'BAD ' if nf or ni else 'ok  '}{defn:<68} {str(kw):<22} fp32 non-finite grads fwd {nf} inv {ni} "
              f"(inverse of own fp32 forward gives {nnan_out} NaN outputs)")
        if nf or ni:
            finding("nan-grad-fp32", f"{defn} {kw}: float32 non-finite gradients at finite inputs fwd {nf} inv {ni}")

    hdr("3g. PROJ parameters that are accepted but silently ignored (core.parse + module)")
    lam = np.radians([10.0, -50.0, 120.0])
    phi = np.radians([50.0, -20.0, 70.0])
    ignored = []
    for defn in ["+proj=laea +lat_0=52 +lon_0=10 +ellps=intl +towgs84=-87,-98,-121",
                 "+proj=laea +lat_0=52 +lon_0=10 +ellps=GRS80 +pm=paris",
                 "+proj=laea +lat_0=52 +lon_0=10 +ellps=GRS80 +units=km",
                 "+proj=moll +ellps=GRS80 +to_meter=0.3048",
                 "+proj=laea +lat_0=52 +lon_0=10 +ellps=GRS80 +R_A",
                 "+proj=laea +lat_0=52 +lon_0=10 +ellps=GRS80 +axis=neu",
                 "+proj=laea +lat_0=52 +lon_0=10 +ellps=GRS80 +geoc",
                 "+proj=healpix +ellps=GRS80 +lon_wrap=180"]:
        try:
            X, Y = ref_fwd(defn, lam, phi)
            x, y = port_fwd(M.make(defn), lam, phi)
            e = float(np.nanmax(np.hypot(x - X, y - Y)))
        except Exception as err:
            e = float("nan")
            print("   ", defn, err)
        print(f"  {defn:<70} max |port - PROJ| = {e:.3g} m (no error raised)")
        if not e < 1e-3:
            ignored.append(f"{defn.split()[-1]} ({e:.3g} m)")
    if ignored:
        finding("params", "accepted but silently ignored: " + ", ".join(ignored))

    hdr("3h. adjlon threshold: PROJ adjlon keeps |lam| < pi+1e-12 (strict), core keeps <= (adjlon.cpp:10)")
    thr = math.pi + 1e-12
    for defn in ["+proj=moll +R=6371000", "+proj=healpix +R=6371000"]:
        lam = np.array([thr, -thr, np.nextafter(thr, 0), np.nextafter(thr, 9)])
        s = fwd_stats(M.make(defn), defn, lam, np.zeros(4))
        print(f"  {defn:<30} lam = +-(pi+1e-12) etc.: max err {s['max']:.3g} m")
        if s["max"] > 1:
            finding("core/adjlon", f"{defn}: lam - lon_0 == +-(pi + 1e-12) exactly maps to the opposite edge "
                    f"({s['max']:.3g} m): core.adjlon uses '<=' where adjlon.cpp:10 uses '<'")

    hdr("3i. thresholds inside the projections: laea polar q >= 1e-15 branch, moll |V| < 1e-7 freeze, fp32 moll fallback")
    for defn in ["+proj=laea +ellps=GRS80 +lat_0=90", "+proj=laea +ellps=GRS80 +lat_0=-90", "+proj=laea +a=6378137 +rf=30 +lat_0=90"]:
        P = M.make(defn)
        d = np.linspace(1e-8, 1e-7, 20000)
        lam = np.full_like(d, 0.7)
        phi = H - d if "lat_0=90" in defn else -H + d
        X, Y = ref_fwd(defn, lam, phi)
        x, y = port_fwd(P, lam, phi)
        zr, zp = (X == 0) & (Y == 0), (x == 0) & (y == 0)
        print(f"  {defn:<45} (0,0)-branch decisions differ at {int((zr != zp).sum())}/{len(d)} pts; zero zone ends at "
              f"d = {d[zr].max():.3e} rad (report says ~4.5e-8); max err {np.nanmax(np.hypot(x - X, y - Y)):.1e} m")
        if (zr != zp).any():
            finding("mask", f"{defn}: q<1e-15 branch decided differently from PROJ at {int((zr != zp).sum())} points")
    rng = np.random.default_rng(3)
    defn = "+proj=moll +R=6371000 +lon_0=10 +x_0=500"
    P = M.make(defn)
    lam = rng.uniform(-math.pi, math.pi, 20000)
    phi = rng.uniform(-H, H, 20000)
    X, Y = ref_fwd(defn, lam, phi)
    x, y = port_fwd(P, lam, phi)
    x32, y32 = port_fwd(P, lam, phi, torch.float32)
    e32 = np.hypot(x32 - X, y32 - Y)
    print(f"  moll fp64 masked: bit-identical to PROJ on {np.mean((x == X) & (y == Y)):.2%} of points (max diff "
          f"{np.hypot(x - X, y - Y).max():.1e} m); fp32: {np.mean(e32 > 1e3):.1%} of points off by > 1 km (+-pi/2 fallback)")

    hdr("6b. one inductor compile (laea ellipsoid OBLIQ fwd+bwd), to spot-check the implementer's inductor claim")
    if DO_COMPILE:
        P = M.make("+proj=laea +lat_0=52 +lon_0=10 +ellps=GRS80")
        t0 = time.time()
        cf = torch.compile(P.forward, backend="inductor", fullgraph=True)
        x1 = torch.tensor([[0.3, 0.8], [-2.0, -1.2], [3.0, 1.5707963267948966]], dtype=torch.float64, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)
        o1, o2 = P.forward(x1), cf(x2)
        torch.where(torch.isfinite(o1), o1, 0 * o1).sum().backward()
        torch.where(torch.isfinite(o2), o2, 0 * o2).sum().backward()
        dv = float(torch.nan_to_num(o1 - o2, 0.0).abs().max())
        dg = float(torch.nan_to_num(x1.grad - x2.grad, 0.0).abs().max())
        print(f"  inductor fullgraph: |dv| {dv:.1e} m, |dg| {dg:.1e}, {time.time() - t0:.0f}s")


# ============================================================================= 6. GPU/compile
def run_static():
    hdr("6. Host syncs / Python control flow on tensor values / device handling")
    src = inspect.getsource(M)
    tree = ast.parse(src)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in ("item", "numpy", "tolist", "cpu", "cuda", "nonzero",
                                                            "any", "all", "__bool__"):
            hits.append((node.lineno, node.attr))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("bool", "int", "float"):
            seg = ast.get_source_segment(src, node)
            hits.append((node.lineno, seg))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "tensor":
            hits.append((node.lineno, "torch.tensor(...)"))
    for ln, what in sorted(hits):
        print(f"   line {ln}: {what}  -> {src.splitlines()[ln - 1].strip()[:90]}")
    # meta device: no data-dependent host sync can run on 'meta' tensors
    ok_meta = []
    for defn, kw in GRAD_DEFS[:20] + [("+proj=moll +R=1", {"method": "implicit"}), ("+proj=moll +R=1", {"method": "fixed"})]:
        P = M.make(defn, **kw)
        try:
            inp = torch.empty(4, 3, 2, dtype=torch.float64, device="meta")
            o1 = P.forward(inp)
            o2 = P.inverse(inp)
            ok_meta.append(o1.device.type == "meta" and o2.device.type == "meta")
        except Exception as err:
            ok_meta.append(False)
            finding("gpu", f"{defn} {kw}: fails on the meta device: {err}")
    print(f"   meta-device forward+inverse (no host sync / no CPU-pinned constants): {sum(ok_meta)}/{len(ok_meta)} ok")
    if not DO_COMPILE:
        return
    import torch._dynamo as dynamo
    print("   torch._dynamo.explain (graph breaks) + aot_eager fullgraph fwd/bwd on configurations not compiled by the implementer:")
    for defn, kw in [("+proj=laea +R=6371000 +lat_0=45 +lon_0=10", {}), ("+proj=laea +R=6371000", {}),
                     ("+proj=laea +ellps=GRS80 +lat_0=-90", {}), ("+proj=laea +ellps=GRS80", {}),
                     ("+proj=healpix +R=1", {}), ("+proj=rhealpix +R=1 +north_square=2", {}), ("+proj=wag5 +R=1", {})]:
        P = M.make(defn, **kw)
        for nm, fn, sample in [("fwd", P.forward, torch.tensor([[0.3, 0.4], [-2.0, -1.2], [3.0, 1.5]], dtype=torch.float64)),
                               ("inv", P.inverse, torch.tensor([[0.3, 0.4], [-2.0, -1.2], [3.0, 1.0]], dtype=torch.float64))]:
            dynamo.reset()
            ex = dynamo.explain(fn)(sample)
            dynamo.reset()
            cf = torch.compile(fn, backend="aot_eager", fullgraph=True)
            x1 = sample.clone().requires_grad_(True)
            x2 = sample.clone().requires_grad_(True)
            o1 = fn(x1)
            o2 = cf(x2)
            torch.where(torch.isfinite(o1), o1, 0 * o1).sum().backward()
            torch.where(torch.isfinite(o2), o2, 0 * o2).sum().backward()
            dv = float(torch.nan_to_num(o1 - o2, 0.0).abs().max())
            dg = float(torch.nan_to_num(x1.grad - x2.grad, 0.0).abs().max())
            print(f"   {nm} {defn:<45} graph breaks {ex.graph_break_count}, graphs {ex.graph_count}; aot_eager |dv| {dv:.1e} |dg| {dg:.1e}")
            if ex.graph_break_count:
                finding("compile", f"{defn} {nm}: {ex.graph_break_count} graph breaks")


# ============================================================================= main
def main():
    t0 = time.time()
    rng = np.random.default_rng(987654321)
    g = run_gie()
    run_headline(rng)
    run_attacks(rng)
    run_shapes()
    run_gradients(rng)
    run_grad_truth()
    run_extra()
    run_static()
    hdr(f"SUMMARY ({time.time() - t0:.0f}s): gie {g['passed']}/{g['cases']} in {g['blocks']} blocks; "
        f"{len(FINDINGS)} findings")
    seen = set()
    for sev, msg in FINDINGS:
        key = (sev, msg.split(":")[0] if sev in ("core/lon", "nan-rule", "nan-grad") else msg)
        if sev == "nan-rule":
            key = (sev,)
        if key in seen:
            continue
        seen.add(key)
        print(f"  [{sev}] {msg}")


if __name__ == "__main__":
    main()
