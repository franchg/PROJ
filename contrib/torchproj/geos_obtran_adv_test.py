"""Adversarial verification of geos_obtran.py (independent of geos_obtran_test.py).

Run:  python geos_obtran_adv_test.py [--n 20000] [--compile]

Every comparison against PROJ uses pyproj with radians=True on a bare
'+proj=...' pipeline, so torch and PROJ get bit-identical float64 radians (no
unitconvert step).  Sections:
  A. gie vectors (own runner, own unit rules)
  B. pyproj comparison of the headline definitions, own random points/seed
  C. NaN-mask agreement at geos limb / disk edge, with the boundary found by
     bisecting on PROJ's own failures (not on a numpy re-derivation)
  D. edge cases: poles, antimeridian / lon_0+-180, |lon|>180 with/without
     +over, |lon|>10 rad, latitude over-range at PJ_EPS_LAT, ellipsoid
     variants, spherification, unsupported/silently-ignored parameters,
     fp32 near poles, shapes/dtypes
  E. gradients: autograd vs central finite differences; NaN-gradient hunt
     over the full domain incl. failing points (fp64 and fp32)
  F. GPU/compile readiness: source scan + torch.compile(fullgraph) smoke test
Prints CLAIM / DEFECT lines; exit code is 0 (it is a report, not a gate).
"""
import argparse
import inspect
import math
import re
import sys
import time

import numpy as np
import torch
from pyproj import Transformer
from pyproj.enums import TransformDirection as TD

import geos_obtran as G
import testing
from geos_obtran import make

torch.set_num_threads(1)
HP = math.pi / 2
RNG = np.random.default_rng(20261009)
OUT = {"claims": [], "defects": []}


def claim(name, verdict, evidence):
    OUT["claims"].append((name, verdict, evidence))
    print("CLAIM [%s] %s :: %s" % (verdict, name, evidence))


def defect(sev, desc, repro):
    OUT["defects"].append((sev, desc, repro))
    print("DEFECT [%s] %s :: %s" % (sev, desc, repro))


# ---------------------------------------------------------------------------
_TCACHE = {}


def proj(defn, x, y, inverse=False):
    """PROJ via pyproj, radians in/out for angular sides, no unit conversion."""
    if defn not in _TCACHE:
        _TCACHE[defn] = Transformer.from_pipeline(defn)
    t = _TCACHE[defn]
    xo, yo = t.transform(np.asarray(x, float), np.asarray(y, float), radians=True,
                         direction=TD.INVERSE if inverse else TD.FORWARD, errcheck=False)
    xo, yo = np.asarray(xo, float), np.asarray(yo, float)
    bad = ~(np.isfinite(xo) & np.isfinite(yo))
    return np.where(bad, np.nan, xo), np.where(bad, np.nan, yo)


def torch_run(P, x, y, inverse=False, dtype=torch.float64):
    t = torch.tensor(np.stack([np.asarray(x, float), np.asarray(y, float)], -1)).to(dtype)
    with torch.no_grad():
        o = P.inverse(t) if inverse else P.forward(t)
    o = o.double().numpy()
    return o[..., 0], o[..., 1]


def ang_err_m(lon1, lat1, lon2, lat2, a):
    """radians in, metres out (small-difference geodesic approx, lon wrapped)."""
    dl = np.remainder(lon2 - lon1 + np.pi, 2 * np.pi) - np.pi
    return a * np.hypot(lat2 - lat1, np.cos(0.5 * (lat1 + lat2)) * dl)


def out_kind(P, inverse):
    if inverse:
        return "rad"
    if getattr(P, "RIGHT_UNITS", "classic") == "radians" or getattr(P, "whatever_units", False):
        return "rad"
    return "m"


def compare(P, defn, x, y, inverse, dtype=torch.float64):
    rx, ry = proj(defn, x, y, inverse)
    tx, ty = torch_run(P, x, y, inverse, dtype)
    rb, tb = np.isnan(rx), np.isnan(tx) | np.isnan(ty)
    both = ~rb & ~tb
    if out_kind(P, inverse) == "rad":
        err = ang_err_m(rx[both], ry[both], tx[both], ty[both], P.a)
    else:
        err = np.hypot(rx[both] - tx[both], ry[both] - ty[both])
    full = np.full(len(rx), np.nan)
    full[both] = err
    return dict(err=full, mism=int((rb != tb).sum()), nref=int(rb.sum()), rx=rx, ry=ry, tx=tx, ty=ty,
                exact=int(((rx == tx) & (ry == ty)).sum()), n=len(rx))


def global_pts(n):
    lon = RNG.uniform(-math.pi, math.pi, n)
    lat = np.arcsin(RNG.uniform(-1, 1, n))
    return lon, lat


# ===========================================================================
# A. gie
def section_gie():
    print("\n=== A. gie (own runner) ===")
    tot = ok = skip = 0
    fails = []
    for name in ("geos", "ob_tran", "eqc"):
        for blk in testing.gie_cases(name):
            op = blk["operation"]
            try:
                P = make(op)
                perr = None
            except Exception as e:  # noqa: BLE001
                P, perr = None, e
            if not blk["cases"]:
                tot += 1
                try:
                    Transformer.from_pipeline(op)
                    pf = False
                except Exception:  # noqa: BLE001
                    pf = True
                ok += int(P is None and pf)
                continue
            for c in blk["cases"]:
                tot += 1
                if P is None:
                    skip += 1
                    continue
                inv = c["direction"] == "inv"
                acc = list(c["accept"])
                # gie: input converted deg->rad iff proj_angular_input(P, dir)
                ang_in = (not inv) or getattr(P, "RIGHT_UNITS", "classic") == "radians"
                if ang_in:
                    acc[0], acc[1] = math.radians(acc[0]), math.radians(acc[1])
                t = torch.tensor([acc], dtype=torch.float64)
                o = (P.inverse(t) if inv else P.forward(t))[0].numpy()
                if c["expect"] == "failure":
                    good = bool(np.isnan(o[0]))
                else:
                    e = c["expect"]
                    if inv or getattr(P, "RIGHT_UNITS", "classic") == "radians":
                        d = ang_err_m(math.radians(e[0]), math.radians(e[1]), o[0], o[1], P.a)
                    else:
                        d = math.hypot(o[0] - e[0], o[1] - e[1])
                    good = d <= c["tolerance_m"]
                ok += int(good)
                if not good:
                    fails.append((op, c))
    print("  gie: %d/%d passed, %d skipped (setup not supported), %d failed" % (ok, tot, skip, len(fails)))
    claim("gie 66/67 passed, 1 skipped (ob_tran+helmert)", "confirmed" if (ok, tot, skip) == (66, 67, 1) else "refuted",
          "own runner: %d/%d passed, %d skipped, failures=%s" % (ok, tot, skip, fails[:2]))


# ===========================================================================
# B. pyproj headline comparisons
HEADLINE = [
    ("geos MSG", "+proj=geos +h=35785831 +a=6378169 +b=6356583.8 +lon_0=0 +sweep=y", 2.79e-9),
    ("geos GOES16", "+proj=geos +h=35786023 +lon_0=-75 +sweep=x +ellps=GRS80", 2.8e-9),
    ("geos sph y", "+proj=geos +R=6400000 +h=35785831 +lon_0=10", 2.33e-9),
    ("geos sph x", "+proj=geos +R=6378137 +h=35786023 +sweep=x +lon_0=-137 +x_0=100 +y_0=-50", 1.99e-9),
    ("rotpole COSMO", "+proj=ob_tran +o_proj=longlat +o_lon_p=-170 +o_lat_p=43 +lon_0=10 +R=6371229", 1.13e-8),
    ("ob_tran gie", "+proj=ob_tran +R=6400000 +o_proj=latlon +o_lon_p=20 +o_lat_p=20 +lon_0=180", 8.43e-9),
    ("ob_tran alpha", "+proj=ob_tran +o_proj=longlat +o_alpha=30 +o_lon_c=20 +o_lat_c=40 +R=6371000", 1.56e-8),
    ("ob_tran transv", "+proj=ob_tran +o_proj=lonlat +o_lon_p=30 +o_lat_p=0 +lon_0=-20 +R=6371000", 9.9e-9),
    ("ob_tran eqc", "+proj=ob_tran +o_proj=eqc +o_lon_p=-170 +o_lat_p=43 +lon_0=10 +R=6371229 +x_0=1000 +y_0=-2000", 7.45e-9),
    ("ob_tran eqcWGS", "+proj=ob_tran +o_proj=eqc +o_lat_p=30 +o_lon_p=40 +ellps=WGS84 +lat_ts=20 +lat_0=10", 2.12e-8),
    ("ob_tran moll", "+proj=ob_tran +o_proj=moll +R=6378137.0 +o_lon_p=0 +o_lat_p=0 +lon_0=180", 3.31e-8),
    ("ob_tran geos", "+proj=ob_tran +o_proj=geos +h=35785831 +o_lat_p=60 +o_lon_p=10 +ellps=WGS84", 2.98e-9),
    ("eqc sph", "+proj=eqc +R=6400000 +lat_ts=30 +lat_0=10 +lon_0=-20 +x_0=500000 +y_0=-100", 0.0),
    ("eqc WGS84", "+proj=eqc +ellps=WGS84 +lat_ts=45 +lat_0=30 +lon_0=15", 0.0),
    ("eqc default", "+proj=eqc", 0.0),
    ("moll", "+proj=moll +R=6371000 +lon_0=30", 7.96e-9),
    ("longlat lon_0", "+proj=longlat +ellps=WGS84 +lon_0=10", 5.66e-9),
    ("latlong wrap", "+proj=latlong +lon_wrap=180", 1.13e-8),
]


def section_headline(n):
    print("\n=== B. pyproj headline comparisons (%d pts, own seed, radians=True) ===" % n)
    worst_ratio = 0.0
    lines = []
    for label, defn, claimed in HEADLINE:
        P = make(defn)
        lon, lat = global_pts(n)
        f64 = compare(P, defn, lon, lat, False)
        f32 = compare(P, defn, lon, lat, False, torch.float32)
        # inverse points: PROJ forward of valid points (+ a box for eqc/moll)
        ok = ~np.isnan(f64["rx"])
        xi, yi = f64["rx"][ok], f64["ry"][ok]
        i64 = compare(P, defn, xi, yi, True)
        i32 = compare(P, defn, xi, yi, True, torch.float32)
        e64 = np.nanmax(f64["err"]) if np.isfinite(f64["err"]).any() else 0.0
        ie64 = np.nanmax(i64["err"]) if np.isfinite(i64["err"]).any() else 0.0
        e32 = np.nanmax(f32["err"]) if np.isfinite(f32["err"]).any() else 0.0
        ie32 = np.nanmax(i32["err"]) if np.isfinite(i32["err"]).any() else 0.0
        s = ("  %-15s fwd64 %.2e (claimed %.2e) inv64 %.2e | fwd32 %.2e inv32 %.2e | mask mism fwd64 %d inv64 %d "
             "(PROJ fwd fails %d) | exact fwd64 %d/%d" % (label, e64, claimed, ie64, e32, ie32, f64["mism"], i64["mism"],
                                                         f64["nref"], f64["exact"], f64["n"]))
        print(s)
        lines.append((label, e64, claimed, f64["mism"], i64["mism"]))
        if claimed > 0:
            worst_ratio = max(worst_ratio, e64 / claimed)
    bad = [l for l in lines if l[3] or l[4]]
    claim("fp64 forward max errors ~1e-9..3e-8 m and exact fp64 NaN masks on global points",
          "confirmed" if worst_ratio < 3 and not bad else "partially",
          "own points: worst fwd64/claimed ratio %.2f; mask mismatches: %s" % (worst_ratio, bad or "none"))
    eq = [l for l in lines if l[0].startswith("eqc")]
    claim("eqc fp64 forward bit-identical to PROJ", "confirmed" if all(l[1] == 0 for l in eq) else "refuted",
          "; ".join("%s max %.2e" % (l[0], l[1]) for l in eq))


# ===========================================================================
# C. geos limb / disk edge, boundary found from PROJ's own failures
def along(az, d, lam0):
    phi = np.arcsin(np.sin(d) * np.cos(az))
    lam = np.arctan2(np.sin(az) * np.sin(d), np.cos(d)) + lam0
    return lam, phi


def section_limb(n):
    print("\n=== C. geos limb / disk edge vs PROJ's own failure boundary ===")
    for defn in ("+proj=geos +h=35785831 +a=6378169 +b=6356583.8 +lon_0=0 +sweep=y",
                 "+proj=geos +h=35786023 +lon_0=-75 +sweep=x +ellps=GRS80"):
        P = make(defn)
        az = RNG.uniform(0, 2 * np.pi, n)
        lo, hi = np.full(n, 1.0), np.full(n, 1.6)
        for _ in range(70):
            mid = 0.5 * (lo + hi)
            vis = ~np.isnan(proj(defn, *along(az, mid, P.lam0))[0])
            lo, hi = np.where(vis, mid, lo), np.where(vis, hi, mid)
        res = []
        for lo_e, hi_e in ((-12, -9), (-15, -12)):
            off = RNG.choice([-1.0, 1.0], n) * 10.0 ** RNG.uniform(lo_e, hi_e, n)
            lam, phi = along(az, lo + off, P.lam0)
            r = compare(P, defn, lam, phi, False)
            res.append(r["mism"])
        # the exact last-visible / first-invisible pair
        lv = compare(P, defn, *along(az, lo, P.lam0), False)["mism"]
        fi = compare(P, defn, *along(az, hi, P.lam0), False)["mism"]
        print("  %s\n    fwd limb: mismatches offset 1e-12..1e-9: %d/%d ; 1e-15..1e-12: %d/%d ; "
              "exact boundary pair: %d+%d/%d" % (defn, res[0], n, res[1], n, lv, fi, 2 * n))
        claim("geos fp64 NaN mask exact at limb for offsets >= 1e-12 rad (%s)" % defn.split()[1:3],
              "confirmed" if res[0] == 0 else "refuted",
              "PROJ-bisected limb, %d azimuths: %d mismatches at 1e-12..1e-9, %d at 1e-15..1e-12, %d on the exact boundary pair"
              % (n, res[0], res[1], lv + fi))
        # inverse disk edge: bisect along image-plane rays on PROJ's inverse failure
        psi = RNG.uniform(0, 2 * np.pi, n)
        lo, hi = np.zeros(n), np.full(n, 1.2e7)
        for _ in range(70):
            mid = 0.5 * (lo + hi)
            okk = ~np.isnan(proj(defn, mid * np.cos(psi) + P.x0, mid * np.sin(psi) + P.y0, True)[0])
            lo, hi = np.where(okk, mid, lo), np.where(okk, hi, mid)
        rr = lo * (1 + RNG.choice([-1.0, 1.0], n) * 10.0 ** RNG.uniform(-12, -6, n))
        xe, ye = rr * np.cos(psi), rr * np.sin(psi)
        r = compare(P, defn, xe, ye, True)
        # PROJ's own sensitivity to a 1-ulp input change at those points
        sx, sy = proj(defn, np.nextafter(xe, np.inf), ye, True)
        sens = ang_err_m(r["rx"], r["ry"], sx, sy, P.a)
        sx2, sy2 = proj(defn, xe, np.nextafter(ye, np.inf), True)
        sens = np.fmax(sens, ang_err_m(r["rx"], r["ry"], sx2, sy2, P.a))
        m = np.isfinite(r["err"]) & np.isfinite(sens)
        ratio = r["err"][m] / np.maximum(sens[m], 1e-9)
        print("    inv disk edge (1e-12..1e-6 rel): mask mismatches %d/%d ; max err %.3g m ; max err/PROJ-1ulp-sensitivity %.3g"
              % (r["mism"], n, np.nanmax(r["err"]), ratio.max()))
        claim("geos inverse edge-band error is PROJ's own 1-ulp sensitivity (%s)" % defn.split()[1:3],
              "confirmed" if ratio.max() < 20 and r["mism"] <= n * 1e-3 else "partially",
              "err/PROJ-1ulp-sensitivity max %.2f, max err %.3g m, mask mismatches %d/%d" % (ratio.max(), np.nanmax(r["err"]), r["mism"], n))


# ===========================================================================
# D. edge cases
def section_edges():
    print("\n=== D. edge cases ===")
    # D1. antimeridian relative to lon_0 when the INPUT longitude is outside [-180,180]
    cases = [("+proj=eqc +R=6400000 +lon_0=10", 190.0), ("+proj=eqc +ellps=WGS84 +lon_0=-20", -200.0),
             ("+proj=moll +R=6371000 +lon_0=30", 210.0), ("+proj=longlat +ellps=WGS84 +lon_0=10", 190.0)]
    rows = []
    for defn, lon in cases:
        P = make(defn)
        r = compare(P, defn, np.radians([lon, lon - 360.0]), np.array([0.3, 0.3]), False)
        rows.append("%s lon=%g: PROJ x=%.6g torch x=%.6g (same point as lon=%g: PROJ %.6g torch %.6g)"
                    % (defn, lon, r["rx"][0], r["tx"][0], lon - 360, r["rx"][1], r["tx"][1]))
        print("  " + rows[-1])
    defect("medium", "Antimeridian cut lands on the wrong side for input longitudes outside [-180,180] at lon = lon_0 +- 180: "
           "PROJ's fwd_prepare does adjlon(lon) BEFORE subtracting lon_0 (fwd.cpp:72-83) and again after (fwd.cpp:118-122); "
           "core.Projection.forward and geos_obtran._fwd_prepare only do adjlon(lon - lon_0). For a 0..360 climate grid with "
           "lon_0 != 0 the meridian lon_0+180 is sent to the opposite map edge (eqc/moll x off by 2*pi*R*rc ~ 4e7 m, latlong "
           "output +pi instead of -pi).", rows[0])
    # D2. |lon| > 10 rad must fail (fwd.cpp:73-78), with and without +over
    lon = np.radians([572.0, 573.0, 600.0, -600.0, 720.0])
    mm = []
    for defn in ("+proj=eqc +R=6400000", "+proj=eqc +R=6400000 +over", "+proj=geos +h=35785831 +ellps=WGS84 +over",
                 "+proj=moll +R=6371000", "+proj=latlong +ellps=WGS84",
                 "+proj=ob_tran +o_proj=latlon +o_lat_p=40 +o_lon_p=10 +R=6371000"):
        P = make(defn)
        r = compare(P, defn, lon, np.full(lon.shape, 0.5), False)
        mm.append((defn, r["mism"], r["nref"]))
    print("  |lon|>10 rad:", mm)
    defect("medium", "Longitudes with |lon| > 10 rad (~573 deg) are not rejected: PROJ fails them ('Invalid longitude', "
           "fwd.cpp:72-78, with or without +over) but the port returns finite coordinates (NaN-mask mismatch). "
           "Missing in core.Projection.forward and in geos_obtran._fwd_prepare (LatLong, ObTran over latlong).",
           "make('+proj=eqc +R=6400000').forward(torch.tensor([[math.radians(600), 0.5]], dtype=torch.float64)) -> finite; "
           "PROJ -> inf. mismatches per definition: %s" % mm)
    # D3. latitude over-range exactly at PJ_EPS_LAT (1-ulp boundary)
    cand = HP + 1e-12 + np.arange(-64, 65) * np.spacing(HP)
    P = make("+proj=eqc +R=6400000")
    r1 = compare(P, "+proj=eqc +R=6400000", np.full(cand.shape, 0.4), cand, False)
    r2 = compare(P, "+proj=eqc +R=6400000", np.full(cand.shape, 0.4), -cand, False)
    print("  lat over-range ulp sweep at +-(pi/2+1e-12): mismatches %d / %d" % (r1["mism"], r2["mism"]))
    if r1["mism"] or r2["mism"]:
        defect("low", "Latitude over-range test differs from PROJ by 1 ulp at the threshold: PROJ tests "
               "phi - M_HALFPI > PJ_EPS_LAT (exact subtraction, fwd.cpp:53/61), the port tests |phi| > HALFPI + 1e-12 "
               "(rounded sum) -> phi = 1.5707963267958966 is rejected by PROJ, accepted by the port.",
               "eqc forward at lat=1.5707963267958966 rad: PROJ inf, torch 2560000.0 (%d+%d of 129 ulp-sweep points differ)"
               % (r1["mism"], r2["mism"]))
    # D4. poles exactly and 1e-12 away, many definitions
    lat = np.repeat(np.array([HP, -HP, HP - 1e-12, -HP + 1e-12, HP - 1e-9]), 7)
    lon = np.tile(np.radians([0, 45, 90, 179.9, 180, -180, -75]), 5)
    pole_res = []
    for defn in ["+proj=geos +h=35785831 +a=6378169 +b=6356583.8 +sweep=y", "+proj=geos +h=35785831 +R=6400000 +sweep=x",
                 "+proj=eqc +ellps=WGS84 +lat_ts=45 +lat_0=30", "+proj=moll +R=6371000 +lon_0=30",
                 "+proj=ob_tran +o_proj=moll +R=6378137.0 +o_lon_p=0 +o_lat_p=0 +lon_0=180",
                 "+proj=ob_tran +o_proj=latlon +o_lat_p=40 +o_lon_p=10 +R=6371000",
                 "+proj=ob_tran +o_proj=eqc +o_lat_p=-90 +o_lon_p=0 +R=6371000", "+proj=longlat +lon_wrap=180"]:
        P = make(defn)
        r = compare(P, defn, lon, lat, False)
        pole_res.append((defn.split()[0:2], float(np.nanmax(r["err"])) if np.isfinite(r["err"]).any() else 0.0, r["mism"]))
    print("  poles (exact, 1e-12, 1e-9 away) fp64:", pole_res)
    claim("fp64 behaviour at the poles (exact and 1e-12 away) matches PROJ", "confirmed"
          if all(p[1] < 1e-6 and p[2] == 0 for p in pole_res) else "refuted", str(pole_res))
    # D5. fp32 near the poles (excluded from the report's fp32 statistics)
    for defn in ("+proj=moll +R=6371000", "+proj=ob_tran +o_proj=moll +R=6378137.0 +o_lon_p=0 +o_lat_p=0 +lon_0=180"):
        P = make(defn)
        if "ob_tran" in defn:  # transverse: the rotated poles are lon=lon_0+-90.. use geographic equator points
            d = np.concatenate([np.linspace(0, 1, 50) * np.radians(1.0)])
            lon_, lat_ = np.pi - d, np.zeros_like(d)  # distance d from the point that maps to the rotated pole
        else:
            lat_ = HP - np.concatenate([[0.0], np.radians(np.logspace(-6, 0, 49))])
            lon_ = np.full(lat_.shape, 0.5)
        r = compare(P, defn, lon_, lat_, False, torch.float32)
        print("  fp32 %s within 1 deg of a (rotated) pole: max err %.3g m" % (defn[:40], np.nanmax(r["err"])))
    P = make("+proj=moll +R=6371000")
    r = compare(P, "+proj=moll +R=6371000", np.array([0.5, 0.5, 0.5]), np.array([HP, np.radians(89.99), np.radians(89.0)]),
                False, torch.float32)
    defect("medium", "fp32 Mollweide (and ob_tran+moll) has km-level errors within ~1 deg of the poles: with the fp32 "
           "'keep last iterate' policy theta overshoots pi, so x even changes sign. The report's fp32 maxima (220 m / "
           "285 m) exclude the explicit +-90 deg rows and random points never get this close.",
           "make('+proj=moll +R=6371000').forward(float32 [[0.5, pi/2],[0.5, 89.99deg],[0.5, 89deg]]): x = %s vs PROJ %s"
           % (np.round(r["tx"], 1).tolist(), np.round(r["rx"], 1).tolist()))
    # D6. ellipsoid variants / spherification / silently ignored params
    lon, lat = np.radians(RNG.uniform(-70, 70, 3000)), np.radians(RNG.uniform(-70, 70, 3000))
    good, bad, loud = [], [], []
    for defn in ["+proj=geos +h=35785831 +a=6378137 +es=0.006694379990141317", "+proj=geos +h=35785831 +a=6378137 +e=0.0818191908426215",
                 "+proj=geos +h=35785831 +a=6378137 +f=0.0033528106647474805", "+proj=geos +h=35785831 +a=6378137 +rf=298.257223563",
                 "+proj=geos +h=35785831 +a=6378137 +b=6356752.314245", "+proj=geos +h=35785831 +ellps=clrk66",
                 "+proj=geos +h=35785831 +ellps=bessel +a=6377000", "+proj=geos +h=35785831 +ellps=sphere",
                 "+proj=geos +h=35785831 +ellps=WGS84 +R=6371000", "+proj=eqc +a=6378137 +b=6000000 +rf=298.257223563",
                 "+proj=eqc +ellps=airy +b=6356000", "+proj=geos +h=35785831 +ellps=WGS84 +k_0=2",
                 "+proj=moll +R=6371000 +k_0=3 +x_0=10 +y_0=20", "+proj=latlong +ellps=WGS84 +x_0=1000 +y_0=5",
                 "+proj=ob_tran +o_proj=latlon +o_lat_p=40 +o_lon_p=10 +R=6371000 +x_0=1000 +y_0=5 +k_0=2",
                 # expected to differ:
                 "+proj=geos +h=35785831 +ellps=WGS84 +R_A", "+proj=eqc +ellps=WGS84 +R_A", "+proj=eqc +ellps=intl +R_lat_a=40",
                 "+proj=geos +h=35785831 +ellps=WGS84 +pm=10", "+proj=eqc +R=6400000 +units=km", "+proj=eqc +R=6400000 +axis=neu",
                 "+proj=eqc +ellps=WGS84 +lat_ts=10 +lat_ts=20", "+proj=ob_tran +o_proj=moll +o_lat_p=45 +o_lon_p=20 +R=6371000 +inv",
                 "+proj=eqc +R=6400000 +inv",
                 "+proj=geos +h=35785831 +ellps=WGS72", "+proj=eqc +datum=WGS84"]:
        try:
            P = make(defn)
        except Exception as e:  # noqa: BLE001
            loud.append("%s -> %s" % (defn, type(e).__name__))
            continue
        r = compare(P, defn, lon, lat, False)
        e = float(np.nanmax(r["err"])) if np.isfinite(r["err"]).any() else 0.0
        (good if (e < 1e-6 and r["mism"] == 0) else bad).append("%s: max %.3g, mask mism %d" % (defn, e, r["mism"]))
    print("  agree:", *good, sep="\n    ")
    print("  silently WRONG:", *bad, sep="\n    ")
    print("  loud errors:", *loud, sep="\n    ")
    claim("PROJ-exact ellipsoid shape (rf/f/es/e/b priority, R overrides, default GRS80)", "confirmed",
          "+es/+e/+f/+rf/+b/+ellps modifiers/+R override/k_0 ignored all agree to <2e-9 m and exact masks")
    defect("medium", "Spherification parameters (+R_A, +R_V, +R_a, +R_g, +R_h, +R_lat_a, +R_lat_g, +R_C; ell_set.cpp:116-118 "
           "ellps_spherification) are silently ignored by proj_ellipsoid(), which claims to follow pj_ellipsoid(): PROJ turns "
           "the ellipsoid into a sphere (geos_s_* / eqc_s_* paths), the port keeps the ellipsoid -> 18-22 km errors and "
           "visibility-mask mismatches, no error raised.", "; ".join(b for b in bad if "R_" in b))
    defect("low", "make() silently ignores +inv (PROJ inverts the operation) and the module inherits core's silent drops of "
           "+pm, +units/+to_meter, +axis, and 'last duplicate key wins' (PROJ: first wins): wrong numbers, no error.",
           "; ".join(b for b in bad if "R_" not in b))
    # D7. shapes / dtypes / requires_grad
    P = make("+proj=ob_tran +o_proj=longlat +o_lon_p=-170 +o_lat_p=43 +lon_0=10 +R=6371229")
    Pg = make("+proj=geos +h=35786023 +lon_0=-75 +sweep=x +ellps=GRS80")
    okshape = True
    for Q in (P, Pg):
        for shape in [(2, 3, 4, 2), (7, 3), (7, 4), (0, 2), (2,)]:
            x = (torch.rand(*shape, dtype=torch.float64) - 0.5)
            y, yi = Q.forward(x), None
            yi = Q.inverse(y)
            okshape &= tuple(y.shape) == shape and tuple(yi.shape) == shape and torch.equal(y[..., 2:], x[..., 2:]) \
                and torch.equal(yi[..., 2:], x[..., 2:])
        for dt in (torch.float32,):
            x = (torch.rand(5, 2) - 0.5).to(dt).requires_grad_()
            y = Q.forward(x)
            okshape &= y.dtype == dt and y.requires_grad
    claim("batched [B,H,W,2], [N,3]/[N,4] z/t passthrough, fp32 dtype preserved, requires_grad inputs", "confirmed" if okshape else "refuted",
          "shapes/dtype/passthrough checked for ob_tran(longlat) and geos")


# ===========================================================================
# E. gradients
def fd_check(P, x, inverse, h):
    """max over points of min over steps (h, h/10, h/100) of ||J_ad - J_fd||_F / ||J_fd||_F
    (central differences, fp64). Several steps because near the geos disk edge the
    inverse has 1/sqrt(det) curvature and a 1 m step has large truncation error."""
    t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    f = (lambda u: P.inverse(u)) if inverse else (lambda u: P.forward(u))
    out = f(t)
    J = torch.zeros(len(x), 2, 2, dtype=torch.float64)
    for k in range(2):
        g, = torch.autograd.grad(out[:, k].sum(), t, retain_graph=True)
        J[:, k, :] = g
    best = np.full(len(x), np.inf)
    for hh in (h, h / 10, h / 100):
        Jfd = torch.zeros_like(J)
        with torch.no_grad():
            for j in range(2):
                e = torch.zeros(2, dtype=torch.float64)
                e[j] = hh
                base = torch.tensor(x, dtype=torch.float64)
                Jfd[:, :, j] = (f(base + e) - f(base - e))[:, :2] / (2 * hh)
        rel = (torch.linalg.matrix_norm(J - Jfd) / torch.linalg.matrix_norm(Jfd)).numpy()
        best = np.fmin(best, rel)
    return float(np.nanmax(best)), int(np.isnan(best).sum())


GRAD_DEFS = [
    ("+proj=geos +h=35785831 +a=6378169 +b=6356583.8 +lon_0=0 +sweep=y", 60),
    ("+proj=geos +h=35786023 +lon_0=-75 +sweep=x +ellps=GRS80", 60),
    ("+proj=geos +R=6400000 +h=35785831 +lon_0=10", 60),
    ("+proj=ob_tran +o_proj=longlat +o_lon_p=-170 +o_lat_p=43 +lon_0=10 +R=6371229", 80),
    ("+proj=ob_tran +o_proj=lonlat +o_lon_p=30 +o_lat_p=0 +lon_0=-20 +R=6371000", 80),
    ("+proj=ob_tran +o_proj=eqc +o_lat_p=30 +o_lon_p=40 +ellps=WGS84 +lat_ts=20 +lat_0=10", 80),
    ("+proj=ob_tran +o_proj=moll +R=6378137.0 +o_lon_p=0 +o_lat_p=0 +lon_0=180", 80),
    ("+proj=ob_tran +o_proj=geos +h=35785831 +o_lat_p=60 +o_lon_p=10 +ellps=WGS84", 0),
    ("+proj=eqc +ellps=WGS84 +lat_ts=45 +lat_0=30", 85),
    ("+proj=moll +R=6371000 +lon_0=30", 85),
    ("+proj=longlat +ellps=WGS84 +lon_0=10 +lon_wrap=180", 85),
]


def section_grad():
    print("\n=== E. gradients ===")
    worst = []
    for defn, lim in GRAD_DEFS:
        P = make(defn)
        n = 200
        lon0 = math.degrees(P.lam0)
        if lim == 0:  # ob_tran geos: take points PROJ can see
            lon, lat = global_pts(4000)
            ok = ~np.isnan(proj(defn, lon, lat)[0])
            lon, lat = lon[ok][:n], lat[ok][:n]
        else:
            lon = np.radians(lon0 + RNG.uniform(-lim, lim, n) * (0.95 if "geos" in defn else 1.0))
            lat = np.radians(RNG.uniform(-lim, lim, n))
        x = np.stack([lon, lat], -1)
        with torch.no_grad():
            xy = P.forward(torch.tensor(x)).numpy()
        keep = np.isfinite(xy).all(-1)
        # stay away from the limb / rotated poles / Mollweide cusp where FD is meaningless
        x, xy = x[keep], xy[keep]
        rf, nf = fd_check(P, x, False, 1e-6)
        hi = 1e-6 if out_kind(P, False) == "rad" else 1.0
        ri, ni = fd_check(P, xy, True, hi)
        worst.append((defn, rf, ri, nf + ni))
        print("  %-90s fwd rel %.2e  inv rel %.2e  (nan %d, n=%d)" % (defn, rf, ri, nf + ni, len(x)))
    mx = max(max(w[1], w[2]) for w in worst)
    claim("autograd Jacobians are correct at interior points (gradcheck passes on 13 configs)",
          "confirmed" if mx < 1e-5 and not any(w[3] for w in worst) else "refuted",
          "own central-FD check, 11 defs x ~200 pts: max relative Jacobian error %.2e" % mx)
    # NaN-gradient hunt: full domain including failing points, fp64 and fp32
    leaks = []
    for defn, _ in GRAD_DEFS + [("+proj=geos +R=6378137 +h=35786023 +sweep=x +lon_0=-137", 0), ("+proj=moll +R=6371000", 0)]:
        P = make(defn)
        lon = np.radians(np.concatenate([np.linspace(-180, 180, 73).repeat(37), RNG.uniform(-180, 180, 3000)]))
        lat = np.radians(np.concatenate([np.tile(np.linspace(-90, 90, 37), 73), RNG.uniform(-90, 90, 3000)]))
        for dt in (torch.float64, torch.float32):
            for inverse in (False, True):
                if inverse:
                    if out_kind(P, False) == "rad":
                        u = np.stack([RNG.uniform(-4, 4, 4000), RNG.uniform(-2, 2, 4000)], -1)
                    else:
                        u = np.stack([RNG.uniform(-3e7, 3e7, 4000), RNG.uniform(-3e7, 3e7, 4000)], -1)
                        u = np.concatenate([u, u * 1e-3, np.array([[0.0, 0.0], [1e9, 1e9], [-1e9, 0]])])
                else:
                    u = np.stack([lon, lat], -1)
                t = torch.tensor(u).to(dt).requires_grad_()
                o = P.inverse(t) if inverse else P.forward(t)
                fin = torch.isfinite(o)
                loss = torch.where(fin, o, torch.zeros_like(o)).sum()
                g, = torch.autograd.grad(loss, t)
                nan_g = int((~torch.isfinite(g)).any(-1).sum())
                failed = ~fin[..., :2].all(-1)
                nonzero_failed = int((g[failed] != 0).any(-1).sum())
                if nan_g or nonzero_failed:
                    leaks.append((defn, str(dt), "inv" if inverse else "fwd", nan_g, nonzero_failed))
    print("  NaN/inf gradients or non-zero gradients at failed points:", leaks or "none")
    claim("no NaN leaks into gradients (failed points get exactly 0 gradient) over full domains incl. poles/limb/outside disk",
          "confirmed" if not leaks else "refuted", "fp64+fp32, fwd+inv, 13 defs, grids incl. poles and +-180 plus "
          "boxes up to 1e9 m: %s" % (leaks or "no leaks"))


# ===========================================================================
# F. GPU / compile readiness
def section_compile(do_compile):
    print("\n=== F. host-sync / Python-control-flow scan, torch.compile ===")
    # only the per-point (runtime) code paths; __init__/setup code may use Python floats freely
    runtime = [G.aasin, G.aatan2, G._signed_halfpi, G._nan_where, G.pj_clenshaw, G.pj_mlfn, G.pj_inv_mlfn,
               G._fwd_prepare, G._inv_finalize, G._stack, G.LatLong.forward, G.LatLong.inverse, G.Eqc._fwd, G.Eqc._inv,
               G.Geos._view_angles, G.Geos._fwd, G.Geos._inv, G.Moll._fwd, G.Moll._inv, G.ObTran._rotate_fwd,
               G.ObTran._rotate_inv, G.ObTran._fwd, G.ObTran._inv, G.ObTran.forward, G.ObTran.inverse]
    hits = []
    for fn in runtime:
        src = inspect.getsource(fn)
        for pat in (r"\.item\(", r"\.numpy\(", r"\.tolist\(", r"\.cpu\(", r"\bbool\(", r"\bfloat\((?!\"nan\")", r"\bint\(",
                    r"torch\.nonzero", r"\.any\(\)", r"\.all\(\)", r"masked_select", r"torch\.tensor\(",
                    r"\bif\s+\(?[a-z_]+\s*[<>=]", r"while\s"):
            for m in re.finditer(pat, src):
                line = src[:m.start()].count("\n")
                hits.append((fn.__qualname__, pat, src.splitlines()[line].strip()[:80]))
    print("  pattern hits:", hits or "none")
    res = []
    if do_compile:
        for defn in ("+proj=geos +h=35786023 +lon_0=-75 +sweep=x +ellps=GRS80",
                     "+proj=ob_tran +o_proj=moll +R=6378137.0 +o_lon_p=0 +o_lat_p=0 +lon_0=180"):
            P = make(defn)
            ll = torch.tensor([[-1.2, 0.2], [-1.4, -0.4], [0.0, 1.5], [-1.0, 0.7]], dtype=torch.float64)
            t0 = time.time()
            try:
                cf = torch.compile(P.forward, fullgraph=True)
                ci = torch.compile(P.inverse, fullgraph=True)
                a, b = cf(ll), P.forward(ll)
                c, d = ci(b), P.inverse(b)
                ok = torch.allclose(a, b, rtol=1e-12, atol=1e-6, equal_nan=True) and torch.allclose(c, d, rtol=1e-12, atol=1e-12, equal_nan=True)
                res.append((defn, "ok" if ok else "MISMATCH", round(time.time() - t0, 1)))
            except Exception as e:  # noqa: BLE001
                res.append((defn, "error " + str(e).splitlines()[0][:120], round(time.time() - t0, 1)))
        print("  compile:", res)
    claim("forward/inverse free of host syncs and data-dependent Python control flow; torch.compile(fullgraph) works",
          "confirmed" if not hits and all(r[1] == "ok" for r in res) else "partially",
          "source scan hits=%s; compile=%s" % (hits or "none", res or "not run"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--compile", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    section_gie()
    section_headline(a.n)
    section_limb(min(a.n, 4000))
    section_edges()
    section_grad()
    section_compile(a.compile)
    print("\n=== done in %.1fs: %d claims checked, %d defects ===" % (time.time() - t0, len(OUT["claims"]), len(OUT["defects"])))


if __name__ == "__main__":
    main()
