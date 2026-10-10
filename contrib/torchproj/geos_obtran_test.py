"""Runnable checks for geos_obtran.py (no pytest). Exits non-zero on failure.

  1. PROJ's own gie vectors for geos, ob_tran, eqc (testing.gie_cases), plus
     the operation-level "expect failure" blocks (setup must fail).
  2. pyproj (PROJ 9.8.1) comparison on 1e5 random points per definition, fp64
     and fp32, forward and inverse, with exact NaN-mask agreement in fp64
     (geos: full disk + points straddling the limb / disk edge).
  3. torch.autograd.gradcheck (fp64) of forward and inverse at interior points.
  4. Gradient behaviour at singular points (no NaN may leak; failed points get
     exactly-zero gradients).
  5. Optional (--compile): torch.compile(fullgraph=True) on a small input.

Usage: python geos_obtran_test.py [--compile] [--n 100000]
"""
import argparse
import math
import sys
import time

import numpy as np
import torch

import testing
from geos_obtran import (DEG_TO_RAD, Geos, ObTran, angular_input_inverse,
                         angular_output, make)

torch.set_num_threads(1)
FAIL = []
REPORT = []


def check(cond, msg):
    if not cond:
        FAIL.append(msg)
        print("  FAIL:", msg)
    return cond


def d2r(deg):
    return np.asarray(deg, dtype=np.float64) * DEG_TO_RAD  # == unitconvert xy_in=deg


def r2d(rad):
    return np.asarray(rad, dtype=np.float64) / DEG_TO_RAD  # == unitconvert inverse


def run(P, xy, inverse, dtype=torch.float64):
    t = torch.as_tensor(np.stack(xy, -1), dtype=torch.float64).to(dtype)
    with torch.no_grad():
        out = P.inverse(t) if inverse else P.forward(t)
    out = out.double().numpy()
    return out[..., 0], out[..., 1]


# ---------------------------------------------------------------------------
# 1. gie
def gie_section():
    print("=== 1. gie test vectors (geos, ob_tran, eqc) ===")
    stats = dict(blocks=0, total=0, passed=0, skipped=0, failures=[])
    for name in ("geos", "ob_tran", "eqc"):
        for blk in testing.gie_cases(name):
            stats["blocks"] += 1
            op = blk["operation"]
            try:
                P = make(op)
                err = None
            except (ValueError, KeyError) as e:
                P, err = None, e
            if not blk["cases"]:
                # operation-level "expect failure errno ..." block: setup must fail,
                # and must fail in PROJ too.
                stats["total"] += 1
                try:
                    testing.pyproj_ref(op, [0.0], [0.0])
                    proj_fails = False
                except Exception:
                    proj_fails = True
                ok = P is None and proj_fails
                stats["passed"] += ok
                if not ok:
                    stats["failures"].append("%s: setup should fail" % op)
                print("  [%s] %-70s setup failure reproduced (%s)" % ("ok" if ok else "XX", op, err))
                continue
            if P is None:
                n = len(blk["cases"])
                stats["total"] += n
                stats["skipped"] += n
                print("  [skip] %-68s %d case(s): %s" % (op, n, err))
                continue
            for c in blk["cases"]:
                stats["total"] += 1
                inv = c["direction"] == "inv"
                acc = list(c["accept"])
                if (not inv) or angular_input_inverse(P):
                    acc[0], acc[1] = d2r(acc[0]), d2r(acc[1])
                t = torch.tensor([acc], dtype=torch.float64)
                out = (P.inverse(t) if inv else P.forward(t))[0].numpy()
                if c["expect"] == "failure":
                    ok = bool(np.isnan(out[0]))
                    d = float("nan")
                else:
                    e = c["expect"]
                    if angular_output(P, inv):  # gie: proj_lpz_dist on P's ellipsoid
                        d = float(testing.angular_error_m(e[0], e[1], r2d(out[0]), r2d(out[1]), a=P.a))
                    else:  # gie: proj_xyz_dist (raw units, e.g. radians for ob_tran+latlon)
                        d = float(math.hypot(out[0] - e[0], out[1] - e[1]))
                    for k in range(2, min(len(e), len(out))):
                        d = max(d, abs(out[k] - e[k]))
                    ok = d <= c["tolerance_m"]
                stats["passed"] += ok
                if not ok:
                    stats["failures"].append("%s %s %s -> %s (d=%g)" % (op, c["direction"], c["accept"], out, d))
            print("  [ok]  %-70s %d case(s)" % (op, len(blk["cases"])))
    for f in stats["failures"]:
        check(False, "gie: " + f)
    print("  gie: %d/%d passed, %d skipped, %d blocks" % (stats["passed"], stats["total"], stats["skipped"], stats["blocks"]))
    REPORT.append(("gie", stats))
    return stats


# ---------------------------------------------------------------------------
# 2. pyproj comparisons
RNG = np.random.default_rng(12345)


def global_lonlat(n):
    lon = RNG.uniform(-180, 180, n)
    lat = np.degrees(np.arcsin(RNG.uniform(-1, 1, n)))
    # edge values: poles, antimeridian, equator, prime meridian
    edge_lon = np.array([-180, 180, 0, -180, 180, 0, 90, -90, 179.9999999, -179.9999999])
    edge_lat = np.array([0, 0, 90, -90, 89.9999999, -90, 0, 0, 45, -45])
    return np.concatenate([lon, edge_lon]), np.concatenate([lat, edge_lat])


def geos_limb_points(P, n):
    """Points straddling the visibility limb of an ellipsoidal geos (degrees).
    For random azimuths, bisect along the great circle from the sub-satellite
    point the angular distance where PROJ's visibility test changes sign, then
    offset by +-10^U(-12,-3) rad."""
    rg, rp, rp2, rpi2 = P.radius_g, P.radius_p, P.radius_p2, P.radius_p_inv2

    def visible(lam, phi):
        ph = np.arctan(rp2 * np.tan(phi))
        r = rp / np.hypot(rp * np.cos(ph), np.sin(ph))
        Vx, Vy, Vz = r * np.cos(lam) * np.cos(ph), r * np.sin(lam) * np.cos(ph), r * np.sin(ph)
        return ((rg - Vx) * Vx - Vy * Vy - Vz * Vz * rpi2) >= 0

    def along(az, d):  # point at distance d (rad) from (0,0) along azimuth az
        phi = np.arcsin(np.sin(d) * np.cos(az))
        lam = np.arctan2(np.sin(az) * np.sin(d), np.cos(d))
        return lam, phi

    az = RNG.uniform(0, 2 * np.pi, n)
    lo, hi = np.full(n, 1.0), np.full(n, 1.6)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        v = visible(*along(az, mid))
        lo, hi = np.where(v, mid, lo), np.where(v, hi, mid)
    off = RNG.choice([-1.0, 1.0], n) * 10.0 ** RNG.uniform(-12, -3, n)
    lam, phi = along(az, lo + off)
    return np.degrees(lam) + math.degrees(P.lam0), np.degrees(phi), off


def geos_disk_xy(P, n_box, n_edge):
    """Image-plane points: a box 10% larger than the disk plus points
    straddling the disk edge (det = 0), in metres."""
    rg1, a = P.radius_g_1, P.a
    ext = 1.1 * rg1 * math.asin(1.0 / P.radius_g) * a
    xb = RNG.uniform(-ext, ext, n_box)
    yb = RNG.uniform(-ext, ext, n_box)
    t = torch.zeros(1, dtype=torch.float64)

    def det_ok(x, y):
        _, ph = P._inv(torch.as_tensor(x / a), torch.as_tensor(y / a))
        return ~np.isnan(ph.numpy())

    psi = RNG.uniform(0, 2 * np.pi, n_edge)
    lo, hi = np.zeros(n_edge), np.full(n_edge, ext)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        v = det_ok(mid * np.cos(psi), mid * np.sin(psi))
        lo, hi = np.where(v, mid, lo), np.where(v, hi, mid)
    rr = lo * (1 + RNG.choice([-1.0, 1.0], n_edge) * 10.0 ** RNG.uniform(-12, -3, n_edge))
    xe, ye = rr * np.cos(psi), rr * np.sin(psi)
    return np.concatenate([xb, xe]), np.concatenate([yb, ye])


N_EDGE = 10  # explicit edge values appended by global_lonlat()


def geos_inv_cond(P, x, y):
    """Conditioning of geos' inverse at (x, y) metres: a few-ulp error in
    det = b^2 - 4aC (catastrophic cancellation at the disk edge) moves the
    surface point by ~ a_m * eps * b^2 / (4 sqrt(a) sqrt(det)) metres. Returns
    (cond_m, det/b^2). Independent numpy re-derivation of geos.cpp:161-183."""
    rg1 = P.radius_g_1
    xs, ys = (np.asarray(x) - P.x0) / P.a, (np.asarray(y) - P.y0) / P.a
    if P.flip_axis:
        Vz = np.tan(ys / rg1)
        Vy = np.tan(xs / rg1) * np.hypot(1.0, Vz)
    else:
        Vy = np.tan(xs / rg1)
        Vz = np.tan(ys / rg1) * np.hypot(1.0, Vy)
    acoef = Vy * Vy + (Vz / P.radius_p) ** 2 + 1.0
    b = -2.0 * P.radius_g
    det = b * b - 4 * acoef * P.C
    eps = np.finfo(np.float64).eps
    cond = P.a * eps * b * b / (4 * np.sqrt(acoef) * np.sqrt(np.maximum(det, 1e-300)))
    return cond, det / (b * b)


def compare(defn, fwd_pts, inv_pts, label, require_mask=True, tol64=1e-6, fwd_edge=N_EDGE,
            inv_cond=None, far=None):
    """Compare torch vs PROJ. fwd_pts in degrees, inv_pts in projected units
    (or radians for ob_tran+latlong). Returns dict of errors.
    fp32 statistics skip the `fwd_edge` trailing explicit edge points (exact
    +-180 deg / +-90 deg inputs are ambiguous once rounded to float32: e.g.
    float32(pi) > pi flips the side of the antimeridian).
    inv_cond(P, x, y) -> per-point conditioning (m): fp64 inverse errors must be
    <= 1e-6 + 16 * cond instead of tol64 (geos near the disk edge).
    far(direction, u, v, rx, ry) -> bool mask of points away from a singularity
    or failure boundary; used for extra fp32 statistics ("_far")."""
    P = make(defn)
    res = {"definition": defn, "label": label}
    for direction, pts in (("fwd", fwd_pts), ("inv", inv_pts)):
        if pts is None:
            continue
        inv = direction == "inv"
        u, v = pts
        rx, ry = testing.pyproj_ref(defn, u, v, inverse=inv)
        ref_bad = ~(np.isfinite(rx) & np.isfinite(ry))
        tin = (u, v) if inv else (d2r(u), d2r(v))
        if inv and angular_input_inverse(P):
            tin = (d2r(u), d2r(v))
        for dtype, tag in ((torch.float64, "64"), (torch.float32, "32")):
            tx, ty = run(P, tin, inv, dtype)
            t_bad = np.isnan(tx) | np.isnan(ty)
            mism = int((t_bad != ref_bad).sum())
            both = ~t_bad & ~ref_bad
            if inv:  # output radians -> degrees; pyproj already returned degrees
                err = testing.angular_error_m(rx[both], ry[both], r2d(tx[both]), r2d(ty[both]), a=P.a)
            elif angular_output(P, False):  # latlong: pyproj converts radians to degrees
                err = testing.angular_error_m(rx[both], ry[both], r2d(tx[both]), r2d(ty[both]), a=P.a)
            elif getattr(P, "whatever_units", False):  # ob_tran over latlong: raw radians
                err = testing.angular_error_m(r2d(rx[both]), r2d(ry[both]), r2d(tx[both]), r2d(ty[both]), a=P.a)
            else:
                err = np.hypot(tx[both] - rx[both], ty[both] - ry[both])
            if tag == "32" and not inv and fwd_edge:
                keep = np.ones(len(u), bool)
                keep[-fwd_edge:] = False
                err_all, err = err, err[keep[both]]
            if tag == "32" and far is not None:
                fm = np.asarray(far(direction, u, v, rx, ry), bool)
                if not inv and fwd_edge:
                    fm = fm & keep
                res["%s_mask_mismatch_32_far" % direction] = int(((t_bad != ref_bad) & fm).sum())
                ef = (err_all if (not inv and fwd_edge) else err)[fm[both]]
                res["%s_32_far" % direction] = float(ef.max()) if ef.size else 0.0
            emax = float(err.max()) if err.size else 0.0
            res["%s_%s" % (direction, tag)] = emax
            res["%s_mask_mismatch_%s" % (direction, tag)] = mism
            res["%s_n" % direction] = int(len(u))
            res["%s_nfail_ref" % direction] = int(ref_bad.sum())
            if inv and inv_cond is not None:
                cond, rel = inv_cond(P, u[both], v[both])
                interior = rel >= (1e-4 if tag == "64" else 1e-3)
                res["inv_interior_%s" % tag] = float(err[interior].max()) if interior.any() else 0.0
                res["inv_edge_%s" % tag] = float(err[~interior].max()) if (~interior).any() else 0.0
                if tag == "64":
                    ratio = err / (1e-6 + cond)
                    res["inv_max_err_over_cond"] = float(ratio.max())
                    check(bool((err <= 1e-6 + 16 * cond).all()),
                          "%s inv fp64 error exceeds 16x edge conditioning (max ratio %.3g)" % (label, ratio.max()))
                    check(res["inv_interior_64"] <= tol64, "%s inv fp64 interior max err %.3g" % (label, res["inv_interior_64"]))
            elif tag == "64":
                check(emax <= tol64, "%s %s fp64 max err %.3g m > %g" % (label, direction, emax, tol64))
            if tag == "64" and require_mask:
                check(mism == 0, "%s %s fp64 NaN-mask mismatches: %d" % (label, direction, mism))
    print("  %-26s fwd64 %.2e fwd32 %.2e | inv64 %.2e inv32 %.2e | NaN-mask mismatch fp64 f/i %s/%s fp32 f/i %s/%s | n=%s/%s fails(PROJ) f/i %s/%s" % (
        label, res.get("fwd_64", float("nan")), res.get("fwd_32", float("nan")),
        res.get("inv_64", float("nan")), res.get("inv_32", float("nan")),
        res.get("fwd_mask_mismatch_64", "-"), res.get("inv_mask_mismatch_64", "-"),
        res.get("fwd_mask_mismatch_32", "-"), res.get("inv_mask_mismatch_32", "-"),
        res.get("fwd_n", "-"), res.get("inv_n", "-"),
        res.get("fwd_nfail_ref", "-"), res.get("inv_nfail_ref", "-")))
    if inv_cond is not None:
        print("     inverse breakdown: fp64 interior(det/b^2>=1e-4) %.2e, edge band %.2e, max err/cond %.2f | "
              "fp32 interior(det/b^2>=1e-3) %.2e, edge band %.2e" % (
                  res["inv_interior_64"], res["inv_edge_64"], res["inv_max_err_over_cond"],
                  res["inv_interior_32"], res["inv_edge_32"]))
    if far is not None:
        print("     fp32 away from singularities/boundaries: fwd err %.2e mask mismatches %s | inv err %.2e mask mismatches %s" % (
            res.get("fwd_32_far", float("nan")), res.get("fwd_mask_mismatch_32_far", "-"),
            res.get("inv_32_far", float("nan")), res.get("inv_mask_mismatch_32_far", "-")))
    REPORT.append(("pyproj", res))
    return res


def fwd_valid_xy(defn, n):
    """Projected points obtained by PROJ from random lon/lat (for inverses)."""
    lon, lat = global_lonlat(n)
    x, y = testing.pyproj_ref(defn, lon, lat)
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def pyproj_section(N):
    print("=== 2. pyproj comparison (max error in metres; %d random points + edge values) ===" % N)
    results = []
    # --- geos: the two satellites of the spec, full disk + limb-straddling points
    for label, defn in (
            ("MSG SEVIRI (geos ell, y)", "+proj=geos +h=35785831 +a=6378169 +b=6356583.8 +lon_0=0 +sweep=y"),
            ("GOES-16 ABI (geos ell, x)", "+proj=geos +h=35786023 +lon_0=-75 +sweep=x +ellps=GRS80")):
        P = make(defn)
        llon, llat, off = geos_limb_points(P, N // 2)
        lon, lat = global_lonlat(N // 2)  # edge values last -> excluded from fp32 stats
        fwd = (np.concatenate([llon, lon]), np.concatenate([llat, lat]))
        far_fwd = np.concatenate([np.abs(off) >= 1e-5, np.ones(len(lon), bool)])
        inv = geos_disk_xy(P, N // 2, N // 2)

        def far(direction, u, v, rx, ry, P=P, far_fwd=far_fwd):
            if direction == "fwd":
                return far_fwd  # >= 1e-5 rad from the limb
            return np.abs(geos_inv_cond(P, u, v)[1]) >= 1e-3  # |det|/b^2 >= 1e-3
        results.append(compare(defn, fwd, inv, label, inv_cond=geos_inv_cond, far=far))
        # ultra-close points (|offset| 1e-17..1e-12 rad): diagnostics only
        Pn = make(defn)
        ulon, ulat = geos_limb_points_ultra(Pn, 20000)
        rx, _ = testing.pyproj_ref(defn, ulon, ulat)
        tx, _ = run(Pn, (d2r(ulon), d2r(ulat)), False)
        mm = int((np.isnan(tx) != ~np.isfinite(rx)).sum())
        print("     ultra-close limb points (offset 1e-17..1e-12 rad): %d/20000 NaN-mask mismatches (diagnostic)" % mm)
        REPORT.append(("ultra", {"label": label, "mismatch": mm, "n": 20000}))
    # --- geos spherical (no visibility test in PROJ's spherical forward!)
    for label, defn in (
            ("geos sphere sweep=y", "+proj=geos +R=6400000 +h=35785831 +lon_0=10"),
            ("geos sphere sweep=x", "+proj=geos +R=6378137 +h=35786023 +sweep=x +lon_0=-137 +x_0=100 +y_0=-50")):
        P = make(defn)
        fwd = global_lonlat(N)
        inv = geos_disk_xy(P, N // 2, N // 2)
        far = lambda d, u, v, rx, ry, P=P: (np.ones(len(u), bool) if d == "fwd"  # noqa: E731
                                            else np.abs(geos_inv_cond(P, u, v)[1]) >= 1e-3)
        results.append(compare(defn, fwd, inv, label, inv_cond=geos_inv_cond, far=far))
    # --- ob_tran over latlong: forward output and inverse input are RADIANS
    for label, defn in (
            ("rotated pole COSMO", "+proj=ob_tran +o_proj=longlat +o_lon_p=-170 +o_lat_p=43 +lon_0=10 +R=6371229"),
            ("ob_tran latlon (gie)", "+proj=ob_tran +R=6400000 +o_proj=latlon +o_lon_p=20 +o_lat_p=20 +lon_0=180"),
            ("ob_tran o_alpha", "+proj=ob_tran +o_proj=longlat +o_alpha=30 +o_lon_c=20 +o_lat_c=40 +R=6371000"),
            ("ob_tran transverse ll", "+proj=ob_tran +o_proj=lonlat +o_lon_p=30 +o_lat_p=0 +lon_0=-20 +R=6371000")):
        fwd = global_lonlat(N)
        rl = RNG.uniform(-math.pi, math.pi, N)
        rp = np.arcsin(RNG.uniform(-1, 1, N))
        # fp32 "far": more than 1 deg from the rotated poles (asin conditioning)
        far = lambda d, u, v, rx, ry: (np.abs(ry) <= math.radians(89.0) if d == "fwd"  # noqa: E731
                                       else np.abs(v) <= math.radians(89.0))
        results.append(compare(defn, fwd, (rl, rp), label, far=far))
    # --- ob_tran over projections (classic units, a-scaled, x_0/y_0)
    for label, defn in (
            ("ob_tran eqc sphere", "+proj=ob_tran +o_proj=eqc +o_lon_p=-170 +o_lat_p=43 +lon_0=10 +R=6371229 +x_0=1000 +y_0=-2000"),
            ("ob_tran eqc WGS84", "+proj=ob_tran +o_proj=eqc +o_lat_p=30 +o_lon_p=40 +ellps=WGS84 +lat_ts=20 +lat_0=10"),
            ("ob_tran eqc equator pts", "+proj=ob_tran +o_proj=eqc +o_lon_1=-30 +o_lat_1=20 +o_lon_2=40 +o_lat_2=-10 +R=6371000"),
            ("ob_tran moll transverse", "+proj=ob_tran +o_proj=moll +R=6378137.0 +o_lon_p=0 +o_lat_p=0 +lon_0=180"),
            ("ob_tran geos ell", "+proj=ob_tran +o_proj=geos +h=35785831 +o_lat_p=60 +o_lon_p=10 +ellps=WGS84")):
        fwd = global_lonlat(N)
        xy = fwd_valid_xy(defn, N)
        # geos link: its inverse inherits the disk-edge conditioning
        cond = (lambda P, x, y: geos_inv_cond(P.link, x, y)) if "o_proj=geos" in defn else None
        results.append(compare(defn, fwd, xy, label, inv_cond=cond))
    # --- eqc, latlong, moll on their own
    for label, defn in (
            ("eqc sphere", "+proj=eqc +R=6400000 +lat_ts=30 +lat_0=10 +lon_0=-20 +x_0=500000 +y_0=-100"),
            ("eqc WGS84", "+proj=eqc +ellps=WGS84 +lat_ts=45 +lat_0=30 +lon_0=15"),
            ("eqc GRS80 default", "+proj=eqc"),
            ("moll", "+proj=moll +R=6371000 +lon_0=30")):
        fwd = global_lonlat(N)
        xy = fwd_valid_xy(defn, N)
        # plus a box of arbitrary projected points (eqc beyond the poles, moll outside the ellipse)
        P = make(defn)
        ext = 2.2e7
        xb, yb = RNG.uniform(-ext, ext, N // 10), RNG.uniform(-ext, ext, N // 10)
        inv = (np.concatenate([xy[0], xb]), np.concatenate([xy[1], yb]))
        results.append(compare(defn, fwd, inv, label))
    for label, defn in (
            ("longlat lon_0", "+proj=longlat +ellps=WGS84 +lon_0=10"),
            ("latlong lon_wrap=180", "+proj=latlong +lon_wrap=180")):
        fwd = global_lonlat(N)
        inv = global_lonlat(N)  # degrees; angular input
        results.append(compare(defn, fwd, inv, label))
    return results


def geos_limb_points_ultra(P, n):
    rg, rp, rp2, rpi2 = P.radius_g, P.radius_p, P.radius_p2, P.radius_p_inv2

    def visible(lam, phi):
        ph = np.arctan(rp2 * np.tan(phi))
        r = rp / np.hypot(rp * np.cos(ph), np.sin(ph))
        Vx, Vy, Vz = r * np.cos(lam) * np.cos(ph), r * np.sin(lam) * np.cos(ph), r * np.sin(ph)
        return ((rg - Vx) * Vx - Vy * Vy - Vz * Vz * rpi2) >= 0

    def along(az, d):
        return (np.arctan2(np.sin(az) * np.sin(d), np.cos(d)), np.arcsin(np.sin(d) * np.cos(az)))

    az = RNG.uniform(0, 2 * np.pi, n)
    lo, hi = np.full(n, 1.0), np.full(n, 1.6)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        v = visible(*along(az, mid))
        lo, hi = np.where(v, mid, lo), np.where(v, hi, mid)
    off = RNG.choice([-1.0, 1.0], n) * 10.0 ** RNG.uniform(-17, -12, n)
    lam, phi = along(az, lo + off)
    return np.degrees(lam) + math.degrees(P.lam0), np.degrees(phi)


# ---------------------------------------------------------------------------
# 2b. composition: explicit link instance, and a projection from another module
def composition_section(N):
    print("=== 2b. ob_tran composition ===")
    from geos_obtran import Eqc, register_projection
    base = "+proj=ob_tran +o_lat_p=43 +o_lon_p=-170 +lon_0=10 +R=6371229 +lat_ts=20"
    P1 = make(base + " +o_proj=eqc")
    P2 = ObTran(base, link=Eqc(base.replace("+proj=ob_tran", "+proj=eqc")))
    lon, lat = global_lonlat(1000)
    ll = torch.tensor(np.stack([d2r(lon), d2r(lat)], -1))
    same = torch.equal(P1.forward(ll), P2.forward(ll))
    check(same, "ObTran(link=Eqc(...)) differs from +o_proj=eqc")
    print("  explicit link instance == +o_proj=eqc (bitwise): %s" % same)
    try:  # sibling agent's module; informational only (not ours to fail on)
        import tmerc
        register_projection("tmerc", tmerc.TransverseMercator)
        defn = "+proj=ob_tran +o_proj=tmerc +o_lat_p=50 +o_lon_p=20 +lon_0=5 +R=6371000"
        fwd = global_lonlat(N)
        res = compare_info(defn, fwd)
        print("  o_proj=tmerc (from tmerc.py, informational): fwd fp64 max err %.2e m, NaN-mask mismatches %d, "
              "PROJ failures %d/%d" % res)
        REPORT.append(("composition_tmerc", res))
    except Exception as e:  # noqa: BLE001
        print("  o_proj=tmerc composition skipped: %s" % e)


def compare_info(defn, fwd):
    P = make(defn)
    rx, ry = testing.pyproj_ref(defn, *fwd)
    tx, ty = run(P, (d2r(fwd[0]), d2r(fwd[1])), False)
    rb, tb = ~(np.isfinite(rx) & np.isfinite(ry)), np.isnan(tx) | np.isnan(ty)
    both = ~rb & ~tb
    err = np.hypot(tx[both] - rx[both], ty[both] - ry[both])
    return float(err.max()), int((rb != tb).sum()), int(rb.sum()), len(rx)


# ---------------------------------------------------------------------------
# 3. gradcheck
def gradcheck_section():
    print("=== 3. torch.autograd.gradcheck (float64, interior points) ===")
    cases = [
        ("+proj=geos +h=35785831 +a=6378169 +b=6356583.8 +lon_0=0 +sweep=y", (10, 30), (-40, 50)),
        ("+proj=geos +h=35786023 +lon_0=-75 +sweep=x +ellps=GRS80", (-110, -40), (-50, 50)),
        ("+proj=geos +R=6400000 +h=35785831 +lon_0=10", (-30, 50), (-50, 50)),
        ("+proj=geos +R=6378137 +h=35786023 +sweep=x +lon_0=-137", (-170, -100), (-50, 50)),
        ("+proj=ob_tran +o_proj=longlat +o_lon_p=-170 +o_lat_p=43 +lon_0=10 +R=6371229", (-20, 40), (30, 65)),
        ("+proj=ob_tran +R=6400000 +o_proj=latlon +o_lon_p=20 +o_lat_p=20 +lon_0=180", (-60, 60), (-60, 60)),
        ("+proj=ob_tran +o_proj=lonlat +o_lon_p=30 +o_lat_p=0 +lon_0=-20 +R=6371000", (-60, 60), (-60, 60)),
        ("+proj=ob_tran +o_proj=eqc +o_lat_p=30 +o_lon_p=40 +ellps=WGS84 +lat_ts=20 +lat_0=10", (-60, 60), (-60, 60)),
        ("+proj=ob_tran +o_proj=moll +R=6378137.0 +o_lon_p=0 +o_lat_p=0 +lon_0=180", (-60, 60), (-60, 60)),
        ("+proj=eqc +R=6400000 +lat_ts=30 +lat_0=10", (-170, 170), (-85, 85)),
        ("+proj=eqc +ellps=WGS84 +lat_ts=45 +lat_0=30", (-170, 170), (-85, 85)),
        ("+proj=moll +R=6371000 +lon_0=30", (-120, 120), (-80, 80)),
        ("+proj=longlat +ellps=WGS84 +lon_0=10", (-120, 120), (-80, 80)),
    ]
    out = []
    g = torch.Generator().manual_seed(7)
    for defn, (l0, l1), (p0, p1) in cases:
        P = make(defn)
        n = 6
        lon = l0 + (l1 - l0) * torch.rand(n, generator=g, dtype=torch.float64)
        lat = p0 + (p1 - p0) * torch.rand(n, generator=g, dtype=torch.float64)
        ll = torch.stack([lon, lat], -1) * DEG_TO_RAD
        whatever = getattr(P, "whatever_units", False) or getattr(P, "RIGHT_UNITS", "") == "radians"
        scale = 1.0 if whatever else P.a
        with torch.no_grad():
            xy = P.forward(ll) / scale
        assert torch.isfinite(xy).all(), defn
        f = lambda t: P.forward(t) / scale  # noqa: E731  unit-scaled for well-conditioned FD
        fi = lambda t: P.inverse(t * scale)  # noqa: E731
        okf = torch.autograd.gradcheck(f, (ll.clone().requires_grad_(),), eps=1e-6, atol=1e-6, rtol=1e-4,
                                       raise_exception=False)
        oki = torch.autograd.gradcheck(fi, (xy.clone().requires_grad_(),), eps=1e-6, atol=1e-6, rtol=1e-4,
                                       raise_exception=False)
        check(okf, "gradcheck fwd " + defn)
        check(oki, "gradcheck inv " + defn)
        out.append((defn, okf, oki))
        print("  fwd %-5s inv %-5s %s" % (okf, oki, defn))
    REPORT.append(("gradcheck", out))
    return out


# ---------------------------------------------------------------------------
# 4. gradients at singular points
def _grads(P, pts, inverse):
    t = torch.tensor(pts, dtype=torch.float64, requires_grad=True)
    out = P.inverse(t) if inverse else P.forward(t)
    fin = torch.isfinite(out)
    loss = torch.where(fin, out, torch.zeros_like(out)).sum()  # standard NaN-masked loss
    (g,) = torch.autograd.grad(loss, t)
    return out.detach(), g


def singular_section():
    print("=== 4. gradients at singular / failing points ===")
    notes = []
    D = DEG_TO_RAD

    def show(label, P, pts, inverse, expect_zero_rows=()):
        out, g = _grads(P, pts, inverse)
        finite = bool(torch.isfinite(g).all())
        check(finite, "non-finite gradient: " + label)
        for r in expect_zero_rows:
            check(bool((g[r] == 0).all()), "gradient of failed point should be 0: %s row %d" % (label, r))
        gmax = g.abs().max(-1).values
        txt = "%s: outputs %s | max|grad| per point %s" % (
            label, np.array2string(out.numpy()[:, :2], precision=4, max_line_width=300).replace("\n", ""),
            np.array2string(gmax.numpy(), precision=3, max_line_width=300))
        print("  " + txt)
        notes.append(txt)

    # geos ellipsoidal forward: visible, just inside/outside limb, far side, pole
    G = make("+proj=geos +h=35785831 +a=6378169 +b=6356583.8 +lon_0=0 +sweep=y")
    show("geos-ell fwd [centre, (81.29,0) visible, (81.31,0) invisible, (180,0) far side, (0,90) pole]",
         G, [[0.0, 0.0], [81.29 * D, 0.0], [81.31 * D, 0.0], [math.pi, 0.0], [0.0, math.pi / 2]],
         False, expect_zero_rows=(2, 3, 4))
    # geos inverse: centre, near edge (det -> 0+), outside disk (det < 0)
    ext = G.radius_g_1 * math.asin(1.0 / G.radius_g) * G.a
    show("geos-ell inv [centre, 0.999999999 edge, 1.000001 edge (miss), far outside]",
         G, [[0.0, 0.0], [ext * 0.999999999, 0.0], [ext * 1.000001, 0.0], [2 * ext, 2 * ext]],
         True, expect_zero_rows=(2, 3))
    # geos spherical forward on the far side: PROJ returns finite values, so do we
    S = make("+proj=geos +R=6400000 +h=35785831")
    show("geos-sph fwd [far side (180,0), (100,10)] (PROJ: no visibility test)", S,
         [[math.pi, 0.0], [100 * D, 10 * D]], False)
    # ob_tran rotated pole: the point that maps onto the rotated north pole,
    # its antipode, geographic poles and the antimeridian
    R = make("+proj=ob_tran +o_proj=longlat +o_lon_p=-170 +o_lat_p=43 +lon_0=10 +R=6371229")
    # north rotated pole is at geographic lat = o_lat_p, lon = lon_0 + 180 (o_lon_p rotates the output)
    pole = [(10 + 180 - 360) * D, 43 * D]
    show("ob_tran fwd [rotated N pole, rotated S pole, geo N pole, geo S pole, antimeridian +-180]",
         R, [pole, [10 * D, -43 * D], [0.0, math.pi / 2], [0.0, -math.pi / 2],
             [math.pi, 10 * D], [-math.pi, 10 * D]], False)
    show("ob_tran inv [rlat=+90, rlat=-90, rlon=+-pi]",
         R, [[0.3, math.pi / 2], [0.3, -math.pi / 2], [math.pi, 0.2], [-math.pi, 0.2]], True)
    # eqc at poles / antimeridian
    E = make("+proj=eqc +ellps=WGS84 +lat_ts=45")
    show("eqc-ell fwd [N pole, S pole, lon=+180, lon=-180]", E,
         [[0.3, math.pi / 2], [0.3, -math.pi / 2], [math.pi, 0.1], [-math.pi, 0.1]], False)
    # moll forward at the pole (Newton does not converge in 30 its -> snapped to +-pi/2)
    M = make("+proj=moll +R=6371000")
    show("moll fwd [N pole, 89.99999, equator]", M,
         [[0.5, math.pi / 2], [0.5, 89.99999 * D], [0.5, 0.0]], False)
    show("moll inv [outside ellipse (fails), y beyond C_y (fails), centre]", M,
         [[2.0e7, 0.0], [0.0, 1.0e7], [0.0, 0.0]], True, expect_zero_rows=(0, 1))
    # composition: failure inside the link (geos misses the earth) must not leak NaN
    OG = make("+proj=ob_tran +o_proj=geos +h=35785831 +o_lat_p=60 +o_lon_p=10 +ellps=WGS84")
    show("ob_tran(geos) inv [disk centre, outside disk (link fails)]", OG,
         [[1000.0, 2000.0], [6.0e6, 6.0e6]], True, expect_zero_rows=(1,))
    show("ob_tran(geos) fwd [visible, invisible (link fails)]", OG,
         [[0.2, 0.9], [2.5, -0.5]], False)
    # out-of-range latitude (core.Projection marks it NaN)
    show("geos fwd [lat = 90.1 deg (invalid)]", G, [[0.0, 90.1 * D]], False, expect_zero_rows=(0,))
    REPORT.append(("singular", notes))
    return notes


# ---------------------------------------------------------------------------
# 5. torch.compile (optional)
def compile_section():
    print("=== 5. torch.compile(fullgraph=True), small input ===")
    res = []
    for defn in ("+proj=geos +h=35785831 +a=6378169 +b=6356583.8 +lon_0=0 +sweep=y",
                 "+proj=ob_tran +o_proj=longlat +o_lon_p=-170 +o_lat_p=43 +lon_0=10 +R=6371229",
                 "+proj=ob_tran +o_proj=moll +R=6378137.0 +o_lon_p=0 +o_lat_p=0 +lon_0=180",
                 "+proj=eqc +ellps=WGS84 +lat_ts=45 +lat_0=30"):
        P = make(defn)
        ll = torch.tensor([[0.1, 0.2], [0.3, -0.4], [1.5, 0.1], [-0.2, 0.7]], dtype=torch.float64)
        t0 = time.time()
        try:
            cf = torch.compile(P.forward, fullgraph=True)
            ci = torch.compile(P.inverse, fullgraph=True)
            xy = cf(ll)
            back = ci(P.forward(ll))
            ok = (torch.allclose(xy, P.forward(ll), equal_nan=True, rtol=1e-12, atol=1e-6)
                  and torch.allclose(back, P.inverse(P.forward(ll)), equal_nan=True, rtol=1e-12, atol=1e-12))
            msg = "ok" if ok else "MISMATCH"
        except Exception as e:  # report, do not fail the suite on compile infra issues
            ok, msg = False, "error: %s" % (str(e).splitlines()[0][:200])
        print("  %-8s %.1fs %s" % (msg, time.time() - t0, defn))
        res.append((defn, msg))
    REPORT.append(("compile", res))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    gie_section()
    pyproj_section(args.n)
    composition_section(args.n)
    gradcheck_section()
    singular_section()
    if args.compile:
        compile_section()
    print("=== summary: %d failure(s), %.1fs ===" % (len(FAIL), time.time() - t0))
    for f in FAIL:
        print("  -", f)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
