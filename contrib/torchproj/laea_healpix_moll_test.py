"""Validation of laea_healpix_moll.py (laea, healpix, rhealpix, moll, wag4, wag5)
against PROJ itself: gie vectors (test/gie/*.gie) and pyproj 3.8 / PROJ 9.8.1.

Run:  python laea_healpix_moll_test.py [--n 100000] [--no-compile] [--inductor]
Prints a summary and exits non-zero on failure.
"""
import math
import sys
import time

import numpy as np
import torch

import testing
import laea_healpix_moll as M

torch.set_num_threads(1)
N_RANDOM = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 100_000
DO_COMPILE = "--no-compile" not in sys.argv
DO_INDUCTOR = "--inductor" in sys.argv
FP64_TOL_M = 2e-5   # pass threshold vs PROJ in float64 (metres)
# Within POLE_ZONE_DEG of a pole (or of the antipode of an azimuthal centre) PROJ's
# inverse formulas evaluate asin(v) with v within a few ulp of 1; a 1-ulp difference
# between torch's and glibc's sin/cos/asin then becomes sqrt(2*1.1e-16) = 1.5e-8 rad
# (~0.1 m). There we only require FP64_TOL_SINGULAR_M.
POLE_ZONE_DEG = 1e-2   # ~1.1 km; error there ~ 1ulp * R / delta
FP64_TOL_SINGULAR_M = 0.5
FAILURES = []
RESULTS = {}


def fail(msg):
    FAILURES.append(msg)
    print("   FAIL:", msg)


def check(cond, msg):
    if not cond:
        fail(msg)
    return cond


def t64(a):
    return torch.tensor(np.asarray(a, dtype=float), dtype=torch.float64)


# =============================================================================
# 1. gie vectors
# =============================================================================
def setup_failure_expected(block):
    """gie: an 'expect failure' directly after the operation (before any accept)."""
    found = False
    for cmd, args in testing._logical_lines(testing.GIE_DIR / block["file"]):
        if cmd == "operation":
            if found:
                return False
            found = testing._clean_operation(args) == block["operation"]
        elif found and cmd == "accept":
            return False
        elif found and cmd == "expect":
            return args.strip().startswith("failure")
    return False


def _all_nan(v):
    return bool(np.isnan(np.asarray(v, dtype=float)).all())


def run_gie():
    print("=" * 78)
    print("1. gie test vectors (PROJ test/gie/*.gie)")
    st = dict(blocks=0, total=0, passed=0, skipped=0, failures=[])
    for name in ("laea", "healpix", "rhealpix", "moll", "wag4", "wag5"):
        for blk in testing.gie_cases(name):
            st["blocks"] += 1
            op = blk["operation"]
            ncase = len(blk["cases"]) + len(blk["roundtrips"])
            try:
                P = M.make(op)
            except (ValueError, KeyError) as err:
                exp_fail = setup_failure_expected(blk)
                st["total"] += max(ncase, 1)
                if exp_fail and ncase == 0:
                    st["passed"] += 1
                    print(f"  [ok]   {blk['file']:<18} {op:<62} setup failure as expected ({err})")
                else:
                    st["failures"].append(f"{op}: unexpected setup error {err}")
                    print(f"  [FAIL] {blk['file']:<18} {op:<62} unexpected setup error {err}")
                continue
            if ncase == 0 and setup_failure_expected(blk):
                st["total"] += 1
                st["failures"].append(f"{op}: setup should fail")
                print(f"  [FAIL] {op}: setup should fail")
                continue
            nok, worst = 0, 0.0
            a = P.a_ellps
            for c in blk["cases"]:
                st["total"] += 1
                acc = c["accept"]
                if c["direction"] == "fwd":
                    out = P.forward(t64([[math.radians(acc[0]), math.radians(acc[1])]]))[0].numpy()
                    if c["expect"] == "failure":
                        ok, d = bool(np.isnan(out[:2]).all()), 0.0
                    elif _all_nan(c["expect"][:2]):
                        ok, d = _all_nan(out[:2]), 0.0
                    else:
                        d = float(np.hypot(out[0] - c["expect"][0], out[1] - c["expect"][1]))
                        ok = d <= c["tolerance_m"]
                else:
                    out = P.inverse(t64([[acc[0], acc[1]]]))[0].numpy()
                    if c["expect"] == "failure":
                        ok, d = bool(np.isnan(out[:2]).all()), 0.0
                    elif _all_nan(c["expect"][:2]):
                        ok, d = _all_nan(out[:2]), 0.0
                    else:
                        d = float(testing.angular_error_m(math.degrees(out[0]), math.degrees(out[1]),
                                                          c["expect"][0], c["expect"][1], a=a))
                        ok = d <= c["tolerance_m"]
                if c["tolerance_m"] > 0:
                    worst = max(worst, d / c["tolerance_m"])
                nok += ok
                st["passed"] += ok
                if not ok:
                    st["failures"].append(f"{blk['file']}: {op}: {c['direction']} {acc} -> err {d:.3e} m "
                                          f"(tol {c['tolerance_m']:.1e}) got {out}")
            for r in blk["roundtrips"]:   # proj_roundtrip (trans.cpp:590-630), fwd direction
                st["total"] += 1
                acc = r["accept"]
                org = t64([[math.radians(acc[0]), math.radians(acc[1])]])
                xy = P.forward(org)
                for _ in range(r["n"] - 1):
                    xy = P.forward(P.inverse(xy))
                lp = P.inverse(xy)[0].numpy()
                if _all_nan(acc[:2]):
                    ok, d = _all_nan(lp[:2]), 0.0
                else:
                    d = float(testing.angular_error_m(acc[0], acc[1], math.degrees(lp[0]),
                                                      math.degrees(lp[1]), a=a))
                    ok = d <= r["tolerance_m"]
                    worst = max(worst, d / r["tolerance_m"])
                nok += ok
                st["passed"] += ok
                if not ok:
                    st["failures"].append(f"{op}: roundtrip {acc} x{r['n']} err {d:.3e} m")
            print(f"  [{'ok' if nok == ncase else 'FAIL'}]   {blk['file']:<18} {op:<62} {nok}/{ncase}"
                  f"  worst err/tol={worst:.1e}")
    print(f"  gie: blocks={st['blocks']} cases={st['total']} passed={st['passed']} skipped={st['skipped']}")
    for f in st["failures"]:
        fail("gie " + f)
    RESULTS["gie"] = st


# =============================================================================
# 2. pyproj comparison, 1e5 random points (+ special points), fp64 and fp32
# =============================================================================
DEFS = [
    # (definition, short label)
    ("+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80", "laea EPSG:3035 (ell, OBLIQ)"),
    ("+proj=laea +ellps=GRS80 +lon_0=-30", "laea ell EQUIT"),
    ("+proj=laea +lat_0=90 +lon_0=-150 +datum=WGS84", "laea ell N_POLE"),
    ("+proj=laea +lat_0=-90 +lon_0=0 +ellps=GRS80", "laea ell S_POLE"),
    ("+proj=laea +a=6378137 +rf=30 +lat_0=40", "laea ell |n|>=0.01 (Newton authalic inverse)"),
    ("+proj=laea +R=6371000 +lat_0=45 +lon_0=10", "laea sph OBLIQ"),
    ("+proj=laea +R=6371000", "laea sph EQUIT"),
    ("+proj=laea +R=6371000 +lat_0=90", "laea sph N_POLE"),
    ("+proj=laea +R=6371000 +lat_0=-90 +lon_0=20", "laea sph S_POLE"),
    ("+proj=healpix +ellps=GRS80", "healpix ell"),
    ("+proj=healpix +R=6400000 +rot_xy=42 +lon_0=30", "healpix sph rot_xy"),
    ("+proj=healpix +ellps=WGS84 +rot_xy=-15 +x_0=1000 +y_0=-2000", "healpix ell rot_xy x_0"),
    ("+proj=healpix +a=6378137 +rf=30", "healpix ell |n|>=0.01"),
    ("+proj=rhealpix +ellps=GRS80", "rhealpix ell"),
    ("+proj=rhealpix +south_square=2 +north_square=3 +ellps=WGS84", "rhealpix ell ns=3 ss=2"),
    ("+proj=rhealpix +R=6400000 +north_square=1 +south_square=0 +lon_0=-40", "rhealpix sph ns=1"),
    ("+proj=moll +ellps=GRS80", "moll (ellps -> sphere a)"),
    ("+proj=moll +R=6371000 +lon_0=10 +x_0=500", "moll sph"),
    ("+proj=wag4 +a=6400000", "wag4"),
    ("+proj=wag5 +a=6400000", "wag5"),
]


def special_lonlat(lon0):
    lats = [0.0, 45.0, -45.0, 41.8103148958, -41.8103148958, 41.8103149, -41.8103149]
    for k in range(1, 13):
        lats += [90 - 10.0 ** -k, -90 + 10.0 ** -k]
    lats += [90.0, -90.0]
    lons = [-180.0, -179.999999, -135.0, -90.0, -45.0, -1e-9, 0.0, 1e-9, 45.0, 90.0, 135.0,
            179.999999, 180.0, lon0 + 180 - 1e-9, lon0 - 180 + 1e-9, lon0]
    lo, la = np.meshgrid(lons, lats)
    lo = lo.ravel()
    return np.where(lo > 180, lo - 360, np.where(lo < -180, lo + 360, lo)), la.ravel()


def _fwd(P, lon, lat, dtype):
    inp = torch.tensor(np.radians(np.stack([lon, lat], -1)), dtype=torch.float64).to(dtype)
    with torch.no_grad():
        return P.forward(inp).double().numpy()


def _inv(P, x, y, dtype):
    inp = torch.tensor(np.stack([x, y], -1), dtype=torch.float64).to(dtype)
    with torch.no_grad():
        return P.inverse(inp).double().numpy()


def compare_one(defn, label, rng, n):
    P = M.make(defn)
    lon0 = float(P.params.get("lon_0", 0.0))
    slon, slat = special_lonlat(lon0)
    lon = np.concatenate([rng.uniform(-180, 180, n), slon])
    lat = np.concatenate([rng.uniform(-90, 90, n), slat])
    israndom = np.arange(len(lon)) < n
    X, Y = testing.pyproj_ref(defn, lon, lat)
    pf = np.isfinite(X) & np.isfinite(Y)
    r = dict(label=label, definition=defn, n_fwd=len(lon), n_random=n)
    # ---- forward
    for tag, dt in (("64", torch.float64), ("32", torch.float32)):
        out = _fwd(P, lon, lat, dt)
        tf = np.isfinite(out[:, 0]) & np.isfinite(out[:, 1])
        both = pf & tf
        e = np.full(len(lon), np.nan)
        e[both] = np.hypot(out[both, 0] - X[both], out[both, 1] - Y[both])
        er = e[both & israndom]   # fp32 statistics on the random points only
        r["fwd" + tag] = float(np.nanmax(e)) if both.any() else 0.0
        r["fwd" + tag + "_random"] = float(er.max()) if er.size else 0.0
        r["fwd" + tag + "_p999"] = float(np.percentile(er, 99.9)) if er.size else 0.0
        r["fwd" + tag + "_p50"] = float(np.percentile(er, 50)) if er.size else 0.0
        r["fwd" + tag + "_mask"] = int((pf != tf).sum())
        if both.any():
            i = int(np.nanargmax(e))
            r["fwd" + tag + "_worst"] = (float(lon[i]), float(lat[i]))
    # ---- inverse: in-domain points (PROJ forward of fresh random lon/lat) + a box
    lon2 = rng.uniform(-180, 180, n)
    lat2 = rng.uniform(-90, 90, n)
    X2, Y2 = testing.pyproj_ref(defn, np.concatenate([lon2, slon]), np.concatenate([lat2, slat]))
    ok2 = np.isfinite(X2) & np.isfinite(Y2)
    X2, Y2 = X2[ok2], Y2[ok2]
    nb = n // 4
    xmin, xmax, ymin, ymax = X2.min(), X2.max(), Y2.min(), Y2.max()
    wx, wy = xmax - xmin, ymax - ymin
    Xb = rng.uniform(xmin - 0.1 * wx, xmax + 0.1 * wx, nb)
    Yb = rng.uniform(ymin - 0.1 * wy, ymax + 0.1 * wy, nb)
    Xi = np.concatenate([X2, Xb])
    Yi = np.concatenate([Y2, Yb])
    israndom = np.ones(len(Xi), bool)          # X2 = [finite random part, finite specials]
    nrand_ok = int(ok2[:len(lon2)].sum())
    israndom[nrand_ok:len(X2)] = False
    L, B = testing.pyproj_ref(defn, Xi, Yi, inverse=True)
    pf = np.isfinite(L) & np.isfinite(B)
    # ill-conditioned zone: near the poles, or near the antipode of an azimuthal centre
    sing = np.zeros(len(Xi), bool)
    sing[pf] = np.abs(B[pf]) > 90 - POLE_ZONE_DEG
    if P.params.get("proj") == "laea":
        lat0 = math.radians(float(P.params.get("lat_0", 0.0)))
        lonA = math.radians(lon0 + 180.0)
        cosd = (np.sin(np.radians(B[pf])) * math.sin(-lat0)
                + np.cos(np.radians(B[pf])) * math.cos(-lat0) * np.cos(np.radians(L[pf]) - lonA))
        sing[np.nonzero(pf)[0]] |= np.degrees(np.arccos(np.clip(cosd, -1, 1))) < POLE_ZONE_DEG
    r["n_inv"] = len(Xi)
    for tag, dt in (("64", torch.float64), ("32", torch.float32)):
        out = _inv(P, Xi, Yi, dt)
        tf = np.isfinite(out[:, 0]) & np.isfinite(out[:, 1])
        both = pf & tf
        e = np.full(len(Xi), np.nan)
        e[both] = testing.angular_error_m(np.degrees(out[both, 0]), np.degrees(out[both, 1]),
                                          L[both], B[both])
        er = e[both & israndom]
        r["inv" + tag] = float(np.nanmax(e)) if both.any() else 0.0
        r["inv" + tag + "_regular"] = float(np.nanmax(e[both & ~sing])) if (both & ~sing).any() else 0.0
        r["inv" + tag + "_singular"] = float(np.nanmax(e[both & sing])) if (both & sing).any() else 0.0
        r["inv" + tag + "_random"] = float(er.max()) if er.size else 0.0
        r["inv" + tag + "_p999"] = float(np.percentile(er, 99.9)) if er.size else 0.0
        r["inv" + tag + "_p50"] = float(np.percentile(er, 50)) if er.size else 0.0
        r["inv" + tag + "_mask"] = int((pf != tf).sum())
        if both.any():
            i = int(np.nanargmax(e))
            r["inv" + tag + "_worst"] = (float(Xi[i]), float(Yi[i]), float(L[i]), float(B[i]))
        if tag == "64":
            r["inv_domain_fail_frac"] = float((~pf).mean())
            mis = np.nonzero(pf != tf)[0][:3]
            r["inv64_mask_examples"] = [(float(Xi[j]), float(Yi[j]), bool(pf[j])) for j in mis]
    return r


def run_pyproj():
    print("=" * 78)
    print(f"2. pyproj (PROJ 9.8.1) comparison: {N_RANDOM} random points + special points per definition")
    rng = np.random.default_rng(20261009)
    rows = []
    print("  fp64 columns: max over random+special points; inv64 split into regular / pole-or-antipode zone")
    print("  fp32 columns: inputs cast to float32, error vs the float64 PROJ result, random points only")
    hdr = (f"  {'definition':<44} {'fwd64 m':>8} {'inv64 m':>8} {'inv64sng':>8} {'mask64':>6} | {'fwd32 m':>8}"
           f" {'p99.9':>8} {'p50':>7} {'inv32 m':>8} {'p99.9':>8} {'p50':>7} {'mask32':>6}")
    print(hdr)
    for defn, label in DEFS:
        r = compare_one(defn, label, rng, N_RANDOM)
        rows.append(r)
        print(f"  {label:<44} {r['fwd64']:8.1e} {r['inv64_regular']:8.1e} {r['inv64_singular']:8.1e}"
              f" {r['fwd64_mask'] + r['inv64_mask']:6d} | {r['fwd32_random']:8.1e} {r['fwd32_p999']:8.1e}"
              f" {r['fwd32_p50']:7.1e} {r['inv32_random']:8.1e} {r['inv32_p999']:8.1e} {r['inv32_p50']:7.1e}"
              f" {r['fwd32_mask'] + r['inv32_mask']:6d}")
        check(r["fwd64"] <= FP64_TOL_M, f"{label}: fwd fp64 max err {r['fwd64']:.3e} m (worst at {r.get('fwd64_worst')})")
        check(r["inv64_regular"] <= FP64_TOL_M,
              f"{label}: inv fp64 max err {r['inv64_regular']:.3e} m (worst at {r.get('inv64_worst')})")
        check(r["inv64_singular"] <= FP64_TOL_SINGULAR_M,
              f"{label}: inv fp64 max err near poles {r['inv64_singular']:.3e} m")
        check(r["fwd64_mask"] == 0, f"{label}: fwd fp64 failure-mask mismatch {r['fwd64_mask']}")
        check(r["inv64_mask"] == 0, f"{label}: inv fp64 failure-mask mismatch {r['inv64_mask']} "
                                    f"e.g. {r['inv64_mask_examples']}")
    RESULTS["pyproj"] = rows
    run_quirks()


def run_quirks():
    """PROJ behaviours that are arguably bugs but are reproduced on purpose."""
    print("  quirk: rHEALPix pnpoly never closes its polygon (healpix.cpp:148: i < nvert, no last->first")
    print("         edge), so points LEFT of the image (x < -pi a, |y| <= pi/4 a) pass the in-image test:")
    xs = np.array([-3.3, -3.2, -4.0, -10.0, 3.3, -3.3, -3.3])
    ys = np.array([0.0, 0.5, 0.7, 0.1, 0.0, 0.8, -0.79])
    for d in ("+proj=rhealpix +R=1", "+proj=rhealpix +R=1 +north_square=2 +south_square=1", "+proj=healpix +R=1"):
        L, B = testing.pyproj_ref(d, xs, ys, inverse=True)
        t = _inv(M.make(d), xs, ys, torch.float64)
        same_mask = np.array_equal(np.isfinite(L), np.isfinite(t[:, 0]))
        fin = np.isfinite(L)
        e = testing.angular_error_m(np.degrees(t[fin, 0]), np.degrees(t[fin, 1]), L[fin], B[fin], a=1.0)
        ok = same_mask and (e.size == 0 or e.max() < 1e-12)
        print(f"    [{'ok' if ok else 'FAIL'}] {d:<52} PROJ accepts {int(fin.sum())}/{len(xs)} "
              f"(x={xs[fin].tolist()}), port agrees")
        check(ok, f"quirk reproduction {d}")


# =============================================================================
# 3. Mollweide Newton solver: masked vs fixed-N vs implicit gradient
# =============================================================================
class _SavedBytes:
    """Count bytes saved for backward (saved_tensors_hooks) = autograd memory."""

    def __init__(self):
        self.n = 0

    def pack(self, t):
        self.n += t.numel() * t.element_size()
        return t

    @staticmethod
    def unpack(t):
        return t


def moll_iterations(phi, tol=M.MOLL_LOOP_TOL, n_iter=M.MOLL_MAX_ITER):
    """How many loop bodies PROJ executes per point (1..30, 31 = no convergence)."""
    k = math.pi * torch.sin(phi)
    th = phi.clone()
    active = torch.ones_like(phi, dtype=torch.bool)
    count = torch.zeros_like(phi, dtype=torch.int64)
    for _ in range(n_iter):
        V = (th + torch.sin(th) - k) / (1 + torch.cos(th))
        th = torch.where(active, th - V, th)
        count = count + active.long()
        active = active & ~(V.abs() < tol)
    return torch.where(active, n_iter + 1, count)


def moll_truth(lon_deg, lat_deg, C_x, C_y, R):
    """Cancellation-free Mollweide forward for the float64 radians PROJ receives:
    with theta2 = pi - eps (sign of phi aside), theta2 + sin(theta2) = pi sin(phi)
    <=> eps - sin(eps) = pi (1 - sin|phi|) = 2 pi sin^2(delta/2), delta = pi/2 - |phi|.
    eps - sin(eps) uses its Taylor series for small eps, 1 - cos(eps) = 2 sin^2(eps/2)."""
    lam = np.radians(lon_deg)
    phi = np.radians(lat_deg)
    delta = (math.pi / 2 - np.abs(phi)) + 6.123233995736766e-17  # pi/2 - fl(pi/2)
    rhs = 2 * math.pi * np.sin(delta / 2) ** 2
    eps = np.minimum(np.cbrt(6 * rhs), math.pi)

    def g(e):
        e2 = e * e
        ser = e * e2 / 6 * (1 - e2 / 20 * (1 - e2 / 42 * (1 - e2 / 72 * (1 - e2 / 110 * (1 - e2 / 156 * (1 - e2 / 210))))))
        return np.where(e < 0.5, ser, e - np.sin(e))

    for _ in range(100):
        eps = eps - (g(eps) - rhs) / np.maximum(2 * np.sin(eps / 2) ** 2, 1e-300)
    x = R * C_x * lam * np.sin(eps / 2)
    y = R * C_y * np.sign(phi) * np.cos(eps / 2)
    return x, y


def run_moll_study():
    print("=" * 78)
    print("3. Mollweide forward Newton iteration: masked (faithful) / fixed-N / implicit gradient")
    defn = "+proj=moll +R=6371000"
    rng = np.random.default_rng(7)
    n = N_RANDOM
    lon = rng.uniform(-180, 180, n)
    lat = rng.uniform(-90, 90, n)
    X, Y = testing.pyproj_ref(defn, lon, lat)
    lp = t64(np.radians(np.stack([lon, lat], -1)))
    res = {}
    # 3a. iteration count needed by PROJ's loop
    it = moll_iterations(lp[:, 1])
    hist = np.bincount(it.numpy(), minlength=32)
    res["iter_hist"] = {int(k): int(v) for k, v in enumerate(hist) if v}
    print(f"  loop bodies executed by PROJ's masked loop (1e5 uniform-lat points): "
          f"max={int(it.max())}, mean={float(it.double().mean()):.2f}, histogram={res['iter_hist']}")
    polar = t64(np.radians([90 - 10.0 ** -k for k in range(1, 13)]))
    print(f"  ... near the pole (lat = 90-1e-k, k=1..12): {moll_iterations(polar).tolist()}")
    # 3b. fixed-N accuracy vs PROJ
    # truth (cancellation-free) on the random points + a near-pole sweep
    lat_p = np.concatenate([90 - 10.0 ** -np.linspace(1, 12, 45), -90 + 10.0 ** -np.linspace(1, 12, 45)])
    lon_p = np.linspace(-179.5, 179.5, lat_p.size)
    lonA, latA = np.concatenate([lon, lon_p]), np.concatenate([lat, lat_p])
    XA, YA = testing.pyproj_ref(defn, lonA, latA)
    Pm0 = M.make(defn)
    XT, YT = moll_truth(lonA, latA, Pm0.C_x, Pm0.C_y, Pm0.a)
    lpA = t64(np.radians(np.stack([lonA, latA], -1)))
    nr = len(lon)
    eP = np.hypot(XA - XT, YA - YT)
    print(f"  PROJ vs cancellation-free truth: max err random pts {eP[:nr].max():.2e} m,"
          f" near-pole sweep {eP[nr:].max():.2e} m")
    res["proj_vs_truth"] = (float(eP[:nr].max()), float(eP[nr:].max()))
    print("  fixed-N (no freezing, PROJ's convergence test on the last step) vs PROJ and vs truth, fp64:")
    res["fixedN"] = {}
    for N in (3, 4, 5, 6, 8, 10, 12, 15, 20, 30):
        P = M.make(defn, method="fixed", n_iter=N)
        with torch.no_grad():
            out = P.forward(lpA).numpy()
        e = np.hypot(out[:, 0] - XA, out[:, 1] - YA)
        et = np.hypot(out[:, 0] - XT, out[:, 1] - YT)
        res["fixedN"][N] = (float(e[:nr].max()), float(e[nr:].max()), float(et[:nr].max()), float(et[nr:].max()))
        print(f"    N={N:2d}: vs PROJ max {e[:nr].max():.2e} m (pole sweep {e[nr:].max():.2e})"
              f" | vs truth {et[:nr].max():.2e} m (pole sweep {et[nr:].max():.2e})"
              f" | random points hitting the +-pi/2 fallback: {int((e[:nr] > 1).sum())}")
    print("  fixed-N without the +-pi/2 fallback (loop_tol=inf: plain N-th Newton iterate), fp64:")
    res["fixedN_nofallback"] = {}
    for N in (4, 5, 6, 8, 10, 12, 15, 20, 25, 30):
        P = M.make(defn, method="fixed", n_iter=N, loop_tol=float("inf"))
        with torch.no_grad():
            out = P.forward(lpA).numpy()
        e = np.hypot(out[:, 0] - XA, out[:, 1] - YA)
        res["fixedN_nofallback"][N] = (float(e[:nr].max()), float(np.percentile(e[:nr], 99.9)), float(e[nr:].max()))
        print(f"    N={N:2d}: vs PROJ max {e[:nr].max():.2e} m, p99.9 {np.percentile(e[:nr], 99.9):.2e} m,"
              f" pole sweep max {e[nr:].max():.2e} m")
    Pmask = M.make(defn)
    with torch.no_grad():
        outm = Pmask.forward(lpA).numpy()
    em = np.hypot(outm[:, 0] - XA, outm[:, 1] - YA)
    emt = np.hypot(outm[:, 0] - XT, outm[:, 1] - YT)
    print(f"    masked (faithful): vs PROJ max {em[:nr].max():.2e} m (pole sweep {em[nr:].max():.2e})"
          f" | vs truth {emt[:nr].max():.2e} m (pole sweep {emt[nr:].max():.2e})")
    res["masked_vs_proj"] = (float(em[:nr].max()), float(em[nr:].max()))
    check(em.max() < FP64_TOL_M, "moll masked differs from PROJ (incl. pole sweep)")
    # fixed-30 keeps iterating after PROJ's |V| < 1e-7 stop. Near the poles the
    # residual theta2 + sin(theta2) - pi sin(phi) is pure rounding noise (triple root),
    # so extra steps just move within the noise band (~1e-4 m at |lat| ~ 89.99 deg).
    check(res["fixedN"][30][0] < 1e-3 and res["fixedN"][30][2] < 1e-3,
          "moll fixed-N=30: unexpected deviation from PROJ / truth")
    # 3c. gradients: masked-unrolled, fixed-unrolled, implicit vs analytic (IFT at converged root)
    m = np.abs(lat) < 89.0
    lpi = lp[m][:20000].clone()
    jac = {}
    saved = {}
    for meth in ("masked", "fixed", "implicit"):
        P = M.make(defn, method=meth)
        x = lpi.clone().requires_grad_(True)
        sb = _SavedBytes()
        with torch.autograd.graph.saved_tensors_hooks(sb.pack, sb.unpack):
            out = P.forward(x)
        saved[meth] = sb.n / x.shape[0]
        gx, = torch.autograd.grad(out[:, 0].sum(), x, retain_graph=True)
        gy, = torch.autograd.grad(out[:, 1].sum(), x)
        jac[meth] = (gx[:, 1].numpy(), gy[:, 1].numpy(), gx[:, 0].numpy())
    # analytic: theta2 root (from the converged iterate), dtheta2/dphi = C_p cos(phi)/(1+cos theta2)
    Pm = M.make(defn)
    with torch.no_grad():
        th2, conv = M.moll_newton_masked(lpi[:, 1], Pm.C_p * torch.sin(lpi[:, 1]), 60, 0.0)
        for _ in range(3):  # polish
            th2 = th2 - (th2 + torch.sin(th2) - Pm.C_p * torch.sin(lpi[:, 1])) / (1 + torch.cos(th2))
    dth = Pm.C_p * torch.cos(lpi[:, 1]) / (1 + torch.cos(th2))
    th = th2 / 2
    dxdphi = (-Pm.a * Pm.C_x * lpi[:, 0] * torch.sin(th) * dth / 2).numpy()
    dydphi = (Pm.a * Pm.C_y * torch.cos(th) * dth / 2).numpy()
    res["grad_rel_err"] = {}
    for meth in jac:
        rx = np.abs(jac[meth][0] - dxdphi) / np.maximum(np.abs(dxdphi), 1.0)
        ry = np.abs(jac[meth][1] - dydphi) / np.maximum(np.abs(dydphi), 1.0)
        res["grad_rel_err"][meth] = float(max(rx.max(), ry.max()))
        print(f"  d(x,y)/dphi [{meth:8s}] max rel. err vs analytic IFT derivative: "
              f"{res['grad_rel_err'][meth]:.2e}; autograd saved tensors: {saved[meth]:.0f} bytes/point")
    res["saved_bytes_per_point"] = saved
    check(res["grad_rel_err"]["implicit"] < 1e-9, "moll implicit gradient inaccurate")
    check(res["grad_rel_err"]["masked"] < 1e-5, "moll masked unrolled gradient inaccurate")
    d_mi = max(np.abs(jac["masked"][k] - jac["implicit"][k]).max() / np.abs(jac["implicit"][k]).max()
               for k in range(2))
    print(f"  masked-unrolled vs implicit: max |diff| / max|J| = {d_mi:.2e}")
    res["masked_vs_implicit"] = float(d_mi)
    # 3d. near-pole gradients
    print("  near-pole d(y)/d(phi) [m/rad] (true value -> 0 like delta^(1/3); dx/dphi -> inf like delta^(-1/3)):")
    rows = []
    for k in (2, 4, 6, 8, 10, 12, None):
        latd = 90.0 if k is None else 90 - 10.0 ** -k
        r = {"lat": latd}
        for meth in ("masked", "fixed", "implicit"):
            P = M.make(defn, method=meth)
            x = t64([[math.radians(170.0), math.radians(latd)]]).requires_grad_(True)
            out = P.forward(x)
            g, = torch.autograd.grad(out[0, 1], x)
            g2, = torch.autograd.grad(P.forward(x)[0, 0], x)
            r[meth] = (float(g[0, 1]), float(g2[0, 1]))
        rows.append(r)
        print(f"    lat={latd!s:<22} dy/dphi masked {r['masked'][0]:10.3e} fixed {r['fixed'][0]:10.3e} "
              f"implicit {r['implicit'][0]:10.3e} | dx/dphi masked {r['masked'][1]:10.3e} implicit {r['implicit'][1]:10.3e}")
    res["near_pole"] = rows
    # 3e. float32: PROJ's LOOP_TOL = 1e-7 is below float32 resolution
    for meth, N, tol in (("masked", 30, M.MOLL_LOOP_TOL), ("masked", 30, 1e-6), ("masked", 30, 1e-5),
                         ("fixed", 20, float("inf")), ("fixed", 30, float("inf"))):
        P = M.make(defn, method=meth, n_iter=N, loop_tol=tol)
        with torch.no_grad():
            out = P.forward(lp.float()).double().numpy()
        e = np.hypot(out[:, 0] - X, out[:, 1] - Y)
        frac = float((e > 1000).mean())
        print(f"  float32 {meth} N={N} loop_tol={tol:g}: max err {e.max():.3e} m, p99.9 "
              f"{np.percentile(e, 99.9):.2e} m, fraction with err > 1 km (+-pi/2 fallback): {frac:.4f}")
        res[f"fp32_{meth}_{N}_{tol:g}"] = (float(e.max()), float(np.percentile(e, 99.9)), frac)
    # 3f. PROJ's own accuracy near the pole (cancellation in theta + sin(theta) - pi sin(phi))
    lat_k = np.array([90 - 10.0 ** -k for k in range(1, 13)])
    Xp, Yp = testing.pyproj_ref(defn, np.full_like(lat_k, 170.0), lat_k)
    xt, yt = moll_truth(np.full_like(lat_k, 170.0), lat_k, Pm.C_x, Pm.C_y, Pm.a)
    errs = np.hypot(Xp - xt, Yp - yt)
    print("  PROJ error vs truth at lon=170, lat=90-1e-k, k=1..12 [m]: " + " ".join(f"{v:.2g}" for v in errs))
    res["proj_pole_error_m"] = errs.tolist()
    RESULTS["moll"] = res


# =============================================================================
# 4. gradcheck at interior points, NaN-leak check, singular points
# =============================================================================
GRAD_DEFS = [
    ("+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80", (-40, 60, -30, 85)),
    ("+proj=laea +ellps=GRS80", (-80, 80, -80, 80)),
    ("+proj=laea +lat_0=90 +lon_0=-150 +ellps=WGS84", (-180, 180, -60, 88)),
    ("+proj=laea +lat_0=-90 +ellps=GRS80", (-180, 180, -88, 60)),
    ("+proj=laea +a=6378137 +rf=30 +lat_0=40", (-60, 60, -40, 85)),
    ("+proj=laea +R=6371000 +lat_0=45 +lon_0=10", (-60, 60, -40, 85)),
    ("+proj=laea +R=6371000", (-80, 80, -80, 80)),
    ("+proj=laea +R=6371000 +lat_0=90", (-180, 180, -60, 88)),
    ("+proj=laea +R=6371000 +lat_0=-90", (-180, 180, -88, 60)),
    ("+proj=healpix +ellps=GRS80", (-179, 179, -89, 89)),
    ("+proj=healpix +R=6400000 +rot_xy=42 +lon_0=30", (-179, 179, -89, 89)),
    ("+proj=healpix +a=6378137 +rf=30", (-179, 179, -89, 89)),
    ("+proj=rhealpix +ellps=GRS80", (-179, 179, -89, 89)),
    ("+proj=rhealpix +south_square=2 +north_square=3 +ellps=WGS84", (-179, 179, -89, 89)),
    ("+proj=moll +R=6371000", (-179, 179, -88, 88)),
    ("+proj=wag4 +a=6400000", (-179, 179, -88, 88)),
    ("+proj=wag5 +a=6400000", (-179, 179, -88, 88)),
]


def run_gradcheck():
    print("=" * 78)
    print("4. torch.autograd.gradcheck (float64, interior points; outputs/inputs scaled by a)")
    rng = np.random.default_rng(11)
    st = dict(total=0, passed=0)
    for defn, (lo0, lo1, la0, la1) in GRAD_DEFS:
        variants = [("", {})]
        if "moll" in defn:
            variants = [(" [masked]", dict(method="masked")), (" [fixed]", dict(method="fixed")),
                        (" [implicit]", dict(method="implicit"))]
        for vtag, kw in variants:
            P = M.make(defn, **kw)
            lon = rng.uniform(lo0, lo1, 12)
            lat = rng.uniform(la0, la1, 12)
            lp = t64(np.radians(np.stack([lon, lat], -1)))
            a = P.a
            fwd = lambda t: P.forward(t) / a  # noqa: E731
            with torch.no_grad():
                xy = P.forward(lp)
            okm = torch.isfinite(xy).all(-1)
            lp = lp[okm]
            xy = xy[okm]
            u = ((xy - torch.tensor([P.x0, P.y0], dtype=torch.float64)) / a)
            inv = lambda t: P.inverse(t * a + torch.tensor([P.x0, P.y0], dtype=torch.float64))  # noqa: E731
            for name, fn, inp in (("fwd", fwd, lp), ("inv", inv, u)):
                st["total"] += 1
                try:
                    ok = torch.autograd.gradcheck(fn, (inp.clone().requires_grad_(True),),
                                                  eps=1e-6, atol=1e-6, rtol=1e-4)
                except Exception as err:  # gradcheck raises on mismatch
                    ok = False
                    print(f"     {str(err)[:400]}")
                st["passed"] += bool(ok)
                print(f"  [{'ok' if ok else 'FAIL'}]  {name}  {defn}{vtag}")
                check(ok, f"gradcheck {name} {defn}{vtag}")
    print(f"  gradcheck: {st['passed']}/{st['total']} passed")
    RESULTS["gradcheck"] = st


def run_nan_leak():
    print("=" * 78)
    print("5. NaN/inf in gradients: batches mixing valid, failing (NaN output) and singular points.")
    print("   failed points must get a finite (zero) gradient; non-finite gradients at valid outputs are")
    print("   listed: they are genuine singularities (inf derivative), never leaks between points")
    rng = np.random.default_rng(5)
    for defn, _ in DEFS:
        P = M.make(defn)
        lon = np.concatenate([rng.uniform(-180, 180, 200), [0, 180, -180, 0, 0, 45, 179.9999]])
        lat = np.concatenate([rng.uniform(-90, 90, 200), [90, -90, 0, -90, 0, 41.8103149, 89.99999999]])
        x = t64(np.radians(np.stack([lon, lat], -1))).requires_grad_(True)
        out = P.forward(x)
        g, = torch.autograd.grad(torch.where(torch.isfinite(out), out, 0.0).sum(), x)
        with torch.no_grad():
            xy = P.forward(t64(np.radians(np.stack([lon, lat], -1))))
        fin = torch.isfinite(xy).all(-1)
        span = xy[fin].abs().max(0).values
        box = (torch.rand(300, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(1)) * 2.4 - 1.2) * span
        xin = torch.cat([xy[fin], box, torch.tensor([[P.x0, P.y0]], dtype=torch.float64)]).requires_grad_(True)
        lpo = P.inverse(xin)
        g2, = torch.autograd.grad(torch.where(torch.isfinite(lpo), lpo, 0.0).sum(), xin)
        res = {}
        for tag, o, gg, inp in (("fwd", out, g, x), ("inv", lpo, g2, xin)):
            failed = ~torch.isfinite(o[:, :2]).all(-1)
            badg = ~torch.isfinite(gg).all(-1)
            leak = int((failed & badg).sum())
            sing = torch.nonzero(~failed & badg).flatten()
            res[tag] = (int(failed.sum()), leak, int(sing.numel()))
            check(leak == 0, f"{defn}: {tag}: {leak} failed points with non-finite gradient")
            if sing.numel():
                pts = inp[sing[:3]].detach()
                if tag == "fwd":
                    pts = torch.rad2deg(pts)
                print(f"        {tag}: singular (finite value, non-finite grad) at "
                      f"{[tuple(round(v, 6) for v in p.tolist()) for p in pts]} -> value "
                      f"{[tuple(round(v, 6) for v in q.tolist()) for q in o[sing[:3], :2].detach()]}")
        print(f"  [{'ok' if res['fwd'][1] == 0 and res['inv'][1] == 0 else 'FAIL'}]  {defn:<66} "
              f"failed fwd/inv={res['fwd'][0]}/{res['inv'][0]}  NaN-grad on failed={res['fwd'][1]}/{res['inv'][1]}"
              f"  singular={res['fwd'][2]}/{res['inv'][2]}")
        RESULTS.setdefault("nan_leak", []).append((defn, res))


def jac_at(P, lonlat_deg=None, xy=None):
    if lonlat_deg is not None:
        x = t64([np.radians(lonlat_deg)]).requires_grad_(True)
        fn = P.forward
    else:
        x = t64([xy]).requires_grad_(True)
        fn = P.inverse
    out = fn(x)
    rows = []
    for i in range(2):
        g, = torch.autograd.grad(out[0, i], x, retain_graph=True)
        rows.append(g[0].tolist())
    return out[0, :2].detach().tolist(), rows


def run_singular():
    print("=" * 78)
    print("6. Gradients at singular points (value, Jacobian d(out)/d(in); fwd: m/rad, inv: rad/m)")
    cases = []
    L = M.make("+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80")
    cases += [("laea 3035 fwd centre (10,52)", L, (10, 52), None),
              ("laea 3035 fwd antipode (-170,-52) [domain error]", L, (-170, -52), None),
              ("laea 3035 fwd near antipode (-170,-51.99)", L, (-170, -51.99), None),
              ("laea 3035 inv centre (x0,y0) [rho<EPS10 branch]", L, None, (4321000.0, 3210000.0)),
              ("laea 3035 inv 1 mm from centre", L, None, (4321000.001, 3210000.0)),
              ("laea 3035 inv outside (|.|>2R) [domain error]", L, None, (4321000.0 + 1.3e7, 3210000.0))]
    LN = M.make("+proj=laea +lat_0=90 +ellps=GRS80")
    cases += [("laea N_POLE fwd at pole (0,90) [q<1e-15 branch]", LN, (30, 90), None),
              ("laea N_POLE fwd 1e-7 deg from pole", LN, (30, 90 - 1e-7), None),
              ("laea N_POLE fwd opposite pole (0,-90) [domain error]", LN, (30, -90), None),
              ("laea N_POLE inv centre (0,0) [q==0 branch]", LN, None, (0.0, 0.0)),
              ("laea N_POLE inv 1 cm from centre", LN, None, (0.01, 0.0)),
              ("laea N_POLE inv 1 m from centre", LN, None, (1.0, 0.0))]
    LS = M.make("+proj=laea +R=6371000")
    cases += [("laea sph EQUIT inv at image of N pole", LS, None, (0.0, math.sqrt(2) * 6371000)),
              ("laea sph EQUIT fwd antipode (180,0) [domain error]", LS, (180, 0), None),
              ("laea sph EQUIT fwd near antipode (180,10)", LS, (180, 10), None)]
    H = M.make("+proj=healpix +ellps=GRS80")
    cases += [("healpix fwd pole (20,90) [sigma=0]", H, (20, 90), None),
              ("healpix fwd 1e-6 deg from pole", H, (20, 90 - 1e-6), None),
              ("healpix fwd eq/polar boundary lat=asin(2/3)", H, (20, math.degrees(math.asin(2 / 3))), None),
              ("healpix fwd cap boundary lon=0 lat=60 (x jumps)", H, (0.0, 60), None),
              ("healpix fwd antimeridian lon=180 lat=10", H, (180, 10), None),
              ("healpix inv cap tip (-3pi/4 a, pi/2 a)", H, None, (-3 * math.pi / 4 * H.a, math.pi / 2 * H.a)),
              ("healpix inv 1 m below cap tip", H, None, (-3 * math.pi / 4 * H.a, math.pi / 2 * H.a - 1)),
              ("healpix inv outside image [error]", H, None, (0.0, 0.8 * math.pi * H.a))]
    for meth in ("masked", "fixed", "implicit"):
        Mo = M.make("+proj=moll +R=6371000", method=meth)
        cases += [(f"moll[{meth}] fwd pole (170,90)", Mo, (170, 90), None),
                  (f"moll[{meth}] fwd antimeridian (180,10)", Mo, (180, 10), None)]
    Mo = M.make("+proj=moll +R=6371000")
    cases += [("moll inv pole y = C_y R (1+1e-15) [aasin |v|>=1 branch]", Mo, None,
               (0.0, Mo.C_y * 6371000 * (1 + 1e-15))),
              ("moll inv y = fl(C_y R) (1 ulp below the pole)", Mo, None, (0.0, Mo.C_y * 6371000)),
              ("moll inv 1 m below pole", Mo, None, (0.0, Mo.C_y * 6371000 - 1.0)),
              ("moll inv outside ellipse (lam > pi) [error]", Mo, None, (3.2 * Mo.C_x * 6371000, 0.0))]
    rows = []
    for label, P, ll, xy in cases:
        val, J = jac_at(P, ll, xy)
        fin = all(math.isfinite(v) for r in J for v in r)
        print(f"  {label:<55} val=({val[0]:.6g}, {val[1]:.6g})  J=[[{J[0][0]:.3g}, {J[0][1]:.3g}],"
              f" [{J[1][0]:.3g}, {J[1][1]:.3g}]]{'' if fin else '  <non-finite>'}")
        rows.append((label, val, J))
    RESULTS["singular"] = rows


# =============================================================================
# 7. torch.compile
# =============================================================================
def run_compile():
    print("=" * 78)
    print("7. torch.compile(fullgraph=True) on small inputs (forward+backward)")
    cfg = [("+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80", {}),
           ("+proj=laea +R=6371000 +lat_0=90", {}),
           ("+proj=laea +a=6378137 +rf=30 +lat_0=40", {}),
           ("+proj=healpix +ellps=GRS80 +rot_xy=10", {}),
           ("+proj=rhealpix +south_square=2 +north_square=3 +ellps=WGS84", {}),
           ("+proj=moll +R=6371000", dict(method="masked")),
           ("+proj=moll +R=6371000", dict(method="fixed", n_iter=8)),
           ("+proj=moll +R=6371000", dict(method="implicit"))]
    backends = ["aot_eager"] + (["inductor"] if DO_INDUCTOR else [])
    lp = t64(np.radians([[10.0, 50.0], [-120.0, 70.0], [170.0, -89.5], [35.0, 10.0], [-60, -45]]))
    st = dict(total=0, passed=0, times={})
    for be in backends:
        for defn, kw in cfg:
            if be == "inductor" and defn not in (cfg[0][0], cfg[3][0], cfg[7][0]):
                continue
            P = M.make(defn, **kw)
            with torch.no_grad():
                xy = P.forward(lp)
            for name, fn, inp in (("fwd", P.forward, lp), ("inv", P.inverse, xy)):
                st["total"] += 1
                torch._dynamo.reset()
                t0 = time.time()
                try:
                    cf = torch.compile(fn, fullgraph=True, backend=be)
                    x1 = inp.clone().requires_grad_(True)
                    y1 = cf(x1)
                    g1, = torch.autograd.grad(torch.nan_to_num(y1).sum(), x1)
                    x0 = inp.clone().requires_grad_(True)
                    y0 = fn(x0)
                    g0, = torch.autograd.grad(torch.nan_to_num(y0).sum(), x0)
                    dv = float((torch.nan_to_num(y1) - torch.nan_to_num(y0)).abs().max().detach())
                    dg = float((g1 - g0).abs().max())
                    ok = dv <= 1e-6 and dg <= 1e-6 * max(1.0, float(g0.abs().max()))
                    msg = f"max|diff| value {dv:.1e} grad {dg:.1e}"
                except Exception as err:
                    ok, msg = False, f"{type(err).__name__}: {str(err)[:300]}"
                dt = time.time() - t0
                st["passed"] += ok
                st["times"][f"{be} {name} {defn} {kw}"] = dt
                print(f"  [{'ok' if ok else 'FAIL'}]  {be:<9} {name}  {defn} {kw or ''}  {msg}  ({dt:.1f}s)")
                check(ok, f"torch.compile {be} {name} {defn} {kw}")
    RESULTS["compile"] = st


# =============================================================================
def main():
    t0 = time.time()
    run_gie()
    run_pyproj()
    run_moll_study()
    run_gradcheck()
    run_nan_leak()
    run_singular()
    if DO_COMPILE:
        run_compile()
    print("=" * 78)
    g = RESULTS["gie"]
    print(f"SUMMARY  gie {g['passed']}/{g['total']} (blocks {g['blocks']})  |  "
          f"gradcheck {RESULTS['gradcheck']['passed']}/{RESULTS['gradcheck']['total']}  |  "
          f"compile {RESULTS.get('compile', {}).get('passed', '-')}/{RESULTS.get('compile', {}).get('total', '-')}")
    rows = RESULTS["pyproj"]
    print(f"         pyproj max fp64 err: fwd {max(r['fwd64'] for r in rows):.2e} m, "
          f"inv {max(r['inv64'] for r in rows):.2e} m over {len(rows)} definitions")
    print(f"         {len(FAILURES)} failure(s); {time.time() - t0:.1f}s")
    for f in FAILURES:
        print("   -", f)
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
