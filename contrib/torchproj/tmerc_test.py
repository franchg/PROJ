"""Validation of tmerc.py (tmerc / etmerc / utm) against PROJ.

Run:  python tmerc_test.py [--no-compile] [--n 100000]
Prints a summary and exits non-zero on failure.
"""
import math
import sys
import time

import numpy as np
import torch

import testing
import tmerc

torch.set_num_threads(1)
N_RANDOM = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 100_000
DO_COMPILE = "--no-compile" not in sys.argv
FP64_TOL_M = 1e-6  # pass threshold vs PROJ in float64 (metres)

FAILURES = []


def fail(msg):
    FAILURES.append(msg)
    print("   FAIL:", msg)


def deg2rad_t(a, dtype=torch.float64):
    return torch.tensor(np.radians(np.asarray(a, dtype=float)), dtype=dtype)


# =============================================================================
# 1. gie vectors
# =============================================================================
def setup_failure_expected(block):
    """True if the gie file expects the operation itself to fail
    (an 'expect failure' before any 'accept' in that operation block)."""
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


def run_gie():
    print("=" * 78)
    print("1. gie test vectors (PROJ test/gie/*.gie)")
    stats = dict(blocks=0, total=0, passed=0, skipped=0, failures=[])
    for name in ("tmerc", "utm", "etmerc"):
        for blk in testing.gie_cases(name):
            stats["blocks"] += 1
            op = blk["operation"]
            ncase = len(blk["cases"]) + len(blk["roundtrips"])
            try:
                P = tmerc.make(op)
                err = None
            except (ValueError, NotImplementedError) as e:
                P, err = None, e
            if ncase == 0:
                stats["total"] += 1
                if setup_failure_expected(blk):
                    ok = P is None and isinstance(err, ValueError)
                    tag = "setup failure as expected" if ok else "setup should fail"
                else:
                    ok, tag = P is not None, "setup ok (no cases)"
                stats["passed"] += ok
                if not ok:
                    stats["failures"].append(f"{blk['file']}: {op}: {tag}")
                print(f"  [{'ok' if ok else 'FAIL'}] {blk['file']:<18} {op:<58} {tag}")
                continue
            if P is None:
                stats["total"] += ncase
                stats["failures"].append(f"{op}: unexpected setup error {err}")
                continue
            worst, nok = 0.0, 0
            for c in blk["cases"]:
                stats["total"] += 1
                acc = list(c["accept"]) + [0.0] * (4 - len(c["accept"]))
                if c["direction"] == "fwd":
                    inp = torch.tensor([[math.radians(acc[0]), math.radians(acc[1]), acc[2], acc[3]]],
                                       dtype=torch.float64)
                    out = P(inp)[0].numpy()
                    if c["expect"] == "failure":
                        ok = bool(np.isnan(out[:2]).all())
                        d = 0.0 if ok else float("inf")
                    else:
                        exp = list(c["expect"]) + [acc[2], acc[3]][len(c["expect"]) - 2:]
                        d = float(np.hypot(out[0] - exp[0], out[1] - exp[1]))
                        ok = d <= c["tolerance_m"] and np.allclose(out[2:], exp[2:4])
                else:
                    inp = torch.tensor([acc], dtype=torch.float64)
                    out = P.inverse(inp)[0].numpy()
                    if c["expect"] == "failure":
                        ok = bool(np.isnan(out[:2]).all())
                        d = 0.0 if ok else float("inf")
                    else:
                        exp = c["expect"]
                        d = float(testing.angular_error_m(math.degrees(out[0]), math.degrees(out[1]),
                                                          exp[0], exp[1], a=P.a))
                        ok = d <= c["tolerance_m"]
                worst = max(worst, d / c["tolerance_m"])
                nok += ok
                stats["passed"] += ok
                if not ok:
                    stats["failures"].append(f"{blk['file']}: {op}: {c['direction']} {c['accept']} -> err {d:.3e} m "
                                             f"(tol {c['tolerance_m']:.1e})")
            for r in blk["roundtrips"]:
                stats["total"] += 1
                acc = r["accept"]
                lp0 = torch.tensor([[math.radians(acc[0]), math.radians(acc[1])]], dtype=torch.float64)
                lp = lp0
                xy = P(lp)
                for _ in range(r["n"] - 1):
                    xy = P(P.inverse(xy))
                lp = P.inverse(xy)[0].numpy()
                d = float(testing.angular_error_m(acc[0], acc[1], math.degrees(lp[0]), math.degrees(lp[1]), a=P.a))
                ok = d <= r["tolerance_m"]
                worst = max(worst, d / r["tolerance_m"])
                nok += ok
                stats["passed"] += ok
                if not ok:
                    stats["failures"].append(f"{op}: roundtrip {acc} err {d:.3e} m")
            print(f"  [{'ok' if nok == ncase else 'FAIL'}] {blk['file']:<18} {op:<58} {nok}/{ncase}"
                  f"  worst err/tol={worst:.2e}  ({P.kind})")
    print(f"  gie: blocks={stats['blocks']} cases={stats['total']} passed={stats['passed']} "
          f"skipped={stats['skipped']}")
    for f in stats["failures"]:
        fail("gie " + f)
    return stats


# =============================================================================
# 2. pyproj comparison on random points (fp64 and fp32)
# =============================================================================
PYPROJ_SETS = [
    # (definition, lon range, lat range, note)
    ("+proj=utm +zone=32 +ellps=WGS84", (3, 15), (0, 84), "UTM zone 32N, default exact (poder_engsager)"),
    ("+proj=utm +zone=32 +ellps=WGS84 +approx", (3, 15), (0, 84), "UTM zone 32N, Evenden/Snyder"),
    ("+proj=utm +zone=33 +south +ellps=WGS84", (9, 21), (-80, 0), "UTM zone 33S"),
    ("+proj=tmerc +lon_0=10 +ellps=WGS84", (-30, 50), (-84, 84), "exact, lon_0 +- 40 deg"),
    ("+proj=tmerc +lon_0=10 +ellps=WGS84 +algo=auto", (0, 20), (-84, 84), "auto: per-point exact/approx"),
    ("+proj=tmerc +lat_0=49 +lon_0=-2 +k_0=0.9996012717 +x_0=400000 +y_0=-100000 +ellps=airy",
     (-9, 3), (49, 61), "British National Grid (lat_0, k_0, false origin)"),
    ("+proj=tmerc +R=6371000 +k_0=0.9996", (-170, 170), (-85, 85), "spherical"),
]


def compare_pyproj(defn, lon_r, lat_r, note, n, seed):
    rng = np.random.default_rng(seed)
    lon = rng.uniform(*lon_r, n)
    lat = rng.uniform(*lat_r, n)
    P = tmerc.make(defn)
    X, Y = testing.pyproj_ref(defn, lon, lat)
    fin = np.isfinite(X) & np.isfinite(Y)
    L, B = testing.pyproj_ref(defn, X[fin], Y[fin], inverse=True)
    res = {"definition": defn, "n_points": n, "kind": P.kind, "note": note}
    for dt, tag in ((torch.float64, "fp64"), (torch.float32, "fp32")):
        inp = torch.tensor(np.radians(np.stack([lon, lat], -1))).to(dt)
        out = P(inp)
        assert out.dtype == dt
        out = out.double().numpy()
        mism = int((np.isfinite(out[:, 0]) != fin).sum())
        ef = np.hypot(out[fin, 0] - X[fin], out[fin, 1] - Y[fin])
        xy = torch.tensor(np.stack([X[fin], Y[fin]], -1)).to(dt)
        inv = P.inverse(xy).double().numpy()
        ei = testing.angular_error_m(np.degrees(inv[:, 0]), np.degrees(inv[:, 1]), L, B)
        fin_i = np.isfinite(L)
        mism_i = int((np.isfinite(ei) != fin_i).sum())
        res[f"fwd_{tag}"] = float(np.nanmax(ef)) if ef.size else 0.0
        res[f"inv_{tag}"] = float(np.nanmax(ei[fin_i])) if fin_i.any() else 0.0
        res[f"nan_mismatch_{tag}"] = mism + mism_i
        if tag == "fp32":
            res["fwd_fp32_p99"] = float(np.percentile(ef, 99))
            res["inv_fp32_p99"] = float(np.percentile(ei[fin_i], 99))
            res["fwd_fp32_where"] = (float(lon[fin][np.argmax(ef)]), float(lat[fin][np.argmax(ef)]))
            res["inv_fp32_where"] = (float(L[np.nanargmax(np.where(fin_i, ei, -1))]),
                                     float(B[np.nanargmax(np.where(fin_i, ei, -1))]))
    res["proj_failures"] = int((~fin).sum())
    return res


def run_pyproj():
    print("=" * 78)
    print(f"2. Random points vs pyproj {__import__('pyproj').proj_version_str} (n={N_RANDOM}); "
          "max error in metres")
    print(f"  {'definition':<62} {'fwd64':>8} {'inv64':>8} {'fwd32':>8} {'inv32':>8} nanmm")
    out = []
    for i, (d, lo, la, note) in enumerate(PYPROJ_SETS):
        r = compare_pyproj(d, lo, la, note, N_RANDOM, seed=100 + i)
        out.append(r)
        print(f"  {d[:62]:<62} {r['fwd_fp64']:8.1e} {r['inv_fp64']:8.1e} {r['fwd_fp32']:8.1e} "
              f"{r['inv_fp32']:8.1e} {r['nan_mismatch_fp64']}/{r['nan_mismatch_fp32']}")
        print(f"      {note}; kind={r['kind']}; fp32 p99 fwd/inv = {r['fwd_fp32_p99']:.2e}/"
              f"{r['inv_fp32_p99']:.2e} m; fp32 worst fwd at lon,lat={r['fwd_fp32_where'][0]:.3f},"
              f"{r['fwd_fp32_where'][1]:.3f} inv at {r['inv_fp32_where'][0]:.3f},{r['inv_fp32_where'][1]:.3f}")
        if r["fwd_fp64"] > FP64_TOL_M or r["inv_fp64"] > FP64_TOL_M:
            fail(f"pyproj fp64 error too large for {d}: {r['fwd_fp64']:.2e} / {r['inv_fp64']:.2e}")
        if r["nan_mismatch_fp64"]:
            fail(f"pyproj fp64 failure-mask mismatch for {d}: {r['nan_mismatch_fp64']}")
    return out


# =============================================================================
# 3. fp32 error growth with distance (exact tmerc, k0=1, x_0=0)
# =============================================================================
def run_fp32_growth():
    print("=" * 78)
    print("3. fp32 error growth: +proj=tmerc +ellps=WGS84 (exact), inputs cast to fp32,"
          " errors vs fp64 PROJ (m)")
    defn = "+proj=tmerc +ellps=WGS84"
    P = tmerc.make(defn)
    rng = np.random.default_rng(7)
    rows = []
    print(f"  {'northing band':<16} {'|E|<=100km fwd/inv':>22} {'100..300km':>22} {'300..500km':>22}"
          f"  {'ulp32(N)/2':>10}")
    for n0 in range(0, 10000, 1000):
        cells = []
        for e0, e1 in ((0, 100), (100, 300), (300, 500)):
            m = 4000
            E = rng.uniform(e0, e1, m) * 1e3 * rng.choice([-1, 1], m)
            Nn = rng.uniform(n0, n0 + 1000, m) * 1e3
            lon, lat = testing.pyproj_ref(defn, E, Nn, inverse=True)
            X, Y = testing.pyproj_ref(defn, lon, lat)
            inp = torch.tensor(np.radians(np.stack([lon, lat], -1)), dtype=torch.float32)
            o = P(inp).double().numpy()
            ef = np.nanmax(np.hypot(o[:, 0] - X, o[:, 1] - Y))
            L2, B2 = testing.pyproj_ref(defn, X, Y, inverse=True)
            li = P.inverse(torch.tensor(np.stack([X, Y], -1), dtype=torch.float32)).double().numpy()
            ei = np.nanmax(testing.angular_error_m(np.degrees(li[:, 0]), np.degrees(li[:, 1]), L2, B2))
            cells.append((ef, ei))
        half_ulp = float(np.spacing(np.float32(n0 * 1e3 + 999e3))) / 2
        rows.append((n0, cells, half_ulp))
        print(f"  {n0:5d}-{n0 + 1000:5d} km  " + " ".join(f"{a:10.3f}/{b:<10.3f}" for a, b in cells)
              + f"  {half_ulp:10.4f}")
    # the point the brief asks about explicitly
    lon, lat = testing.pyproj_ref(defn, np.array([500e3]), np.array([10000e3]), inverse=True)
    X, Y = testing.pyproj_ref(defn, lon, lat)
    o32 = P(torch.tensor(np.radians([[lon[0], lat[0]]]), dtype=torch.float32)).double().numpy()[0]
    o64 = P(torch.tensor(np.radians([[lon[0], lat[0]]]), dtype=torch.float64)).numpy()[0]
    i32 = P.inverse(torch.tensor([[X[0], Y[0]]], dtype=torch.float32)).double().numpy()[0]
    e_i32 = testing.angular_error_m(np.degrees(i32[0]), np.degrees(i32[1]), lon[0], lat[0])
    print(f"  point E=500 km, N=10000 km (lon={lon[0]:.4f}, lat={lat[0]:.4f}):"
          f" fp32 fwd err={np.hypot(o32[0] - X[0], o32[1] - Y[0]):.3f} m,"
          f" fp32 inv err={e_i32:.3f} m, fp64 fwd err={np.hypot(o64[0] - X[0], o64[1] - Y[0]):.1e} m")
    print("  floor: fp32 spacing is 1 m for |N| in [8388608, 16777216) m, 0.5 m in [4194304, 8388608),"
          " 0.03125 m at 500 km; fp32 spacing of lat=1.5 rad is 1.19e-7 rad = 0.76 m on the ground.")
    # spherical formulas (acos/asin of quantities near 1) collapse in fp32 near the equator
    print("  3b. spherical tmerc (+R=6371000) in fp32, |lon|<30, by latitude band:")
    ds = "+proj=tmerc +R=6371000 +k_0=0.9996"
    S = tmerc.make(ds)
    for latb in ((-0.01, 0.01), (-1, 1), (5, 30), (30, 80)):
        lon = rng.uniform(-30, 30, 20000)
        lat = rng.uniform(*latb, 20000)
        X, Y = testing.pyproj_ref(ds, lon, lat)
        o = S(torch.tensor(np.radians(np.stack([lon, lat], -1)), dtype=torch.float32)).double().numpy()
        L, B = testing.pyproj_ref(ds, X, Y, inverse=True)
        li = S.inverse(torch.tensor(np.stack([X, Y], -1), dtype=torch.float32)).double().numpy()
        ei = testing.angular_error_m(np.degrees(li[:, 0]), np.degrees(li[:, 1]), L, B)
        print(f"      lat {str(latb):<14} fp32 fwd max {np.hypot(o[:, 0] - X, o[:, 1] - Y).max():9.2f} m,"
              f" inv max {ei.max():9.2f} m")
    return rows


# =============================================================================
# 4. autograd
# =============================================================================
def unit_fwd(P):
    return lambda lp: P(lp) / P.a


def unit_inv(P):
    return lambda xy: P.inverse(xy * P.a)


def run_gradcheck():
    print("=" * 78)
    print("4a. torch.autograd.gradcheck (float64, interior points, outputs scaled to unit sphere)")
    rng = np.random.default_rng(3)
    defs = [
        ("+proj=tmerc +ellps=WGS84", (-40, 40), (-80, 80)),
        ("+proj=tmerc +ellps=WGS84 +approx", (-6, 6), (-80, 80)),
        ("+proj=tmerc +ellps=WGS84 +algo=auto", (-8, 8), (-80, 80)),
        ("+proj=tmerc +R=6371000 +lat_0=30 +k_0=0.9", (-80, 80), (-80, 80)),
        ("+proj=utm +zone=32 +south +ellps=GRS80", (5, 13), (-80, -1)),
        ("+proj=tmerc +ellps=airy +lat_0=49 +lon_0=-2 +k_0=0.9996012717 +x_0=400000 +y_0=-100000",
         (-9, 3), (49, 61)),
    ]
    results = []
    for d, lo, la in defs:
        P = tmerc.make(d)
        lon = rng.uniform(*lo, 12)
        lat = rng.uniform(*la, 12)
        lat = np.where(np.abs(lat) < 0.5, lat + 1.0, lat)  # stay off the sphere's acos/sqrt seams
        lp = deg2rad_t(np.stack([lon, lat], -1)).requires_grad_(True)
        okf = torch.autograd.gradcheck(unit_fwd(P), (lp,), eps=1e-6, atol=1e-8, rtol=1e-6,
                                       raise_exception=False)
        xy = (P(lp.detach()) / P.a)[..., :2].clone().requires_grad_(True)
        oki = torch.autograd.gradcheck(unit_inv(P), (xy,), eps=1e-7, atol=1e-8, rtol=1e-6,
                                       raise_exception=False)
        results.append((d, P.kind, okf, oki))
        print(f"  {d[:66]:<66} {P.kind:<15} fwd={'pass' if okf else 'FAIL'} inv={'pass' if oki else 'FAIL'}")
        if not (okf and oki):
            fail(f"gradcheck {d}: fwd={okf} inv={oki}")
    return results


def jac(fn, pt):
    """2x2 Jacobian (rows: outputs 0,1; cols: inputs 0,1) of fn at one point."""
    x = torch.tensor([pt], dtype=torch.float64)
    J = torch.autograd.functional.jacobian(lambda t: fn(t)[0, :2], x)
    return J[:, 0, :].numpy()


def fmt(a):
    return np.array2string(np.asarray(a), precision=3, floatmode="maxprec", suppress_small=False,
                           max_line_width=200).replace("\n", "")


def run_singular():
    print("=" * 78)
    print("4b. Gradients at singular / special points (Jacobian of unit-scaled maps)")
    r = math.radians
    ex = tmerc.make("+proj=tmerc +ellps=WGS84")
    ap = tmerc.make("+proj=tmerc +ellps=WGS84 +approx")
    au = tmerc.make("+proj=tmerc +ellps=WGS84 +algo=auto")
    sp = tmerc.make("+proj=tmerc +R=6371000")
    q_ex = ex.Qn * math.pi / 2  # unit northing of the pole (exact)
    cases = [
        ("exact fwd  origin (0,0)", ex, "f", (0.0, 0.0)),
        ("exact fwd  north pole lat=90", ex, "f", (r(30), r(90))),
        ("exact fwd  south pole lat=-90", ex, "f", (r(30), r(-90))),
        ("exact fwd  lam=90 lat=0 (x=inf -> NaN)", ex, "f", (r(90), 0.0)),
        ("exact fwd  lam=120 lat=10 (beyond 90)", ex, "f", (r(120), r(10))),
        ("exact fwd  antimeridian lam=180 lat=10", ex, "f", (math.pi, r(10))),
        ("exact inv  pole x=0 y=Qn*pi/2", ex, "i", (0.0, q_ex)),
        ("exact inv  |x|>2.62*Qn (NaN)", ex, "i", (2.7, 0.1)),
        ("approx fwd north pole", ap, "f", (r(30), r(90))),
        ("approx fwd lam=100 (NaN)", ap, "f", (r(100), r(10))),
        ("approx inv y beyond pole", ap, "i", (0.01, 1.6)),
        ("auto fwd   lam=3deg-eps (approx side)", au, "f", (r(3) - 1e-12, r(40))),
        ("auto fwd   lam=3deg+eps (exact side)", au, "f", (r(3) + 1e-12, r(40))),
        ("auto fwd   lam=120 (approx NaN, exact used)", au, "f", (r(120), r(10))),
        ("sphere fwd equator phi=0 lam=10", sp, "f", (r(10), 0.0)),
        ("sphere fwd phi=1e-9 lam=10", sp, "f", (r(10), 1e-9)),
        ("sphere fwd phi=1e-5 lam=10", sp, "f", (r(10), 1e-5)),
        ("sphere fwd lam=90 phi=0 (NaN)", sp, "f", (r(90), 0.0)),
        ("sphere fwd north pole", sp, "f", (r(10), r(90))),
        ("sphere inv y=0 (equator)", sp, "i", (0.1, 0.0)),
        ("sphere inv y=1e-6", sp, "i", (0.1, 1e-6)),
        ("sphere inv pole x=0 y=pi/2", sp, "i", (0.0, math.pi / 2)),
    ]
    rows = []
    for label, P, d, pt in cases:
        fn = unit_fwd(P) if d == "f" else unit_inv(P)
        val = fn(torch.tensor([pt], dtype=torch.float64))[0, :2].detach().numpy()
        J = jac(fn, pt)
        state = "finite" if np.isfinite(J).all() else ("NaN" if np.isnan(J).any() else "inf")
        rows.append((label, val, J, state))
        print(f"  {label:<44} out={fmt(val):<36} J={fmt(J.ravel()):<52} {state}")
    # analytic check of the sphere near the equator: dy/dphi = k0 / cos(lam) at phi=0 (unit sphere)
    true = 1.0 / math.cos(r(10))
    for phi in (0.0, 1e-9, 1e-7, 1e-5, 1e-3):
        J = jac(unit_fwd(sp), (r(10), phi))
        print(f"    sphere fwd dy/dphi at phi={phi:<6g}: autograd={J[1, 1]:.10f}  true~{true:.10f}"
              f"  rel.err={abs(J[1, 1] - true) / true:.1e}")
    # pole: core.Projection.forward clamps phi; torch.clamp has zero gradient AT the bound
    exs = tmerc.make("+proj=tmerc +ellps=WGS84")
    lp = torch.tensor([r(30), math.pi / 2], dtype=torch.float64, requires_grad=True)
    gx_core = torch.autograd.grad(exs(lp[None])[0, 0], lp)[0][1].item()
    gx_raw = torch.autograd.grad(exs._exact_fwd(lp[0:1] - exs.lam0, lp[1:2])[0].sum(), lp)[0][1].item()
    print(f"    pole lat=90 lon=30: dx/dphi via core forward = {gx_core:.4f}; via _exact_fwd directly = "
          f"{gx_raw:.4f} (true ~ -0.50: core.py's phi.clamp kills the gradient at exactly +-90)")
    # exact fwd at the same points for contrast
    for phi in (0.0, 1e-9):
        J = jac(unit_fwd(exs), (r(10), phi))
        print(f"    exact  fwd dy/dphi at phi={phi:<6g}: autograd={J[1, 1]:.10f}  (smooth, finite)")
    return rows


def run_nan_leak():
    print("=" * 78)
    print("4c. NaN-leak check: batch with failing + valid points, loss over valid outputs only")
    print("  (failing points are the ones PROJ itself flags; approx_e_inv never flags, so its")
    print("   special input is the beyond-the-pole branch; auto includes points whose UNSELECTED")
    print("   approx branch would overflow)")
    ok_all = True
    for d, pts, xys in [
        ("+proj=tmerc +ellps=WGS84", [(10, 45), (90, 0), (179.9, 0.0), (5, 89.9999)],
         [(1e5, 5e6), (2.0e7, 1e6), (-1e200, 1e5), (3e5, -2e6)]),
        ("+proj=tmerc +ellps=WGS84 +approx", [(10, 45), (100, 10), (-120, -30)],
         [(1e5, 5e6), (1e3, 1.2e7), (3e5, -2e6)]),
        ("+proj=tmerc +ellps=WGS84 +algo=auto", [(10, 45), (100, 10), (90, 0), (1, 1)],
         [(1e5, 5e6), (2.0e7, 1e6), (-1e200, 1e5), (1e200, 1e5), (3e5, -2e6)]),
        ("+proj=tmerc +R=6371000", [(10, 45), (90, 0), (-90, 0)],
         [(1e5, 5e6), (-1e200, 1e5), (3e5, -2e6)]),
    ]:
        P = tmerc.make(d)
        lp = deg2rad_t(pts).requires_grad_(True)
        out = P(lp)
        m = torch.isfinite(out)
        out[m].sum().backward()
        g = lp.grad
        good = bool(torch.isfinite(g).all())
        print(f"  fwd {d:<44} nan outputs={int((~m).any(-1).sum())} grads finite={good}")
        ok_all &= good
        xy = torch.tensor(xys, dtype=torch.float64, requires_grad=True)
        o = P.inverse(xy)
        m = torch.isfinite(o)
        o[m].sum().backward()
        good_i = bool(torch.isfinite(xy.grad).all())
        print(f"  inv {d:<44} nan outputs={int((~m).any(-1).sum())} grads finite={good_i}")
        ok_all &= good_i
    if not ok_all:
        fail("NaN leaked into gradients of a masked batch")
    return ok_all


# =============================================================================
# 5. complex dtype alternative for clenS (justification of real/imag choice)
# =============================================================================
def clenS_complex(a, z):
    """sum a[k] sin((2k+2) z) by complex Clenshaw, z complex tensor."""
    two_cos = 2 * torch.cos(2 * z)
    u0 = torch.zeros_like(z)
    u1 = torch.zeros_like(z)
    for k in range(len(a) - 1, -1, -1):
        u0, u1 = two_cos * u0 - u1 + a[k], u0
    return torch.sin(2 * z) * u0


def run_complex_check():
    print("=" * 78)
    print("5. complex-dtype clenS vs explicit real/imag clenS (tmerc.cpp:270-297)")
    P = tmerc.make("+proj=tmerc +ellps=WGS84")
    rng = np.random.default_rng(5)
    Cn = torch.tensor(rng.uniform(-1.5, 1.5, 1000))
    Ce = torch.tensor(rng.uniform(-1.5, 1.5, 1000))
    R, I = tmerc.clenS(P.utg, torch.sin(2 * Cn), torch.cos(2 * Cn), torch.sinh(2 * Ce), torch.cosh(2 * Ce))
    zc = torch.complex(Cn, Ce)
    w = clenS_complex(P.utg, zc)
    d = max(float((w.real - R).abs().max()), float((w.imag - I).abs().max()))
    inp = torch.stack([Cn[:5], Ce[:5]], -1).clone().requires_grad_(True)

    def f(t):
        w = clenS_complex(P.utg, torch.complex(t[:, 0], t[:, 1]))
        return torch.stack([w.real, w.imag], -1)
    ok = torch.autograd.gradcheck(f, (inp,), eps=1e-6, atol=1e-9, rtol=1e-6, raise_exception=False)
    print(f"  max |complex - real/imag| = {d:.2e} (unit sphere); gradcheck through complex dtype: "
          f"{'pass' if ok else 'FAIL'}")
    if d > 1e-14 or not ok:
        fail("complex clenS check")
    return d, ok


# =============================================================================
# 6. torch.compile (small input)
# =============================================================================
def run_compile():
    print("=" * 78)
    print("6. torch.compile(fullgraph=True) on 64 points (proves no graph breaks / host syncs)")
    out = {}
    for d in ("+proj=utm +zone=32 +ellps=WGS84", "+proj=tmerc +ellps=WGS84 +algo=auto",
              "+proj=tmerc +R=6371000"):
        P = tmerc.make(d)
        rng = np.random.default_rng(11)
        lp = deg2rad_t(np.stack([rng.uniform(5, 13, 64), rng.uniform(-80, 80, 64)], -1))
        t0 = time.time()
        try:
            cf = torch.compile(P.forward, fullgraph=True, dynamic=False)
            ci = torch.compile(P.inverse, fullgraph=True, dynamic=False)
            xy = cf(lp)
            lp2 = ci(xy)
            ref_xy = P(lp)
            ref_lp = P.inverse(ref_xy)
            ex = float((xy - ref_xy).abs().max())
            ei = float((lp2 - ref_lp).abs().max())
            # gradients through the compiled graph
            lpg = lp.clone().requires_grad_(True)
            cf(lpg)[..., 0].sum().backward()
            lpe = lp.clone().requires_grad_(True)
            P(lpe)[..., 0].sum().backward()
            eg = float((lpg.grad - lpe.grad).abs().max() / lpe.grad.abs().max())
            msg = f"ok: |compiled-eager| fwd {ex:.1e} m, inv {ei:.1e} rad, grad rel {eg:.1e}"
            ok = ex < 1e-6 and ei < 1e-12 and eg < 1e-12
        except Exception as e:  # noqa: BLE001
            msg, ok = f"error: {type(e).__name__}: {str(e).splitlines()[0][:150]}", False
        print(f"  {d:<44} {msg}  ({time.time() - t0:.1f}s)")
        out[d] = (ok, msg)
        if not ok:
            fail(f"torch.compile {d}: {msg}")
    # informational: Inductor and complex dtypes (motivates real/imag clenS)
    import warnings
    P = tmerc.make("+proj=tmerc +ellps=WGS84")
    cn = torch.rand(64, dtype=torch.float64)
    ce = torch.rand(64, dtype=torch.float64)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = torch.compile(lambda a, b: clenS_complex(P.utg, torch.complex(a, b)), fullgraph=True)(cn, ce)
    e = float((r - clenS_complex(P.utg, torch.complex(cn, ce))).abs().max())
    msgs = [str(x.message)[:90] for x in w if "complex" in str(x.message)]
    print(f"  complex-dtype clenS compiled: |diff|={e:.1e}; inductor warning: {msgs[:1] or 'none'}")
    return out


def main():
    t0 = time.time()
    gie = run_gie()
    run_pyproj()
    run_fp32_growth()
    run_gradcheck()
    run_singular()
    run_nan_leak()
    run_complex_check()
    if DO_COMPILE:
        run_compile()
    print("=" * 78)
    print(f"SUMMARY: gie {gie['passed']}/{gie['total']} passed ({gie['blocks']} blocks); "
          f"{len(FAILURES)} failure(s); {time.time() - t0:.1f}s")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
