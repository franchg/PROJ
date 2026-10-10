"""Adversarial verification of tmerc.py (tmerc / etmerc / utm) against PROJ itself.

Independent of tmerc_test.py: own gie runner (setup-failure expectation taken from
PROJ, not from re-parsing the gie file), radians-direct PROJ oracle (no degree round
trip), wide-domain failure-mask comparison, ulp-level probes at every PROJ threshold,
parameter-parsing attacks, FD-vs-autograd Jacobians, NaN-gradient leak scans, shape /
dtype / passthrough checks, a graph-break scan with torch.compile(backend="eager").

Run:  python tmerc_adv_test.py            (about 1-2 min on one core)
Prints every finding as  ** [SEVERITY] ...  and a summary at the end.
"""
import inspect
import math
import re
import sys
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

from pyproj import Transformer  # noqa: E402
from pyproj.enums import TransformDirection as TD  # noqa: E402

import testing  # noqa: E402
import tmerc  # noqa: E402
from pathlib import Path

_PROJ_ROOT = Path(__file__).resolve().parents[2]  # root of the PROJ source tree

HP = math.pi / 2
FIND = []
T0 = time.time()


def finding(sev, sec, msg):
    FIND.append((sev, sec, msg))
    print(f"  ** [{sev}] {msg}")


def hdr(s):
    print("=" * 100)
    print(s)


_TR = {}


def _tr(defn):
    if defn not in _TR:
        _TR[defn] = Transformer.from_pipeline(defn)
    return _TR[defn]


def proj_ok(defn):
    try:
        Transformer.from_pipeline(defn)
        return True
    except Exception:  # noqa: BLE001
        return False


def ref(defn, a, b, inverse=False, z=None):
    """PROJ with radians in/out (pyproj radians=True), errors -> NaN."""
    t = _tr(defn)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    dr = TD.INVERSE if inverse else TD.FORWARD
    if z is None:
        x, y = t.transform(a, b, direction=dr, radians=True)
        zz = None
    else:
        x, y, zz = t.transform(a, b, np.asarray(z, float), direction=dr, radians=True)
    x = np.where(np.isinf(x), np.nan, np.asarray(x, float))
    y = np.where(np.isinf(y), np.nan, np.asarray(y, float))
    return (x, y) if z is None else (x, y, np.asarray(zz, float))


def mine(P, a, b, inverse=False, dtype=torch.float64):
    t = torch.tensor(np.stack([np.asarray(a, float), np.asarray(b, float)], -1)).to(dtype)
    o = (P.inverse(t) if inverse else P(t)).double().numpy()
    return o[:, 0], o[:, 1]


def ang_err_m(L1, B1, L2, B2, a):
    dl = np.remainder(L2 - L1 + math.pi, 2 * math.pi) - math.pi
    return a * np.hypot(B2 - B1, np.cos(0.5 * (B1 + B2)) * dl)


def plus(op):
    return " ".join(t if t.startswith("+") else "+" + t for t in op.split())


# =====================================================================================
# 0. Coefficient tables copied into tmerc.py vs latitudes.cpp
# =====================================================================================
def check_coeff_tables():
    hdr("0. _AUX_COEFFS in tmerc.py vs the coeffs[]/ptrs[] arrays of src/latitudes.cpp")
    src = open(f"{_PROJ_ROOT}/src/latitudes.cpp").read()
    body = re.search(r"static const double coeffs\[\] = \{(.*?)\};", src, re.S).group(1)
    body = re.sub(r"//[^\n]*", "", body)
    vals = []
    for tok in body.split(","):
        tok = tok.strip()
        if not tok:
            continue
        m = re.fullmatch(r"(-?[0-9.]+)\s*/\s*([0-9.]+)", tok)
        vals.append(float(m.group(1)) / float(m.group(2)) if m else float(tok))
    ptrs = [int(v) for v in re.search(r"static constexpr int ptrs\[\] = \{(.*?)\};", src, re.S)
            .group(1).replace("\n", " ").split(",") if v.strip()]
    bad = 0
    for (aout, ain), arr in tmerc._AUX_COEFFS.items():
        k = 6 * aout + ain
        cpp = vals[ptrs[k]:ptrs[k + 1]]
        same = len(cpp) == len(arr) and all(x == y for x, y in zip(cpp, arr))
        bad += not same
        print(f"  C[{aout},{ain}] {len(arr)} coeffs: {'identical' if same else 'DIFFERENT'}")
    if bad:
        finding("HIGH", "coeffs", f"{bad} coefficient tables differ from latitudes.cpp")
    # setup-time helpers vs a direct re-implementation
    P = tmerc.make("+proj=tmerc +ellps=WGS84")
    n = P.n
    print(f"  third flattening n={n!r}; Qn={P.Qn!r} Zb={P.Zb!r}")


# =====================================================================================
# 1. gie, independently re-derived
# =====================================================================================
def run_gie():
    hdr("1. gie vectors re-derived (setup expectation from PROJ itself, not from the gie text)")
    blocks = cases = passed = 0
    fails = []
    for name in ("tmerc", "utm", "etmerc"):
        for blk in testing.gie_cases(name):
            blocks += 1
            op = blk["operation"]
            pok = proj_ok(plus(op))
            try:
                P = tmerc.make(op)
                err = None
            except Exception as e:  # noqa: BLE001
                P, err = None, e
            n_c = len(blk["cases"]) + len(blk["roundtrips"])
            if n_c == 0:
                cases += 1
                ok = (P is not None) == pok
                passed += ok
                if not ok:
                    fails.append(f"{op}: PROJ setup ok={pok}, torch err={err!r}")
                continue
            if P is None:
                cases += n_c
                fails.append(f"{op}: torch setup failed {err!r} but PROJ ok={pok}")
                continue
            for c in blk["cases"]:
                cases += 1
                acc = list(c["accept"]) + [0.0] * (4 - len(c["accept"]))
                if c["direction"] == "fwd":
                    inp = torch.tensor([[math.radians(acc[0]), math.radians(acc[1]), acc[2], acc[3]]],
                                       dtype=torch.float64)
                    out = P(inp)[0].numpy()
                    if c["expect"] == "failure":
                        ok = bool(np.isnan(out[:2]).all())
                    else:
                        e = c["expect"]
                        ok = math.hypot(out[0] - e[0], out[1] - e[1]) <= c["tolerance_m"]
                        if len(e) > 2:
                            ok &= bool(np.allclose(out[2:2 + len(e) - 2], e[2:], atol=1e-9))
                else:
                    out = P.inverse(torch.tensor([acc], dtype=torch.float64))[0].numpy()
                    if c["expect"] == "failure":
                        ok = bool(np.isnan(out[:2]).all())
                    else:
                        e = c["expect"]
                        d = ang_err_m(math.radians(e[0]), math.radians(e[1]), out[0], out[1], P.a)
                        ok = d <= c["tolerance_m"]
                passed += ok
                if not ok:
                    fails.append(f"{op}: {c}")
            for r in blk["roundtrips"]:
                cases += 1
                lp0 = torch.tensor([[math.radians(r["accept"][0]), math.radians(r["accept"][1])]],
                                   dtype=torch.float64)
                lp = lp0
                for _ in range(r["n"]):
                    lp = P.inverse(P(lp))
                d = ang_err_m(lp0[0, 0].item(), lp0[0, 1].item(), lp[0, 0].item(), lp[0, 1].item(), P.a)
                ok = d <= r["tolerance_m"]
                passed += ok
                if not ok:
                    fails.append(f"{op}: roundtrip {r} -> {d}")
    print(f"  gie: blocks={blocks} cases={cases} passed={passed}")
    for f in fails:
        print("   FAIL", f)
    # gie blocks excluded by testing.gie_cases because they are pipelines (more_builtins.gie:127-188)
    U = tmerc.make("+proj=utm +zone=32 +ellps=GRS80")
    xy = U(torch.tensor([[math.radians(12), math.radians(55), 0, 0]], dtype=torch.float64))[0].numpy()
    d1 = math.hypot(xy[0] - 691875.63214, xy[1] - 6098907.82501)
    lp = U.inverse(torch.tensor([[691875.63214, 6098907.82501, 0, 0]], dtype=torch.float64))[0].numpy()
    d2 = ang_err_m(math.radians(12), math.radians(55), lp[0], lp[1], U.a)
    print(f"  extra: single-step pipeline block more_builtins.gie:182 fwd err {d1:.2e} m, inv err {d2:.2e} m "
          f"(tol 5e-4): {'ok' if max(d1, d2) < 5e-4 else 'FAIL'}")
    if fails:
        finding("HIGH", "gie", f"{len(fails)} gie cases fail in the independent runner")
    return blocks, cases, passed


# =====================================================================================
# 2. Headline pyproj numbers re-derived (new seeds), degree oracle AND radians oracle
# =====================================================================================
SETS = [
    ("+proj=utm +zone=32 +ellps=WGS84", (3, 15), (0, 84), (1.9e-9, 2.8e-9, 2.2, 2.4)),
    ("+proj=utm +zone=32 +ellps=WGS84 +approx", (3, 15), (0, 84), (9.3e-10, 2.8e-9, 2.3, 1.9)),
    ("+proj=utm +zone=33 +south +ellps=WGS84", (9, 21), (-80, 0), (1.9e-9, 2.8e-9, 1.9, 2.4)),
    ("+proj=tmerc +lon_0=10 +ellps=WGS84", (-30, 50), (-84, 84), (2.1e-9, 2.8e-9, 2.2, 2.1)),
    ("+proj=tmerc +lon_0=10 +ellps=WGS84 +algo=auto", (0, 20), (-84, 84), (1.9e-9, 2.8e-9, 2.1, 2.1)),
    ("+proj=tmerc +lat_0=49 +lon_0=-2 +k_0=0.9996012717 +x_0=400000 +y_0=-100000 +ellps=airy",
     (-9, 3), (49, 61), (3e-9, 2.1e-9, 2.1, 2.3)),
    ("+proj=tmerc +R=6371000 +k_0=0.9996", (-170, 170), (-85, 85), (3.3e-7, 8.5e-9, 43000, 1300)),
]


def run_headline(n=100_000):
    hdr(f"2. Headline accuracy re-derived, n={n}, new seeds. Columns: max error (m); "
        "'rad' = PROJ called with radians directly")
    print(f"  {'definition':<58} {'fwd64':>8} {'inv64':>8} {'inv64rad':>8} {'fwd32':>8} {'inv32':>8}"
          f" {'p99f32':>7} {'p99i32':>7} | claimed fwd64/inv64/fwd32/inv32")
    rows = []
    for i, (d, lo, la, claim) in enumerate(SETS):
        rng = np.random.default_rng(9000 + i)
        lon = rng.uniform(*lo, n)
        lat = rng.uniform(*la, n)
        P = tmerc.make(d)
        X, Y = testing.pyproj_ref(d, lon, lat)
        L, B = testing.pyproj_ref(d, X, Y, inverse=True)
        Lr, Br = ref(d, X, Y, inverse=True)
        res = {}
        for dt, tag in ((torch.float64, "64"), (torch.float32, "32")):
            x, y = mine(P, np.radians(lon), np.radians(lat), dtype=dt)
            ef = np.hypot(x - X, y - Y)
            l, b = mine(P, X, Y, inverse=True, dtype=dt)
            ei = ang_err_m(np.radians(L), np.radians(B), l, b, 6378137.0)
            res["fwd" + tag], res["inv" + tag] = np.nanmax(ef), np.nanmax(ei)
            res["nan" + tag] = int(np.isnan(ef).sum() + np.isnan(ei).sum())
            if tag == "64":
                res["invrad"] = np.nanmax(ang_err_m(Lr, Br, l, b, 6378137.0))
            else:
                res["p99f"], res["p99i"] = np.percentile(ef, 99), np.percentile(ei, 99)
        rows.append((d, res, claim))
        print(f"  {d[:58]:<58} {res['fwd64']:8.1e} {res['inv64']:8.1e} {res['invrad']:8.1e} {res['fwd32']:8.1e}"
              f" {res['inv32']:8.1e} {res['p99f']:7.2f} {res['p99i']:7.2f} | {claim}")
        if res["nan64"] or res["nan32"]:
            finding("MEDIUM", "headline", f"{d}: unexpected NaN outputs ({res['nan64']}/{res['nan32']})")
        if res["fwd64"] > 5 * claim[0] + 1e-9 or res["inv64"] > 5 * claim[1] + 1e-9:
            finding("LOW", "headline", f"{d}: fp64 error {res['fwd64']:.1e}/{res['inv64']:.1e} m exceeds the "
                    f"claimed {claim[0]:.1e}/{claim[1]:.1e} by >5x")
        if res["fwd32"] > 3 * claim[2] or res["inv32"] > 3 * claim[3]:
            finding("LOW", "headline", f"{d}: fp32 error {res['fwd32']:.2f}/{res['inv32']:.2f} m exceeds the "
                    f"claimed {claim[2]}/{claim[3]} by >3x")
    return rows


# =====================================================================================
# 3. Failure masks: wide domains + ulp-level probes at every PROJ threshold
# =====================================================================================
def run_masks():
    hdr("3. Failure masks vs PROJ on wide domains (where PROJ really fails) and at thresholds")
    rng = np.random.default_rng(31)
    n = 150_000
    for d in ["+proj=tmerc +ellps=WGS84", "+proj=tmerc +ellps=WGS84 +approx", "+proj=tmerc +R=6371000",
              "+proj=tmerc +ellps=WGS84 +algo=auto", "+proj=utm +zone=60 +south +ellps=GRS80",
              "+proj=etmerc +ellps=intl +lat_0=-30 +k_0=0.5 +x_0=1e6", "+proj=tmerc +a=6378137 +rf=50"]:
        P = tmerc.make(d)
        lam = np.concatenate([rng.uniform(-math.pi, math.pi, n),
                              rng.choice([-1, 1], n) * rng.uniform(HP - 0.2, HP + 0.2, n)]) + P.lam0
        phi = np.concatenate([rng.uniform(-HP, HP, n), rng.uniform(-0.2, 0.2, n)])
        X, Y = ref(d, lam, phi)
        x, y = mine(P, lam, phi)
        fP, fT = np.isfinite(X), np.isfinite(x)
        e = np.hypot(X - x, Y - y)[fP & fT]
        xx = rng.uniform(-3.2, 3.2, 2 * n) * P.a + P.x0
        yy = rng.uniform(-3.5, 3.5, 2 * n) * P.a + P.y0
        L, B = ref(d, xx, yy, inverse=True)
        l, b = mine(P, xx, yy, inverse=True)
        gP, gT = np.isfinite(L), np.isfinite(l)
        ei = ang_err_m(L, B, l, b, P.a)[gP & gT & (np.abs(B) < 10)]
        print(f"  {d[:52]:<52} fwd: PROJ fails {(~fP).sum():6d} mismatch {(fP != fT).sum():3d} max err {e.max():.1e} m"
              f" | inv: PROJ fails {(~gP).sum():6d} mismatch {(gP != gT).sum():3d} max err {ei.max():.1e} m")
        if (fP != fT).sum() or (gP != gT).sum():
            finding("MEDIUM", "masks", f"{d}: failure mask differs from PROJ on {(fP != fT).sum()} fwd / "
                    f"{(gP != gT).sum()} inv random points")

    # --- ulp probes at thresholds -----------------------------------------------------
    def probe(label, d, pts, inverse=False, tol=1e-9):
        P = tmerc.make(d)
        a, b = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
        X, Y = ref(d, a, b, inverse=inverse)
        x, y = mine(P, a, b, inverse=inverse)
        mm = np.isfinite(X) != np.isfinite(x)
        both = np.isfinite(X) & np.isfinite(x)
        if inverse:
            err = np.where(both, ang_err_m(X, Y, x, y, P.a), 0)
        else:
            err = np.where(both, np.hypot(X - x, Y - y), 0)
        bad = mm | (err > tol)
        print(f"  {label:<62} n={len(pts):5d} mask-mismatch={int(mm.sum()):3d} value-mismatch(>{tol:g} m)="
              f"{int((err > tol).sum()):3d} max err={err.max():.1e}")
        for i in np.where(bad)[0][:3]:
            print(f"      in=({a[i]!r},{b[i]!r}) PROJ=({X[i]!r},{Y[i]!r}) torch=({x[i]!r},{y[i]!r})")
        return int(mm.sum()), int((err > tol).sum())

    def ulps(v, k=4):
        out = [v]
        lo = hi = v
        for _ in range(k):
            lo, hi = np.nextafter(lo, -np.inf), np.nextafter(hi, np.inf)
            out += [lo, hi]
        return out

    res = {}
    # latitude range check (fwd.cpp:46-69): phi - HALFPI > PJ_EPS_LAT
    pts = [(0.3, s * p) for s in (1, -1) for p in ulps(HP + 1e-12, 6) + ulps(HP, 2) + [HP - 1e-12]]
    for d in ("+proj=tmerc +ellps=WGS84", "+proj=tmerc +R=6371000", "+proj=tmerc +ellps=WGS84 +approx"):
        res["lat_eps " + d] = probe(f"lat = +-(pi/2 + 1e-12) +- 6 ulp  [{d.split(' ', 1)[1]}]", d, pts)
    # longitude range check (fwd.cpp:72-79): |lam| > 10 rad is an error even with +over
    lams = [9.9, 10.0, np.nextafter(10.0, 11), 10.5, -10.5, 12.0, 20.0, 100.0]
    for d in ("+proj=tmerc +ellps=WGS84", "+proj=tmerc +ellps=WGS84 +over", "+proj=tmerc +R=6371000 +over"):
        res["lon10 " + d] = probe(f"|lon| around 10 rad  [{d.split(' ', 1)[1]}]", d, [(v, 0.2) for v in lams])
    # antimeridian, lon_0 +-180, outside [-pi, pi], with and without +over
    lam_list = [math.pi, -math.pi, np.nextafter(math.pi, 4), np.nextafter(-math.pi, -4), math.pi + 1e-12,
                math.pi + 2e-12, 3.5, -3.5, 2 * math.pi, 7.0, -7.0, 9.42]
    for d in ("+proj=tmerc +ellps=WGS84", "+proj=tmerc +ellps=WGS84 +over", "+proj=tmerc +ellps=WGS84 +lon_0=180",
              "+proj=tmerc +ellps=WGS84 +lon_0=-180", "+proj=tmerc +ellps=WGS84 +lon_0=-180 +over",
              "+proj=tmerc +R=6371000 +lon_0=180", "+proj=tmerc +ellps=WGS84 +approx +lon_0=170 +over",
              "+proj=utm +zone=1 +ellps=WGS84", "+proj=utm +zone=60 +ellps=WGS84 +over"):
        pts = [(v, p) for v in lam_list for p in (0.0, -0.0, 0.3, -1.2, HP, -HP)]
        res["lon " + d] = probe(f"antimeridian / lon outside [-pi,pi]  [{d.split(' ', 1)[1]}]", d, pts, tol=1e-8)
    # exact fwd: Ce limit 2.623395162778 (tmerc.cpp:375). bisect the boundary with torch, probe PROJ at +-ulps
    P = tmerc.make("+proj=tmerc +ellps=WGS84")
    pts = []
    for phi in np.linspace(-0.12, 0.12, 25):
        lo, hi = 1.2, HP  # valid at lo, invalid at hi (phi small)
        f = lambda lam: bool(torch.isnan(P(torch.tensor([[lam, phi]], dtype=torch.float64))[0, 0]))  # noqa: E731
        if not f(hi) or f(lo):
            continue
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if f(mid) else (mid, hi)
        pts += [(v, phi) for v in ulps(lo, 3)]
    res["Ce_fwd"] = probe("exact fwd: |Ce| = 2.623395162778 boundary +-3 ulp (bisected)",
                          "+proj=tmerc +ellps=WGS84", pts, tol=1e-6)
    # exact inv: |x/Qn| = 2.623395162778
    xs = ulps(tmerc.EXACT_CE_MAX * P.Qn * P.a, 6)
    res["Ce_inv"] = probe("exact inv: |x| = 2.623395162778*Qn*a +- 6 ulp",
                          "+proj=tmerc +ellps=WGS84", [(s * v, y) for v in xs for s in (1, -1) for y in (0, 3e6)],
                          inverse=True, tol=1e-6)
    # auto fwd switch at |lam| = 3 deg
    thr = 3 * tmerc.DEG_TO_RAD
    res["auto_fwd"] = probe("auto fwd: |lam| = 3 deg +- 4 ulp (approx/exact differ ~1e-5 m there)",
                            "+proj=tmerc +ellps=WGS84 +algo=auto",
                            [(s * v, p) for v in ulps(thr) for s in (1, -1) for p in (0.0, 0.7, -1.3)], tol=1e-10)
    # auto inv switch |x| = 0.053 - 0.022 y^2 (unit ellipsoid)
    A = tmerc.make("+proj=tmerc +ellps=WGS84 +algo=auto")
    pts = []
    for yu in (0.0, 0.5, 1.0, 1.4, -1.2):
        xu = 0.053 - 0.022 * yu * yu
        for v in ulps(xu * A.a, 4):
            pts.append((v, yu * A.a))
    res["auto_inv"] = probe("auto inv: |x| = 0.053 - 0.022 y^2 +- 4 ulp", "+proj=tmerc +ellps=WGS84 +algo=auto",
                            pts, inverse=True, tol=1e-10)
    # approx fwd |lam| = 90 deg exactly (tmerc.cpp:83)
    res["approx90"] = probe("approx fwd: |lam| = pi/2 +- 4 ulp", "+proj=tmerc +ellps=WGS84 +approx",
                            [(s * v, p) for v in ulps(HP) for s in (1, -1) for p in (0.0, 0.5)], tol=1e-6)
    # approx: |cos phi| > 1e-10 test (tmerc.cpp:93,170)
    phis = [HP - 1e-10 + k * 2e-12 for k in range(-20, 21)]
    res["approx_cos"] = probe("approx fwd/inv: |cos phi| = 1e-10 threshold",
                              "+proj=tmerc +ellps=WGS84 +approx", [(0.01, p) for p in phis], tol=1e-8)
    # sphere fwd: ||b|-1| <= EPS10 (tmerc.cpp:125) and cos(phi) == 1.0 (tmerc.cpp:131)
    pts = []
    for lam in (HP, HP - 1e-5, HP + 3e-6, -HP):
        for phi in np.linspace(-3e-5, 3e-5, 61):
            pts.append((lam, phi))
    res["sph_eps10"] = probe("sphere fwd: ||b|-1| = 1e-10 neighbourhood", "+proj=tmerc +R=6371000", pts, tol=1e-3)
    pts = [(lam, s * p) for lam in (0.2, 2.0, -2.5) for s in (1, -1) for p in np.linspace(0.9e-8, 1.2e-8, 40)]
    res["sph_cos1"] = probe("sphere fwd: cos(phi)==1 threshold (|phi| ~ 1.05e-8)", "+proj=tmerc +R=6371000",
                            pts, tol=1e-9)
    # sphere inverse: exp underflow (tmerc.cpp:204-207)
    res["sph_exp"] = probe("sphere inv: exp(x/esp) underflow near x/esp = -745",
                           "+proj=tmerc +R=1", [(-v, 0.1) for v in np.linspace(744.0, 746.0, 41)] +
                           [(v, 0.1) for v in (709.0, 709.8, 710.0, 800.0)], inverse=True, tol=1e-6)
    return res


# =====================================================================================
# 4. Edge points vs PROJ (poles, hemispheres, k_0, x_0/y_0, ellipsoids, lat_0, EPS-ish)
# =====================================================================================
EDGE_DEFS = [
    "+proj=tmerc +ellps=WGS84",
    "+proj=tmerc +ellps=WGS84 +approx",
    "+proj=tmerc +ellps=WGS84 +algo=auto",
    "+proj=tmerc +R=6371000",
    "+proj=tmerc +R=6371000 +lat_0=90 +k_0=0.5",
    "+proj=tmerc +R=6371000 +lat_0=-90",
    "+proj=tmerc +ellps=GRS80 +lat_0=90",
    "+proj=tmerc +ellps=GRS80 +lat_0=-90 +approx",
    "+proj=tmerc +ellps=bessel +lat_0=-33.5 +lon_0=-70 +k_0=1.5 +x_0=-1e6 +y_0=2e7",
    "+proj=tmerc +ellps=krass +k_0=1e-6",
    "+proj=tmerc +ellps=krass +k_0=100 +approx",
    "+proj=tmerc +a=6378137 +rf=50",
    "+proj=tmerc +a=6378137 +rf=20 +approx",
    "+proj=tmerc +a=6378137 +rf=5",
    "+proj=tmerc +a=6378137 +es=0.1 +algo=auto",
    "+proj=tmerc +a=6378137 +es=0.1000001 +algo=auto",
    "+proj=tmerc +a=6378137 +rf=1e12",
    "+proj=tmerc +a=6378137 +es=1e-300",
    "+proj=tmerc +a=6400000 +b=6000000",
    "+proj=tmerc +ellps=clrk66 +k_0=0.99 +algo=auto",
    "+proj=tmerc +ellps=WGS84 +k_0=1.01 +algo=auto",
    "+proj=tmerc +ellps=WGS84 +lat_0=1e-15 +algo=auto",
    "+proj=etmerc +ellps=GRS80 +lon_0=-179.5",
    "+proj=utm +zone=1 +south +ellps=WGS84",
    "+proj=utm +zone=60 +ellps=intl +approx",
    "+proj=utm +lon_0=-183 +ellps=WGS84 +algo=auto",
    "+proj=utm +zone=30 +ellps=GRS80 +to_meter=0.3048006096012192",
]


def run_edges():
    hdr("4. Edge points vs PROJ (radians oracle) for 27 definitions; fwd tol 1e-6 m, inv tol 1e-6 m")
    lam_e = [0.0, 1e-300, 1e-12, 0.05, -0.05, 0.6, -1.4, HP, -HP, 2.0, -3.0, math.pi, -math.pi]
    phi_e = [0.0, -0.0, 1e-300, 1e-12, -1e-12, 1e-8, -0.7, 0.9, HP - 1e-12, -(HP - 1e-12), HP, -HP,
             HP - 1e-6, -1.55]
    tot_m = tot_v = 0
    for d in EDGE_DEFS:
        P = tmerc.make(d)
        pts = [(l + P.lam0, p) for l in lam_e for p in phi_e]
        a, b = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
        X, Y = ref(d, a, b)
        x, y = mine(P, a, b)
        mm = np.isfinite(X) != np.isfinite(x)
        both = np.isfinite(X) & np.isfinite(x)
        scale = np.maximum(1.0, np.hypot(X, Y) * 1e-14)  # allow 1e-14 relative for huge coordinates
        err = np.where(both, np.hypot(X - x, Y - y) / scale, 0)
        # inverse at PROJ's own valid outputs + raw grid
        xs = np.concatenate([X[both], P.x0 + P.a * np.array([0, 0.01, -0.5, 1.2, 2.6, -2.7, 0.0, 3.0])])
        ys = np.concatenate([Y[both], P.y0 + P.a * np.array([0, 1.5, -1.5707, 1.6, 0.1, 0.0, 3.2, -2.0])])
        L, B = ref(d, xs, ys, inverse=True)
        l, bb = mine(P, xs, ys, inverse=True)
        mmi = np.isfinite(L) != np.isfinite(l)
        bothi = np.isfinite(L) & np.isfinite(l) & (np.abs(B) <= HP + 1e-9)
        erri = np.where(bothi, ang_err_m(L, B, l, bb, P.a), 0)
        nv, nvi = int((err > 1e-6).sum()), int((erri > 1e-6).sum())
        tot_m += int(mm.sum() + mmi.sum())
        tot_v += nv + nvi
        print(f"  {d[:64]:<64} fwd mask-mm {int(mm.sum()):2d} >tol {nv:2d} max {err.max():.1e} | "
              f"inv mask-mm {int(mmi.sum()):2d} >tol {nvi:2d} max {erri.max():.1e}")
        for i in list(np.where(mm | (err > 1e-6))[0][:2]):
            print(f"      fwd in=({a[i]!r},{b[i]!r}) PROJ=({X[i]!r},{Y[i]!r}) torch=({x[i]!r},{y[i]!r})")
        for i in list(np.where(mmi | (erri > 1e-6))[0][:2]):
            print(f"      inv in=({xs[i]!r},{ys[i]!r}) PROJ=({L[i]!r},{B[i]!r}) torch=({l[i]!r},{bb[i]!r})")
    return tot_m, tot_v


# =====================================================================================
# 5. Setup / parameter semantics vs PROJ
# =====================================================================================
SETUP_DEFS = [
    # boolean flags: PROJ's pj_param "b" type parses =F/=f/=false as FALSE (param.cpp:195-210)
    "+proj=utm +zone=32 +ellps=WGS84 +south=false",
    "+proj=utm +zone=32 +ellps=WGS84 +south=F",
    "+proj=tmerc +ellps=WGS84 +approx=false",
    "+proj=utm +zone=32 +ellps=WGS84 +approx=f",
    # init.cpp:653-677 checks
    "+proj=tmerc +ellps=WGS84 +k_0=0",
    "+proj=tmerc +ellps=WGS84 +k_0=-1",
    "+proj=tmerc +ellps=WGS84 +k=-0.5",
    "+proj=tmerc +ellps=WGS84 +lat_0=91",
    # ell_set.cpp: R overrules shape keys; shape-key precedence rf > f > es > e > b
    "+proj=tmerc +R=6371000 +rf=0",
    "+proj=tmerc +R=6371000 +es=2",
    "+proj=tmerc +a=6378137 +rf=298.257223563 +b=6000000",
    "+proj=tmerc +a=6378137 +rf=300 +f=0.01",
    "+proj=tmerc +a=6378137 +es=0.006 +e=0.1",
    "+proj=tmerc +a=6378137 +rf=1e400",
    "+proj=tmerc +rf=298.257223563",
    # spherification (ell_set.cpp:343-420) silently ignored?
    "+proj=tmerc +ellps=WGS84 +R_A",
    "+proj=tmerc +ellps=WGS84 +R_a",
    "+proj=utm +zone=32 +ellps=WGS84 +R_A",
    # prime meridian / geocentric latitude / axis / units precedence / z_0 (fwd.cpp:90-118, init.cpp)
    "+proj=tmerc +ellps=WGS84 +pm=paris",
    "+proj=tmerc +ellps=WGS84 +pm=10",
    "+proj=tmerc +ellps=WGS84 +geoc",
    "+proj=tmerc +ellps=WGS84 +axis=wsu",
    "+proj=tmerc +ellps=WGS84 +units=m +to_meter=2",
    "+proj=tmerc +ellps=WGS84 +units=us-ft",
    "+proj=tmerc +ellps=WGS84 +lat_0=10d30",
    # UTM zone parser (param.cpp:180-189)
    "+proj=utm +ellps=WGS84 +zone=032",
    "+proj=utm +ellps=WGS84 +zone=+32",
    "+proj=utm +ellps=WGS84 +zone=٣٢",
    "+proj=utm +ellps=WGS84 +lon_0=-1000",
    "+proj=utm +ellps=WGS84 +lon_0=1e6",
    "+proj=tmerc +ellps=WGS84 +algo=Auto",
]


def run_setup():
    hdr("5. Setup / parameter semantics vs PROJ (point lon=10 lat=40 unless the zone needs another)")
    nbad = 0
    for d in SETUP_DEFS:
        pok = proj_ok(d)
        try:
            P = tmerc.make(d)
            terr = None
        except Exception as e:  # noqa: BLE001
            P, terr = None, f"{type(e).__name__}: {str(e)[:50]}"
        lon, lat = (10.0, 40.0)
        verdict, detail = "ok", ""
        if pok and P is not None:
            X, Y = ref(d, [math.radians(lon)], [math.radians(lat)])
            x, y = mine(P, [math.radians(lon)], [math.radians(lat)])
            e = float(np.hypot(X - x, Y - y)[0]) if np.isfinite(X[0]) and np.isfinite(x[0]) else (
                0.0 if np.isfinite(X[0]) == np.isfinite(x[0]) else float("inf"))
            detail = f"PROJ ({X[0]:.3f}, {Y[0]:.3f})  torch ({x[0]:.3f}, {y[0]:.3f})  diff {e:.3g} m"
            if not e <= 1e-6:
                verdict = "SILENTLY WRONG"
        elif pok and P is None:
            verdict, detail = "torch refuses, PROJ accepts", terr
        elif not pok and P is not None:
            x, y = mine(P, [math.radians(lon)], [math.radians(lat)])
            verdict, detail = "torch accepts, PROJ refuses", f"torch gives ({x[0]:.3f}, {y[0]:.3f})"
        else:
            detail = f"both refuse ({terr})"
        nbad += verdict != "ok"
        print(f"  [{verdict:<27}] {d:<52} {detail}")
    if nbad:
        finding("MEDIUM", "setup", f"{nbad}/{len(SETUP_DEFS)} parameter cases disagree with PROJ (see section 5)")
    # +over=false: core.Projection uses bool(params['over']) -> True; visible on the inverse longitude range
    d = "+proj=tmerc +ellps=WGS84 +lon_0=170 +over=false"
    P = tmerc.make(d)
    X, Y = ref(d, [math.radians(-170.0)], [0.5])
    L, _ = ref(d, X, Y, inverse=True)
    l, _ = mine(P, X, Y, inverse=True)
    print(f"  {d}: inverse lon PROJ {math.degrees(L[0]):.6f}  torch {math.degrees(l[0]):.6f}")
    if abs(L[0] - l[0]) > 1e-9:
        finding("MEDIUM", "setup", "boolean flags written as =false/=F are parsed as TRUE (+south, +approx in tmerc.py; "
                "+over in core.py): utm +south=false is 10,000 km off, +over=false returns lon 190 instead of -170")
    return nbad


def run_fp32_pole():
    hdr("5b. float32 at / near the pole (every lat/lon grid with a pole row hits this)")
    f32 = np.float32
    v = f32(HP)
    vals = [v]
    for _ in range(40):
        v = np.nextafter(v, f32(0))
        vals.append(v)
    vals = np.array(vals, dtype=np.float32).astype(float)
    for d in ["+proj=tmerc +ellps=WGS84", "+proj=tmerc +ellps=WGS84 +approx", "+proj=utm +zone=32 +ellps=WGS84 +approx",
              "+proj=tmerc +ellps=WGS84 +algo=auto", "+proj=tmerc +R=6371000"]:
        P = tmerc.make(d)
        lam = np.full(len(vals), P.lam0 + 0.01)
        x, y = mine(P, lam, vals, dtype=torch.float32)
        x64, y64 = mine(P, lam, np.minimum(vals, HP))  # fp64 port at the clamped latitude = intended answer
        nonfin = ~(np.isfinite(x) & np.isfinite(y))
        err = np.hypot(x - x64, y - y64)
        print(f"  {d:<44} fp32 lats within 4.7e-6 rad of the pole: non-finite {int(nonfin.sum())}/41 "
              f"(+-inf: {int(np.isinf(x).sum())}); at the fp32 pole -> ({x[0]}, {y[0]}); max finite err "
              f"{np.nanmax(np.where(nonfin, np.nan, err)):.2f} m")
        if nonfin.any():
            finding("MEDIUM", "fp32", f"{d}: float32 input within ~3e-7 rad of the pole (incl. the pole itself) "
                    f"gives +-inf (t=tan^2(phi)~5e14 cubed overflows fp32 in approx_e_fwd); not NaN, not the "
                    f"pole value")
    # auto mode: the UNSELECTED approx branch only gets lam zeroed (tmerc.py:333), phi is passed through,
    # so in fp32 it overflows near the pole and 0*inf leaks NaN into d/dphi of exact-selected points
    A = tmerc.make("+proj=tmerc +ellps=WGS84 +algo=auto")
    t = torch.tensor([[0.3, float(v)] for v in vals[:6]], dtype=torch.float32, requires_grad=True)
    o = A(t)[:, :2]
    fin = torch.isfinite(o).all(-1)
    torch.where(fin[:, None], o, torch.zeros_like(o)).sum().backward()
    leak = (~torch.isfinite(t.grad).all(-1)) & fin
    print(f"  auto fp32, lam=0.3 rad (exact branch selected), 6 fp32 lats at/below the pole: outputs finite "
          f"{int(fin.sum())}/6, NaN gradients at finite outputs {int(leak.sum())}; grads {t.grad.tolist()}")
    if leak.any():
        finding("MEDIUM", "grad", "+algo=auto in float32: exact-selected points within ~3e-7 rad of the pole get NaN "
                "d/dphi from the unselected approx branch (double where sanitises lam only, not phi)")
    P = tmerc.make("+proj=tmerc +ellps=WGS84 +approx")
    yp = P(torch.tensor([[0.0, HP]], dtype=torch.float64))[0, 1].item()
    ys = np.array([yp - 5, yp - 1, yp - 0.5])
    l32, _ = mine(P, np.full(3, 100.0), ys, inverse=True, dtype=torch.float32)
    L, _ = ref("+proj=tmerc +ellps=WGS84 +approx", np.full(3, 100.0), ys, inverse=True)
    print(f"  approx inverse fp32 within 5 m of the pole northing: torch lon {l32.tolist()} PROJ lon {L.tolist()}")
    if np.isnan(l32).any():
        finding("LOW", "fp32", "approx inverse in float32 returns NaN within ~5 m of the pole northing "
                "(PROJ returns finite values)")


# =====================================================================================
# 6. Shapes, z/t passthrough, to_meter on z, dtypes
# =====================================================================================
def run_shapes():
    hdr("6. Batched shapes, z/t passthrough, to_meter on z, dtypes, non-contiguous input")
    P = tmerc.make("+proj=utm +zone=32 +ellps=WGS84")
    rng = np.random.default_rng(6)
    g = torch.tensor(np.stack([np.radians(rng.uniform(5, 13, (2, 3, 4))), np.radians(rng.uniform(-70, 70, (2, 3, 4)))],
                              -1))
    o = P(g)
    flat = P(g.reshape(-1, 2)).reshape(2, 3, 4, 2)
    ok_bhw = o.shape == (2, 3, 4, 2) and torch.equal(o, flat)
    oi = P.inverse(o)
    ok_bhw_inv = oi.shape == g.shape and float((oi - g).abs().max()) < 1e-14
    z = torch.tensor(rng.uniform(-100, 9000, 24), dtype=torch.float64)
    t = torch.tensor(rng.uniform(1990, 2030, 24), dtype=torch.float64)
    g3 = torch.cat([g.reshape(-1, 2), z[:, None]], -1)
    g4 = torch.cat([g3, t[:, None]], -1)
    o3, o4 = P(g3), P(g4)
    ok_z = torch.equal(o3[:, 2], z) and torch.equal(o4[:, 2], z) and torch.equal(o4[:, 3], t)
    ok_zi = torch.equal(P.inverse(o4)[:, 2:], g4[:, 2:])
    print(f"  [B,H,W,2] fwd == flattened: {ok_bhw}; inverse roundtrip <1e-14 rad: {ok_bhw_inv}; "
          f"[N,3]/[N,4] z,t passthrough fwd: {ok_z}, inv: {ok_zi}")
    # to_meter with z (fwd.cpp:163-165: z *= vfr_meter, vto_meter defaults to to_meter)
    d = "+proj=utm +zone=32 +ellps=WGS84 +to_meter=0.3048"
    Q = tmerc.make(d)
    lam, phi = g.reshape(-1, 2)[:, 0].numpy(), g.reshape(-1, 2)[:, 1].numpy()
    X, Y, Z = ref(d, lam, phi, z=z.numpy())
    oq = Q(g3).numpy()
    ez = np.abs(oq[:, 2] - Z).max()
    exy = np.hypot(oq[:, 0] - X, oq[:, 1] - Y).max()
    L, B, Z2 = ref(d, X, Y, z=Z, inverse=True)
    iq = Q.inverse(torch.tensor(np.stack([X, Y, Z], -1))).numpy()
    print(f"  to_meter=0.3048 3D: fwd xy err {exy:.1e} m, z err {ez:.1e}; inv z err {np.abs(iq[:, 2] - Z2).max():.1e}")
    if ez > 1e-9:
        finding("MEDIUM", "shapes", f"to_meter z handling differs from PROJ by {ez}")
    d2 = "+proj=utm +zone=32 +ellps=WGS84 +to_meter=0.3048 +vto_meter=1"
    try:
        Q2 = tmerc.make(d2)
        X, Y, Z = ref(d2, lam, phi, z=z.numpy())
        ez2 = np.abs(Q2(g3).numpy()[:, 2] - Z).max()
        print(f"  +vto_meter=1 with to_meter=0.3048: z err vs PROJ {ez2:.3g} m")
        if ez2 > 1e-9:
            finding("LOW", "shapes", f"+vto_meter / +vunits ignored: z is scaled by fr_meter anyway "
                    f"(z error {ez2:.0f} m for z up to 9 km)")
    except Exception as e:  # noqa: BLE001
        print("  +vto_meter:", e)
    d3 = "+proj=utm +zone=32 +ellps=WGS84 +z_0=100"
    try:
        X, Y, Z = ref(d3, lam, phi, z=z.numpy())
        ez3 = np.abs(tmerc.make(d3)(g3).numpy()[:, 2] - Z).max()
        print(f"  +z_0=100: z err vs PROJ {ez3:.3g} m")
        if ez3 > 1e-9:
            finding("LOW", "shapes", f"+z_0 ignored (fwd.cpp:165 adds z0): z error {ez3:.0f} m")
    except Exception as e:  # noqa: BLE001
        print("  +z_0:", e)
    # dtypes
    for dt in (torch.float32, torch.float64):
        for name in ("+proj=tmerc +ellps=WGS84", "+proj=tmerc +ellps=WGS84 +approx", "+proj=tmerc +R=6371000",
                     "+proj=tmerc +ellps=WGS84 +algo=auto", "+proj=utm +zone=32 +to_meter=2"):
            R = tmerc.make(name)
            inp = g.reshape(-1, 2).to(dt)
            out = R(inp)
            back = R.inverse(out)
            if out.dtype != dt or back.dtype != dt:
                finding("MEDIUM", "dtype", f"{name}: dtype {dt} -> {out.dtype}/{back.dtype}")
    print("  output dtype follows input for fp32/fp64 (fwd and inv), all kinds: checked")
    # non-contiguous and requires_grad leaf
    nc = g.reshape(-1, 2).t().contiguous().t()
    ok_nc = torch.equal(P(nc), P(g.reshape(-1, 2)))
    rg = g.reshape(-1, 2).clone().requires_grad_(True)
    og = P(rg)
    og.sum().backward()
    print(f"  non-contiguous input identical: {ok_nc}; requires_grad leaf -> grad finite: "
          f"{bool(torch.isfinite(rg.grad).all())}")
    # empty batch
    e0 = P(torch.zeros(0, 2, dtype=torch.float64))
    print(f"  empty [0,2] input -> {tuple(e0.shape)}")
    if not (ok_bhw and ok_bhw_inv and ok_z and ok_zi and ok_nc):
        finding("MEDIUM", "shapes", "shape/passthrough check failed")


# =====================================================================================
# 7. Gradients: FD vs autograd, own gradcheck, NaN leaks, singular bands
# =====================================================================================
def jac_autograd(fn, pts):
    """Per-point 2x2 Jacobians (points are independent), via two backward passes."""
    x = pts.clone().requires_grad_(True)
    out = fn(x)[..., :2]
    rows = []
    for k in range(2):
        g, = torch.autograd.grad(out[..., k].sum(), x, retain_graph=True)
        rows.append(g)
    return torch.stack(rows, -2)  # [N, out, in]


def jac_fd(fn, pts, h):
    cols = []
    for j in range(2):
        e = torch.zeros_like(pts)
        e[..., j] = h
        cols.append((fn(pts + e)[..., :2] - fn(pts - e)[..., :2]) / (2 * h))
    return torch.stack(cols, -1)


GRAD_SETS = [
    ("+proj=tmerc +ellps=WGS84", (-80, 80), (-88, 88)),
    ("+proj=tmerc +ellps=WGS84 +approx", (-20, 20), (-88, 88)),
    ("+proj=tmerc +ellps=WGS84 +algo=auto", (-10, 10), (-85, 85)),
    ("+proj=tmerc +R=6371000 +lat_0=30 +k_0=0.9", (-80, 80), (-85, 85)),
    ("+proj=utm +zone=32 +south +ellps=GRS80", (3, 15), (-84, -0.5)),
    ("+proj=tmerc +ellps=airy +lat_0=49 +lon_0=-2 +k_0=0.9996012717 +x_0=400000 +y_0=-100000", (-9, 3), (49, 61)),
    ("+proj=etmerc +ellps=intl +to_meter=0.3048", (-60, 60), (-80, 80)),
    ("+proj=tmerc +a=6378137 +rf=20", (-60, 60), (-80, 80)),
]


def run_grads():
    hdr("7a. Autograd Jacobian vs central finite differences (float64), 200 random interior points each")
    rng = np.random.default_rng(77)
    worst_all = 0.0
    for d, lo, la in GRAD_SETS:
        P = tmerc.make(d)
        lon = np.radians(rng.uniform(*lo, 200)) + P.lam0
        lat = np.radians(rng.uniform(*la, 200))
        lat = np.where(np.abs(lat) < 0.01, lat + 0.02, lat)
        pts = torch.tensor(np.stack([lon, lat], -1))
        f = lambda t: P(t) / P.a  # noqa: E731
        Ja, Jf = jac_autograd(f, pts), jac_fd(f, pts, 1e-6)
        rel = ((Ja - Jf).abs().amax((-1, -2)) / Jf.abs().amax((-1, -2))).numpy()
        xy = P(pts)[..., :2]
        g = lambda t: P.inverse(t * P.a)  # noqa: E731
        Ia, If = jac_autograd(g, xy / P.a), jac_fd(g, xy / P.a, 1e-7)
        reli = ((Ia - If).abs().amax((-1, -2)) / If.abs().amax((-1, -2))).numpy()
        # Jacobian of the inverse must be the matrix inverse of the forward one
        prod = torch.linalg.solve(Ja, torch.eye(2, dtype=torch.float64).expand_as(Ja))
        consist = ((prod - Ia * 1.0).abs().amax((-1, -2)) / Ia.abs().amax((-1, -2))).numpy()
        worst = max(np.nanmax(rel), np.nanmax(reli))
        worst_all = max(worst_all, worst)
        print(f"  {d[:70]:<70} fwd rel max {np.nanmax(rel):.1e} med {np.median(rel):.0e} | inv rel max "
              f"{np.nanmax(reli):.1e} | J_inv vs inv(J_fwd) {np.nanmax(consist):.1e} | nonfinite "
              f"{int((~torch.isfinite(Ja)).sum() + (~torch.isfinite(Ia)).sum())}")
        if worst > 1e-5 or not (torch.isfinite(Ja).all() and torch.isfinite(Ia).all()):
            finding("HIGH", "grad", f"{d}: autograd disagrees with FD (rel {worst:.1e})")

    # exact algorithm beyond 90 deg from the central meridian (up to the antimeridian)
    P = tmerc.make("+proj=tmerc +ellps=WGS84")
    lam = np.radians(rng.uniform(95, 179.9, 300)) * rng.choice([-1, 1], 300)
    phi = np.radians(rng.uniform(5, 85, 300)) * rng.choice([-1, 1], 300)
    pts = torch.tensor(np.stack([lam, phi], -1))
    f = lambda t: P(t) / P.a  # noqa: E731
    Ja, Jf = jac_autograd(f, pts), jac_fd(f, pts, 1e-6)
    rel = ((Ja - Jf).abs().amax((-1, -2)) / Jf.abs().amax((-1, -2))).numpy()
    xy = f(pts)[:, :2]
    g = lambda t: P.inverse(t * P.a)  # noqa: E731
    Ia, If = jac_autograd(g, xy), jac_fd(g, xy, 1e-7)
    reli = ((Ia - If).abs().amax((-1, -2)) / If.abs().amax((-1, -2))).numpy()
    print(f"  exact tmerc, 95..180 deg from the CM (300 pts): fwd rel max {np.nanmax(rel):.1e} inv rel max "
          f"{np.nanmax(reli):.1e}")

    hdr("7b. torch.autograd.gradcheck on new random points (eps 1e-6, atol 1e-8, rtol 1e-6)")
    for d, lo, la in GRAD_SETS:
        P = tmerc.make(d)
        lon = np.radians(rng.uniform(*lo, 10)) + P.lam0
        lat = np.radians(rng.uniform(*la, 10))
        lat = np.where(np.abs(lat) < 0.01, lat + 0.02, lat)
        lp = torch.tensor(np.stack([lon, lat], -1), requires_grad=True)
        okf = torch.autograd.gradcheck(lambda t: P(t) / P.a, (lp,), eps=1e-6, atol=1e-8, rtol=1e-6,
                                       raise_exception=False)
        xy = (P(lp.detach()) / P.a)[..., :2].clone().requires_grad_(True)
        oki = torch.autograd.gradcheck(lambda t: P.inverse(t * P.a), (xy,), eps=1e-7, atol=1e-8, rtol=1e-6,
                                       raise_exception=False)
        print(f"  {d[:70]:<70} fwd={'pass' if okf else 'FAIL'} inv={'pass' if oki else 'FAIL'}")
        if not (okf and oki):
            finding("HIGH", "grad", f"gradcheck fails for {d}")

    hdr("7c. NaN/inf gradient leaks: masked loss over finite outputs only")

    def leak(d, pts, dtype, inverse=False):
        P = tmerc.make(d)
        t = torch.tensor(pts).to(dtype).requires_grad_(True)
        o = (P.inverse(t) if inverse else P(t))[..., :2]
        fin = torch.isfinite(o).all(-1)
        torch.where(fin[:, None], o, torch.zeros_like(o)).sum().backward()
        bad = ~torch.isfinite(t.grad).all(-1)
        return int((~fin).sum()), int((bad & ~fin).sum()), int((bad & fin).sum()), t, o, bad
    lam = rng.choice([-1, 1], 20000) * rng.uniform(HP - 0.3, HP + 0.3, 20000)
    phi = rng.uniform(-0.3, 0.3, 20000)
    fwd_pts = np.stack([lam, phi], -1)
    glob = np.stack([rng.uniform(-math.pi, math.pi, 20000), rng.uniform(-HP, HP, 20000)], -1)
    inv_pts = np.stack([rng.uniform(-3.2, 3.2, 20000), rng.uniform(-3.5, 3.5, 20000)], -1) * 6378137.0
    eq_band = np.stack([rng.uniform(-3e6, 3e6, 4000), rng.uniform(-3000, 3000, 4000)], -1)
    extreme = np.array([[1e10, 1e5], [-1e10, 1e5], [5e9, 0.0], [-5e9, 0.0], [-5e9, 0.01], [1e200, 1e5],
                        [-1e200, 0.0], [1e5, 1e200], [1e5, -1e200], [0.0, 1e15]])
    for d in ["+proj=tmerc +ellps=WGS84", "+proj=tmerc +ellps=WGS84 +approx", "+proj=tmerc +ellps=WGS84 +algo=auto",
              "+proj=tmerc +R=6371000", "+proj=utm +zone=31 +ellps=WGS84"]:
        for dt in (torch.float64, torch.float32):
            for label, pts, inv in (("fwd near lam=90", fwd_pts, False), ("fwd global", glob, False),
                                    ("inv wide", inv_pts, True), ("inv |y|<3km", eq_band, True),
                                    ("inv extreme", extreme, True)):
                if dt == torch.float32 and label == "inv extreme":
                    continue
                nnan, leak_fail, leak_valid, t, o, bad = leak(d, pts, dt, inv)
                if leak_fail or leak_valid:
                    i = int(torch.where(bad)[0][0])
                    print(f"  {d[:44]:<44} {str(dt)[6:]:<8} {label:<16} NaN outputs {nnan:5d}; non-finite grad at "
                          f"failing pts {leak_fail}, at VALID pts {leak_valid}; e.g. in={t[i].tolist()} "
                          f"out={o[i].tolist()}")
                    sev = "MEDIUM" if (leak_valid and label != "inv extreme") else "LOW"
                    finding(sev, "grad", f"{d} {str(dt)[6:]} {label}: non-finite gradients at {leak_fail} failing "
                            f"and {leak_valid} valid points")
    print("  (combinations not printed had all-finite gradients)")

    hdr("7d. Singular bands, pole, auto switch")
    S = tmerc.make("+proj=tmerc +R=6371000")
    for dt in (torch.float64, torch.float32):
        ys = torch.tensor([0.0, 1e-3, 0.05, 0.067, 0.1, 1.0, 10.0, 100.0, 1000.0, 2000.0, 3000.0, 5000.0])
        pts = torch.stack([torch.full_like(ys, 2e5), ys], -1).to(dt)
        J = jac_autograd(lambda t: S.inverse(t), pts)
        nonf = ~torch.isfinite(J).all(-1).all(-1)
        print(f"  sphere inverse {str(dt)[6:]}: y (m) with non-finite Jacobian: {ys[nonf].tolist()}")
    for dt in (torch.float64, torch.float32):
        phis = torch.tensor([0.0, 5e-9, 1e-8, 2e-8, 1e-7, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 1e-2])
        pts = torch.stack([torch.full_like(phis, 0.2), phis], -1).to(dt)
        J = jac_autograd(lambda t: S(t) / S.a, pts)
        true = 1 / math.cos(0.2)
        relerr = ((J[:, 1, 1].double() - true).abs() / true).tolist()
        print(f"  sphere forward {str(dt)[6:]}: rel err of dy/dphi at phi={phis.tolist()}:\n      "
              + " ".join(f"{r:.0e}" for r in relerr))
    E = tmerc.make("+proj=tmerc +ellps=WGS84")
    for ph in (HP, HP - 1e-15, HP - 1e-9):
        J = jac_autograd(lambda t: E(t) / E.a, torch.tensor([[0.5, ph]], dtype=torch.float64))
        print(f"  exact fwd at phi=pi/2-{HP - ph:.0e}: dx/dphi={J[0, 0, 1]:.4f} dy/dphi={J[0, 1, 1]:.4f}")
    A = tmerc.make("+proj=tmerc +ellps=WGS84 +algo=auto")
    thr = 3 * tmerc.DEG_TO_RAD
    jumps = []
    for ph in np.radians([0.0, 20, 45, 70, 85]):
        p = torch.tensor([[np.nextafter(thr, 0), ph], [np.nextafter(thr, 1), ph]], dtype=torch.float64)
        v = A(p)
        J = jac_autograd(lambda t: A(t) / A.a, p)
        jumps.append((float((v[0] - v[1]).abs().max()), float(((J[0] - J[1]).abs().max() / J[0].abs().max()))))
    print(f"  auto fwd at |lam|=3deg: value jump (m) / Jacobian rel jump: "
          + ", ".join(f"{a:.1e}/{b:.1e}" for a, b in jumps))
    return worst_all


# =====================================================================================
# 8. GPU / torch.compile readiness: static scan + fullgraph with the eager backend
# =====================================================================================
def run_compile_scan():
    hdr("8. Host-sync scan of tmerc.py and torch.compile(fullgraph=True, backend='eager'/'aot_eager')")
    src = inspect.getsource(tmerc)
    pats = [r"\.item\(", r"\.numpy\(", r"\.tolist\(", r"\.cpu\(", r"bool\(", r"\.any\(\)", r"\.all\(\)",
            r"nonzero", r"masked_select", r"torch\.unique", r"\[.*bad.*\]"]
    for p in pats:
        hits = [ln.strip() for ln in src.splitlines() if re.search(p, ln) and not ln.strip().startswith("#")]
        if hits:
            print(f"  pattern {p!r}: {hits[:3]}")
    # Python `if` statements inside runtime methods
    for name in ("_fwd", "_inv", "_approx_fwd", "_approx_inv", "_sph_fwd", "_sph_inv", "_exact_fwd", "_exact_inv",
                 "forward", "inverse"):
        body = inspect.getsource(getattr(tmerc.TransverseMercator, name))
        ifs = [ln.strip() for ln in body.splitlines() if re.match(r"\s*(if|elif|while)\b", ln)]
        print(f"  {name:<12} python if/while: {ifs}")
    import torch._dynamo as dynamo
    rng = np.random.default_rng(8)
    lp = torch.tensor(np.radians(np.stack([rng.uniform(-20, 20, 64), rng.uniform(-80, 80, 64)], -1)))
    for d in ["+proj=utm +zone=32 +ellps=WGS84", "+proj=tmerc +ellps=WGS84 +algo=auto", "+proj=tmerc +R=6371000",
              "+proj=tmerc +ellps=WGS84 +approx", "+proj=utm +zone=32 +to_meter=0.3048"]:
        P = tmerc.make(d)
        msgs = []
        for backend in ("eager", "aot_eager"):
            for dyn in (False, True):
                dynamo.reset()
                try:
                    cf = torch.compile(P.forward, fullgraph=True, backend=backend, dynamic=dyn)
                    ci = torch.compile(P.inverse, fullgraph=True, backend=backend, dynamic=dyn)
                    x = lp.clone().requires_grad_(True)
                    xy = cf(x)
                    xy[..., 0].sum().backward()
                    back = ci(xy.detach())
                    ok = (torch.allclose(xy, P(lp), rtol=0, atol=1e-9)
                          and torch.allclose(back, P.inverse(P(lp)), rtol=0, atol=1e-15)
                          and bool(torch.isfinite(x.grad).all()))
                    msgs.append(f"{backend}/dyn={dyn}:{'ok' if ok else 'MISMATCH'}")
                except Exception as e:  # noqa: BLE001
                    msgs.append(f"{backend}/dyn={dyn}:ERROR {type(e).__name__} {str(e).splitlines()[0][:80]}")
                    finding("MEDIUM", "compile", f"{d}: torch.compile {backend} dynamic={dyn} fails: {e}")
        ex = dynamo.explain(P.forward)(lp)
        print(f"  {d:<44} {' '.join(msgs)}; graph breaks={ex.graph_break_count}")
    dynamo.reset()


def main():
    check_coeff_tables()
    blocks, cases, passed = run_gie()
    run_headline()
    run_masks()
    run_edges()
    run_setup()
    run_fp32_pole()
    run_shapes()
    run_grads()
    run_compile_scan()
    hdr(f"SUMMARY ({time.time() - T0:.0f}s): gie {passed}/{cases} in {blocks} blocks; {len(FIND)} findings")
    for sev, sec, msg in FIND:
        print(f"  [{sev}] ({sec}) {msg}")


if __name__ == "__main__":
    main()
