"""Test script for lcc_stere.py (no pytest needed).

  cd contrib/torchproj && python lcc_stere_test.py [--quick] [--compile]

Sections
  1. PROJ gie vectors (lcc, stere, ups): point cases, roundtrips, setup failures
  2. pyproj (PROJ 9.8.1) on 1e5 random points per definition, fp64 and fp32
  3. pj_phi2 / stere inverse: masked (faithful) vs fixed-N iterations
  4. autograd: gradcheck (fp64) at interior points, behaviour at singular points
  5. (optional --compile) torch.compile smoke test
Exits non-zero on failure.
"""
import math
import sys
import time

import numpy as np
import torch

import lcc_stere as L
import testing

torch.set_num_threads(1)
QUICK = "--quick" in sys.argv
NPTS = 20000 if QUICK else 100000
D2R = L.DEG_TO_RAD
FAILURES = []


def fail(msg):
    FAILURES.append(msg)
    print("  FAIL:", msg)


def t64(a):
    return torch.as_tensor(np.asarray(a, dtype=np.float64))


# =============================================================================
# 1. gie
# =============================================================================
def gie_setup_failures(name):
    """Blocks of the form `operation ...` + `expect failure` with no accept in the
    block. Mirrors gie.cpp: if proj_create succeeds, gie transforms the *stale*
    T.a (the last accept seen in the file, in the block's direction=forward)
    and expects HUGE_VAL (gie.cpp:1051-1075)."""
    out = []
    for f in sorted(testing.GIE_DIR.glob("*.gie")):
        op, last_accept, seen_accept = None, None, False
        for cmd, args in testing._logical_lines(f):
            if cmd == "operation":
                op = testing._clean_operation(args)
                seen_accept = False
            elif cmd == "accept":
                try:
                    last_accept = [testing.parse_num(v) for v in args.split()[:4]]
                except ValueError:
                    pass
                seen_accept = True
            elif cmd == "expect" and args.strip().startswith("failure") and not seen_accept:
                toks = op.replace("+", " ").split()
                if f"proj={name}" in toks and "proj=pipeline" not in toks:
                    out.append({"operation": op, "file": f.name, "stale_accept": last_accept})
    return out


def run_gie():
    print("=" * 78)
    print("1. gie vectors")
    stats = {"blocks": 0, "total": 0, "passed": 0, "skipped": 0, "failures": []}
    worst = {}
    for name in ("lcc", "stere", "ups"):
        blocks = testing.gie_cases(name)
        setup_fail = {b["operation"]: b for b in gie_setup_failures(name)}
        stats["blocks"] += len(blocks)
        for b in blocks:
            op = b["operation"]
            try:
                P = L.make(op)
            except L.ProjSetupError as ex:
                P, err = None, ex
            if op in setup_fail and not b["cases"]:
                stats["total"] += 1
                ok = P is None
                how = "setup raised: %s" % (err if P is None else "")
                if P is not None:
                    a = setup_fail[op]["stale_accept"] or [0, 0]
                    out = P.forward(t64([a[:2]]) * D2R)
                    ok = bool(torch.isnan(out[0, 0]))
                    how = "setup ok in PROJ 9.8.1 too; stale accept %s -> %s" % (a[:2], out[0, :2].tolist())
                if ok:
                    stats["passed"] += 1
                else:
                    stats["failures"].append("%s: expected failure, got %s" % (op, how))
                print("  [%s] %-60s expect failure: %s" % ("ok" if ok else "FAIL", op[:60], how[:70]))
                continue
            if P is None:
                n = len(b["cases"]) + len(b["roundtrips"])
                stats["total"] += n
                stats["failures"].append("%s: unexpected setup failure %s" % (op, err))
                continue
            for c in b["cases"]:
                stats["total"] += 1
                a = t64([c["accept"][:2]])
                if c["direction"] == "fwd":
                    out = P.forward(a * D2R)[0].numpy()
                    if c["expect"] == "failure":
                        ok = bool(np.isnan(out[0]))
                        e = 0.0
                    else:
                        e = float(np.hypot(*(out[:2] - np.array(c["expect"][:2]))))
                        ok = e <= c["tolerance_m"]
                else:
                    out = P.inverse(a)[0].numpy() / D2R
                    if c["expect"] == "failure":
                        ok = bool(np.isnan(out[0]))
                        e = 0.0
                    else:
                        e = float(testing.angular_error_m(out[0], out[1], *c["expect"][:2], a=P.a))
                        ok = e <= c["tolerance_m"]
                worst[op] = max(worst.get(op, 0.0), e)
                if ok:
                    stats["passed"] += 1
                else:
                    stats["failures"].append("%s %s %s: err %.3g m > tol %.3g" % (
                        op, c["direction"], c["accept"][:2], e, c["tolerance_m"]))
            for r in b["roundtrips"]:
                stats["total"] += 1
                a = t64([r["accept"][:2]]) * D2R
                cur = P.forward(a)
                for _ in range(r["n"] - 1):
                    cur = P.forward(P.inverse(cur))
                back = P.inverse(cur)[0].numpy() / D2R
                e = float(testing.angular_error_m(back[0], back[1], *r["accept"][:2], a=P.a))
                if e <= r["tolerance_m"]:
                    stats["passed"] += 1
                else:
                    stats["failures"].append("%s roundtrip %s: %.3g m > %.3g" % (op, r["accept"][:2], e, r["tolerance_m"]))
            print("  [%s] %-60s %2d cases %d roundtrips, worst %.2e m" % (
                "ok", op[:60], len(b["cases"]), len(b["roundtrips"]), worst.get(op, 0.0)))
    print("  gie: %d/%d passed (%d blocks, %d skipped)" % (
        stats["passed"], stats["total"], stats["blocks"], stats["skipped"]))
    for f in stats["failures"]:
        fail("gie " + f)
    return stats


# =============================================================================
# 2. pyproj comparison
# =============================================================================
DEFS = [
    # (definition, lon range, lat range)
    ("+proj=lcc +lat_1=30 +lat_2=60 +lat_0=45 +lon_0=10 +R=6370000", (-40, 60), (15, 80)),
    ("+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +ellps=WGS84", (-140, -50), (10, 65)),
    ("+proj=lcc +lat_1=-30 +lat_2=-60 +lat_0=-45 +lon_0=140 +x_0=500000 +y_0=1000000 +ellps=WGS84",
     (100, 180), (-80, -10)),
    ("+proj=lcc +lat_1=46.8 +lat_0=46.8 +lon_0=2.337229167 +k_0=0.99987742 +x_0=600000 +y_0=2200000 +ellps=clrk80ign",
     (-6, 10), (41, 52)),
    ("+proj=lcc +ellps=clrk66 +lat_1=44d11'N +lat_2=45d42'N +x_0=609601.2192 +lon_0=84d20'W "
     "+lat_0=43d19'N +k_0=1.0000382 +units=us-ft", (-91, -82), (41, 48.5)),
    ("+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +ellps=WGS84", (-180, 180), (30, 90)),
    ("+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +ellps=WGS84", (-180, 180), (-90, -40)),
    ("+proj=stere +lat_0=90 +lon_0=0 +k=0.994 +x_0=2000000 +y_0=2000000 +ellps=WGS84", (-180, 180), (50, 90)),
    ("+proj=ups +ellps=WGS84", (-180, 180), (60, 90)),
    ("+proj=ups +south +ellps=WGS84", (-180, 180), (-90, -60)),
    ("+proj=stere +lat_0=45 +lon_0=10 +k=0.9999 +x_0=100000 +y_0=200000 +ellps=WGS84", (-30, 50), (10, 80)),
    ("+proj=stere +lat_0=0 +lon_0=0 +ellps=GRS80", (-60, 60), (-60, 60)),
    ("+proj=stere +lat_0=60 +lon_0=-100 +R=6371229", (-150, -50), (20, 89)),
    ("+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-105 +R=6371229", (-180, 180), (0, 90)),
    ("+proj=stere +lat_0=0 +lon_0=30 +R=6371000", (-30, 90), (-60, 60)),
]


def compare_pyproj():
    print("=" * 78)
    print("2. pyproj comparison (%d random points per definition)" % NPTS)
    print("   errors in metres; fp32 = inputs cast to float32 vs the float64 PROJ reference")
    rng = np.random.default_rng(12345)
    rows = []
    hdr = "  %-58s %9s %9s %9s %9s %s" % ("definition", "fwd64", "inv64", "fwd32", "inv32", "nan-mismatch")
    print(hdr)
    for defn, (lo0, lo1), (la0, la1) in DEFS:
        P = L.make(defn)
        lon = rng.uniform(lo0, lo1, NPTS)
        lat = rng.uniform(la0, la1, NPTS)
        X, Y = testing.pyproj_ref(defn, lon, lat)
        okref = np.isfinite(X) & np.isfinite(Y)
        rlon, rlat = testing.pyproj_ref(defn, X[okref], Y[okref], inverse=True)
        res = {"definition": defn, "n_points": NPTS}
        mism = 0
        for dt, tag in ((torch.float64, "64"), (torch.float32, "32")):
            inp = torch.stack([t64(lon) * D2R, t64(lat) * D2R], -1).to(dt)
            with torch.no_grad():
                out = P.forward(inp).double().numpy()
            ours_ok = np.isfinite(out[:, 0]) & np.isfinite(out[:, 1])
            if tag == "64":
                mism += int((ours_ok != okref).sum())
            both = ours_ok & okref
            ef = np.hypot(out[both, 0] - X[both], out[both, 1] - Y[both]) * P.to_meter
            res["fwd_max_err_fp%s_m" % tag] = float(ef.max())
            xin = torch.stack([t64(X[okref]), t64(Y[okref])], -1).to(dt)
            with torch.no_grad():
                back = P.inverse(xin).double().numpy() / D2R
            bok = np.isfinite(back[:, 0]) & np.isfinite(back[:, 1])
            if tag == "64":
                mism += int((~bok).sum())
            ei = testing.angular_error_m(back[bok, 0], back[bok, 1], rlon[bok], rlat[bok], a=P.a)
            res["inv_max_err_fp%s_m" % tag] = float(ei.max())
            if tag == "32":
                # floor: float64 arithmetic on float32-rounded inputs
                with torch.no_grad():
                    oq = P.forward(inp.double()).numpy()
                    bq = P.inverse(xin.double()).numpy() / D2R
                okq = np.isfinite(oq[:, 0]) & okref
                res["fp32_fwd_input_floor_m"] = float((np.hypot(oq[okq, 0] - X[okq], oq[okq, 1] - Y[okq]) * P.to_meter).max())
                bqk = np.isfinite(bq[:, 0])
                res["fp32_inv_input_floor_m"] = float(testing.angular_error_m(
                    bq[bqk, 0], bq[bqk, 1], rlon[bqk], rlat[bqk], a=P.a).max())
                res["fp32_inv_nan"] = int((~bok).sum())
                res["fp32_fwd_nan"] = int((~ours_ok & okref).sum())
                # where is the fp32 inverse worst?
                k = int(np.argmax(ei))
                res["fp32_inv_worst_at"] = (float(rlon[bok][k]), float(rlat[bok][k]))
                res["fp32_inv_p99_m"] = float(np.percentile(ei, 99))
                res["fp32_fwd_p99_m"] = float(np.percentile(ef, 99))
        res["nan_mismatch_fp64"] = mism
        res["ref_failures"] = int((~okref).sum())
        rows.append(res)
        print("  %-58s %9.2e %9.2e %9.2e %9.2e %d" % (
            defn[:58], res["fwd_max_err_fp64_m"], res["inv_max_err_fp64_m"],
            res["fwd_max_err_fp32_m"], res["inv_max_err_fp32_m"], mism))
        if res["fwd_max_err_fp64_m"] > 1e-6 or res["inv_max_err_fp64_m"] > 1e-6 or mism:
            fail("pyproj fp64 mismatch for %s: %s" % (defn, res))
    print("  fp32 detail: p99 fwd/inv; input-rounding floor fwd/inv (fp64 maths on fp32-rounded")
    print("  inputs); NaN counts; location of worst inverse error")
    for r in rows:
        print("    %-50s p99 %.1e/%.1e floor %.1e/%.1e nan %d/%d worst inv at %.2f,%.4f" % (
            r["definition"][:50], r["fp32_fwd_p99_m"], r["fp32_inv_p99_m"], r["fp32_fwd_input_floor_m"],
            r["fp32_inv_input_floor_m"], r["fp32_fwd_nan"], r["fp32_inv_nan"], *r["fp32_inv_worst_at"]))
    return rows


def special_points():
    """Edge points vs PROJ: poles, antimeridian, projection centre, antipode."""
    print("=" * 78)
    print("2b. special points vs PROJ (fp64)")
    cases = [
        ("+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +ellps=WGS84",
         [(-96, 90), (-96, -90), (84, 10), (-276 + 1e-9, 10), (-96, 39), (0, 89.9999999), (-96, -89.99)]),
        ("+proj=lcc +lat_1=-30 +lat_2=-60 +lat_0=-45 +lon_0=140 +ellps=WGS84",
         [(140, -90), (140, 90), (-40, -10), (140, -45)]),
        ("+proj=lcc +lat_1=30 +lat_2=60 +lat_0=45 +lon_0=10 +R=6370000",
         [(10, 90), (10, -90), (-170, 30), (10, 45)]),
        ("+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +ellps=WGS84",
         [(0, 90), (-45, 90), (135, 89.9999999), (0, -90), (0, -89), (180, 0)]),
        ("+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +ellps=WGS84", [(0, -90), (10, 90), (90, 0)]),
        ("+proj=stere +lat_0=45 +lon_0=10 +k=0.9999 +ellps=WGS84", [(10, 45), (-170, -45), (-170, -44.9), (10, 90)]),
        ("+proj=stere +lat_0=0 +lon_0=0 +ellps=GRS80", [(0, 0), (180, 0), (179.999, 0), (0, 90), (0, -90)]),
        ("+proj=stere +lat_0=0 +lon_0=0 +R=6400000", [(0, 0), (180, 0), (179.99, 0), (0, 90)]),
        ("+proj=stere +lat_0=60 +lon_0=-100 +R=6371229", [(-100, 60), (80, -60), (80, -59.99), (0, 90)]),
        ("+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-105 +R=6371229", [(0, 90), (0, -90), (0, -89.99), (75, 89.9999)]),
        ("+proj=ups +ellps=WGS84", [(0, 90), (0, -90), (45, 84)]),
    ]
    nbad = 0
    for defn, pts in cases:
        P = L.make(defn)
        lon = np.array([p[0] for p in pts], float)
        lat = np.array([p[1] for p in pts], float)
        X, Y = testing.pyproj_ref(defn, lon, lat)
        with torch.no_grad():
            out = P.forward(torch.stack([t64(lon) * D2R, t64(lat) * D2R], -1)).numpy()
        for i in range(len(pts)):
            ref_bad = not (np.isfinite(X[i]) and np.isfinite(Y[i]))
            our_bad = not (np.isfinite(out[i, 0]) and np.isfinite(out[i, 1]))
            if ref_bad or our_bad:
                same = ref_bad and our_bad
                note = "both fail" if same else "PROJ (%g, %g) ours (%g, %g)" % (X[i], Y[i], out[i, 0], out[i, 1])
                # documented divergence: ellipsoidal EQUIT antipode -> PROJ (0, HUGE_VAL), ours NaN
                if not same and np.isfinite(X[i]) and X[i] == 0 and np.isinf(Y[i]) and np.isnan(out[i, 1]):
                    same, note = True, "PROJ returns (0, inf) [HUGE_VAL y]; port returns NaN (documented)"
            else:
                d = float(np.hypot(out[i, 0] - X[i], out[i, 1] - Y[i]))
                same = d <= 1e-6 * max(1.0, abs(X[i]), abs(Y[i])) * 1e-3 or d < 1e-6
                note = "fwd diff %.2e m" % d
            # inverse of PROJ's output
            inv_note = ""
            if np.isfinite(X[i]) and np.isfinite(Y[i]):
                rl, rp = testing.pyproj_ref(defn, X[i:i + 1], Y[i:i + 1], inverse=True)
                with torch.no_grad():
                    b = P.inverse(t64([[X[i], Y[i]]]))[0].numpy() / D2R
                if np.isfinite(rl[0]):
                    ei = float(testing.angular_error_m(b[0], b[1], rl[0], rp[0]))
                    inv_note = " inv diff %.2e m" % ei
                    if not ei < 1e-6:
                        same = False
            if not same:
                nbad += 1
            print("  [%s] %-50s (%g, %g): %s%s" % ("ok" if same else "DIFF", defn[:50], pts[i][0], pts[i][1], note, inv_note))
    if nbad:
        fail("%d special points differ from PROJ" % nbad)


# =============================================================================
# 3. iterations: masked (faithful) vs fixed-N
# =============================================================================
def bench(f, reps=3):
    f()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        t.append(time.perf_counter() - t0)
    return min(t)


def iterations():
    print("=" * 78)
    print("3. iterative parts: masked loop with PROJ cap vs fixed-N unrolled")
    out = {}
    # --- pj_phi2 iteration distribution over all latitudes, WGS84
    e = math.sqrt(0.0066943799901413165)
    phi = torch.linspace(-89.999, 89.999, 200001, dtype=torch.float64) * D2R
    ts = L.tsfn(phi, torch.sin(phi), e)
    st = {}
    ref, f0 = L.phi2(ts, e, "masked", stats=st)
    it = st["iters"].numpy()
    hist = {int(k): int((it == k).sum()) for k in np.unique(it)}
    print("  pj_phi2 Newton iterations PROJ performs (WGS84, |lat|<89.999): %s" % hist)
    out["phi2_iter_hist"] = hist
    a = 6378137.0
    for n in (1, 2, 3):
        fx, _ = L.phi2(ts, e, "fixed", n)
        d = float((fx - ref).abs().max()) * a
        dt = float((fx - phi).abs().max()) * a
        out["phi2_fixed%d_vs_masked_m" % n] = d
        print("    fixed N=%d: max |phi - phi_masked| = %.2e m, max |phi - phi_true| = %.2e m (masked: %.2e m)" % (
            n, d, dt, float((ref - phi).abs().max()) * a))
    fx32, f32 = L.phi2(ts.float(), e, "masked")
    orig_eps = L._eps
    try:  # test-only: PROJ's double tolerances applied to float32
        L._eps = lambda dt: torch.finfo(torch.float64).eps
        _, f32p = L.phi2(ts.float(), e, "masked")
    finally:
        L._eps = orig_eps
    out["phi2_fp32_fail_frac_proj_tol"] = float(f32p.double().mean())
    print("    float32 masked: dtype-aware tol max err %.2e m, %d failures; with PROJ's float64 tol"
          " %d/%d points (%.2f%%) flagged as failures" % (
              float((fx32.double() - phi).abs().max()) * a, int(f32.sum()), int(f32p.sum()), len(ts),
              100 * float(f32p.double().mean())))
    # --- lcc inverse accuracy vs pyproj and vs masked, and timing
    defn = "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +ellps=WGS84"
    rng = np.random.default_rng(7)
    lon = rng.uniform(-140, -50, NPTS)
    lat = rng.uniform(10, 65, NPTS)
    X, Y = testing.pyproj_ref(defn, lon, lat)
    rlon, rlat = testing.pyproj_ref(defn, X, Y, inverse=True)
    xin = torch.stack([t64(X), t64(Y)], -1)
    Pm = L.make(defn)
    with torch.no_grad():
        bm = Pm.inverse(xin)
    print("  lcc inverse (%d pts, CONUS WGS84), single thread:" % NPTS)
    print("    %-22s %12s %14s %10s" % ("variant", "max err m", "max |d| vs mask", "time ms"))
    variants = [("masked (cap 5)", dict())] + [("fixed N=%d" % n, dict(iter_mode="fixed", n_iter=n)) for n in (1, 2, 3)]
    for label, kw in variants:
        P = L.make(defn, **kw)
        with torch.no_grad():
            b = P.inverse(xin)
            tm = bench(lambda: P.inverse(xin)) * 1e3
        bb = b.numpy() / D2R
        err = testing.angular_error_m(bb[:, 0], bb[:, 1], rlon, rlat).max()
        dm = testing.angular_error_m(bb[:, 0], bb[:, 1], bm.numpy()[:, 0] / D2R, bm.numpy()[:, 1] / D2R).max()
        out["lcc_inv_" + label] = (float(err), float(dm), tm)
        print("    %-22s %12.3e %14.3e %10.1f" % (label, err, dm, tm))
    # --- stere ellipsoidal inverse (polar and oblique)
    for defn, (lo0, lo1), (la0, la1) in [
            ("+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +ellps=WGS84", (-180, 180), (30, 90)),
            ("+proj=stere +lat_0=45 +lon_0=10 +k=0.9999 +x_0=100000 +y_0=200000 +ellps=WGS84", (-30, 50), (10, 80))]:
        lon = rng.uniform(lo0, lo1, NPTS)
        lat = rng.uniform(la0, la1, NPTS)
        X, Y = testing.pyproj_ref(defn, lon, lat)
        rlon, rlat = testing.pyproj_ref(defn, X, Y, inverse=True)
        xin = torch.stack([t64(X), t64(Y)], -1)
        Pm = L.make(defn)
        Pm.stats = {}
        with torch.no_grad():
            bm = Pm.inverse(xin)
        it = Pm.stats["iters"].numpy()
        hist = {int(k): int((it == k).sum()) for k in np.unique(it)}
        print("  stere inverse %s: PROJ iteration counts %s" % (defn[:45], hist))
        out["stere_iter_hist " + defn] = hist
        print("    %-22s %12s %14s %10s" % ("variant", "max err m", "max |d| vs mask", "time ms"))
        for label, kw in [("masked (cap 8)", dict())] + [
                ("fixed N=%d" % n, dict(iter_mode="fixed", n_iter=n)) for n in (3, 4, 5, 6)]:
            P = L.make(defn, **kw)
            with torch.no_grad():
                b = P.inverse(xin)
                tm = bench(lambda: P.inverse(xin)) * 1e3
            bb = b.numpy() / D2R
            err = testing.angular_error_m(bb[:, 0], bb[:, 1], rlon, rlat).max()
            dm = testing.angular_error_m(bb[:, 0], bb[:, 1], bm.numpy()[:, 0] / D2R, bm.numpy()[:, 1] / D2R).max()
            with torch.no_grad():
                rt = P.forward(b) - xin
            res_m = float(torch.hypot(rt[:, 0], rt[:, 1]).max())
            out["stere_inv_%s %s" % (label, defn[:20])] = (float(err), float(dm), tm, res_m)
            print("    %-22s %12.3e %14.3e %10.1f   fwd(inv(xy))-xy %.2e m" % (label, err, dm, tm, res_m))
    # forward timing reference
    P = L.make("+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +ellps=WGS84")
    inp = torch.stack([t64(rng.uniform(-140, -50, NPTS)) * D2R, t64(rng.uniform(10, 65, NPTS)) * D2R], -1)
    with torch.no_grad():
        tf = bench(lambda: P.forward(inp)) * 1e3
    print("  (lcc forward, same size, for scale: %.1f ms)" % tf)
    out["lcc_fwd_ms"] = tf
    return out


# =============================================================================
# 4. autograd
# =============================================================================
GRAD_DEFS = [
    ("+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +ellps=WGS84", [(-100, 35), (-80, 50), (-120, 20)]),
    ("+proj=lcc +lat_1=30 +lat_2=60 +lat_0=45 +lon_0=10 +R=6370000", [(0, 30), (30, 70), (10, 45)]),
    ("+proj=lcc +lat_1=-30 +lat_2=-60 +lat_0=-45 +lon_0=140 +ellps=WGS84", [(130, -30), (160, -70)]),
    ("+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +ellps=WGS84", [(0, 60), (100, 85), (-170, 40)]),
    ("+proj=stere +lat_0=45 +lon_0=10 +k=0.9999 +ellps=WGS84", [(0, 30), (30, 70), (11, 46)]),
    ("+proj=stere +lat_0=0 +lon_0=0 +ellps=GRS80", [(10, 10), (-40, -30)]),
    ("+proj=stere +lat_0=60 +lon_0=-100 +R=6371229", [(-90, 50), (-130, 80)]),
    ("+proj=stere +lat_0=0 +lon_0=30 +R=6371000", [(40, 10), (0, -30)]),
    ("+proj=stere +lat_0=-90 +lat_ts=-60 +lon_0=0 +R=6371229", [(20, -60), (-150, -80)]),
    ("+proj=ups +ellps=WGS84", [(20, 70), (-120, 88)]),
]


def run_gradcheck():
    print("=" * 78)
    print("4a. torch.autograd.gradcheck (float64, interior points); x,y scaled to units of a")
    res = []
    for defn, pts in GRAD_DEFS:
        P = L.make(defn)
        lonlat = torch.tensor([[p[0] * D2R, p[1] * D2R] for p in pts], dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            xy = P.forward(lonlat).clone()
        xy.requires_grad_(True)
        f_fwd = lambda c: P.forward(c) / P.a  # noqa: E731  (keep Jacobian O(1))
        f_inv = lambda c: P.inverse(c)  # noqa: E731
        ok_f = torch.autograd.gradcheck(f_fwd, (lonlat,), eps=1e-7, atol=1e-6, rtol=1e-5, raise_exception=False)
        # inverse: perturb metres with eps = 1e-3 m
        ok_i = torch.autograd.gradcheck(f_inv, (xy,), eps=1e-3, atol=1e-10, rtol=1e-4, raise_exception=False)
        # also check the analytical inverse Jacobian is the inverse of the forward Jacobian
        Jf = torch.autograd.functional.jacobian(lambda c: P.forward(c[None])[0], lonlat[0].detach())
        Ji = torch.autograd.functional.jacobian(lambda c: P.inverse(c[None])[0], xy[0].detach())
        prod_err = float((Ji @ Jf - torch.eye(2, dtype=torch.float64)).abs().max())
        res.append((defn, ok_f, ok_i, prod_err))
        print("  %-62s fwd %-5s inv %-5s |Ji*Jf - I| = %.1e" % (defn[:62], ok_f, ok_i, prod_err))
        if not (ok_f and ok_i) or prod_err > 1e-6:
            fail("gradcheck %s fwd=%s inv=%s prod=%.2e" % (defn, ok_f, ok_i, prod_err))
    # phi2 in both modes
    e = 0.0818191908426215
    ts = torch.tensor([0.3, 0.9, 1.5, 3.0], dtype=torch.float64, requires_grad=True)
    ok_m = torch.autograd.gradcheck(lambda t: L.phi2(t, e, "masked")[0], (ts,), raise_exception=False)
    ok_x = torch.autograd.gradcheck(lambda t: L.phi2(t, e, "fixed", 2)[0], (ts,), raise_exception=False)
    print("  pj_phi2 gradcheck: masked %s, fixed-2 %s" % (ok_m, ok_x))
    if not (ok_m and ok_x):
        fail("phi2 gradcheck")
    return res


def grad_of(fn, x):
    x = x.clone().requires_grad_(True)
    y = fn(x)
    g = []
    for k in range(2):
        gk, = torch.autograd.grad(y[..., k].sum(), x, retain_graph=True, allow_unused=True)
        g.append(gk)
    return y.detach(), torch.stack(g, -2)  # [..., out, in]


def singular_points():
    print("=" * 78)
    print("4b. gradients at singular points (rows: d out_k / d in, in metres or radians)")
    notes = []

    def show(label, y, J, expect=None):
        finite = bool(torch.isfinite(J).all())
        print("  %-64s out=%s\n      J=%s finite=%s" % (label, np.array2string(y.numpy().ravel(), precision=6),
                                                    np.array2string(J.numpy().reshape(-1), precision=4), finite))
        notes.append((label, y.numpy().ravel().tolist(), J.numpy().reshape(-1).tolist(), finite))
        if expect is not None:
            err = float((J - expect).abs().max() / max(1.0, float(expect.abs().max())))
            print("      vs expected (limit / FD): rel diff %.1e" % err)
            if err > 1e-4:
                fail(label + " gradient differs from limit: %.2e" % err)
        return finite

    def rad(lo, la):
        return torch.tensor([[lo * D2R, la * D2R]], dtype=torch.float64)

    # LCC apex (n > 0): north pole
    P = L.make("+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +ellps=WGS84")
    y, J = grad_of(P.forward, rad(-90, 90))
    show("lcc fwd at apex pole (90N): value exact, true dy/dphi = inf", y, J)
    y, J = grad_of(P.forward, rad(-90, 90 - 1e-6))
    show("lcc fwd 1e-6 deg from apex (shows the blow-up)", y, J)
    y, J = grad_of(P.forward, rad(-90, -90))
    show("lcc fwd at the excluded pole (90S): NaN output", y, J)
    y, J = grad_of(P.forward, rad(84, 40))
    show("lcc fwd on the antimeridian cut (lon_0+180)", y, J)
    apex = P.forward(rad(0, 90))
    y, J = grad_of(P.inverse, apex)
    show("lcc inv at the apex (x,y)=(0,rho0): constant branch", y, J)
    y, J = grad_of(P.forward, rad(-90, 95))
    show("lcc fwd out of domain (lat 95): NaN, grad", y, J)
    # Stereographic polar ellipsoidal: forward at the pole
    P = L.make("+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +ellps=WGS84")
    lam = 30.0
    y, J = grad_of(P.forward, rad(lam, 90))
    _, Jn = grad_of(P.forward, rad(lam, 90 - 1e-9))
    show("stere polar fwd at the pole (straight-through grad)", y, J, Jn)
    Praw = L.make("+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +ellps=WGS84", smooth_grad=False)
    y, J = grad_of(Praw.forward, rad(lam, 90))
    show("  same with smooth_grad=False (PROJ branch only): zero grad", y, J)
    y, J = grad_of(P.inverse, torch.zeros(1, 2, dtype=torch.float64))
    show("stere polar inv at the pole (cone point, lam undefined)", y, J)
    # Oblique ellipsoidal inverse at the centre
    P = L.make("+proj=stere +lat_0=45 +lon_0=10 +k=0.9999 +x_0=100000 +y_0=200000 +ellps=WGS84")
    c = torch.tensor([[100000.0, 200000.0]], dtype=torch.float64)
    y, J = grad_of(P.inverse, c)
    _, Jn = grad_of(P.inverse, c + torch.tensor([[1e-3, 1e-3]], dtype=torch.float64))
    show("stere oblique ell inv at the centre (smooth formula)", y, J, Jn)
    Praw = L.make("+proj=stere +lat_0=45 +lon_0=10 +k=0.9999 +x_0=100000 +y_0=200000 +ellps=WGS84",
                  smooth_grad=False)
    y, J = grad_of(Praw.inverse, c)
    show("  same with smooth_grad=False: wrong (zero) Jacobian", y, J)
    y, J = grad_of(P.forward, rad(-170, -45))
    show("stere oblique ell fwd at the antipode", y, J)
    # Spherical equatorial inverse at the centre
    P = L.make("+proj=stere +lat_0=0 +lon_0=30 +R=6371000")
    y, J = grad_of(P.inverse, torch.zeros(1, 2, dtype=torch.float64))
    _, Jn = grad_of(P.inverse, torch.tensor([[1e-3, 1e-3]], dtype=torch.float64))
    show("stere equatorial sph inv at the centre", y, J, Jn)
    P = L.make("+proj=stere +lat_0=60 +lon_0=-100 +R=6371229")
    y, J = grad_of(P.inverse, torch.zeros(1, 2, dtype=torch.float64))
    _, Jn = grad_of(P.inverse, torch.tensor([[1e-3, 1e-3]], dtype=torch.float64))
    show("stere oblique sph inv at the centre", y, J, Jn)
    P = L.make("+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-105 +R=6371229")
    y, J = grad_of(P.forward, rad(0, -90))
    show("stere polar sph fwd at the opposite pole (NaN, TOL check)", y, J)
    xy = torch.tensor([[1e-4, 2e-4]], dtype=torch.float64)
    y, J = grad_of(P.inverse, xy)
    show("stere polar sph inv 0.2 mm from pole (rh <= EPS10 branch)", y, J)
    x_, y_ = 0.6, 0.8  # 1 m from the pole: asin(cos c) has cos c = 1 - 1e-14
    rh, A = 1.0, P.a * P.akm1
    dphi = -2.0 / A / (1 + (rh / A) ** 2)
    Jexp = torch.tensor([[[y_ / rh ** 2 * -1, x_ / rh ** 2], [dphi * x_ / rh, dphi * y_ / rh]]], dtype=torch.float64)
    y, J = grad_of(P.inverse, torch.tensor([[x_, y_]], dtype=torch.float64))
    show("stere polar sph inv 1 m from pole (straight-through pi/2-c)", y, J, Jexp)
    Praw = L.make("+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-105 +R=6371229", smooth_grad=False)
    y, J = grad_of(Praw.inverse, torch.tensor([[x_, y_]], dtype=torch.float64))
    show("  same with smooth_grad=False (d asin(cos c), ill-conditioned)", y, J)

    # Batch hygiene: mixed batch incl. singular/failed points; masked sum loss
    print("  batch hygiene: grads of sum(finite outputs) over batches with singular points")
    rng = np.random.default_rng(3)
    allfinite = True
    for defn, _ in GRAD_DEFS:
        P = L.make(defn)
        lon = np.concatenate([rng.uniform(-180, 180, 200), [0, 0, 180, -180, 0, 90]])
        lat = np.concatenate([rng.uniform(-90, 90, 200), [90, -90, 0, 0, 0, 95]])
        lonlat = torch.stack([t64(lon) * D2R, t64(lat) * D2R], -1).requires_grad_(True)
        out = P.forward(lonlat)
        m = torch.isfinite(out)
        loss = torch.where(m, out, torch.zeros_like(out)).sum() / P.a
        g, = torch.autograd.grad(loss, lonlat)
        fin_f = bool(torch.isfinite(g).all())
        xy = torch.where(m, out, torch.zeros_like(out)).detach()
        xy = torch.cat([xy, torch.tensor([[P.x0, P.y0], [0.0, 0.0]], dtype=torch.float64)]).requires_grad_(True)
        back = P.inverse(xy)
        mb = torch.isfinite(back)
        g2, = torch.autograd.grad(torch.where(mb, back, torch.zeros_like(back)).sum(), xy)
        fin_i = bool(torch.isfinite(g2).all())
        allfinite &= fin_f and fin_i
        print("    %-62s fwd grads finite %s, inv grads finite %s" % (defn[:62], fin_f, fin_i))
    if not allfinite:
        fail("non-finite gradients in batch hygiene test")
    return notes


# =============================================================================
# 5. torch.compile (optional)
# =============================================================================
def try_compile():
    print("=" * 78)
    print("5. torch.compile smoke test (small input)")
    import torch._dynamo as dynamo
    out = {}
    for defn in ["+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +ellps=WGS84",
                 "+proj=stere +lat_0=45 +lon_0=10 +k=0.9999 +ellps=WGS84",
                 "+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-105 +R=6371229"]:
        P = L.make(defn)
        x = torch.tensor([[-0.3, 0.7], [0.2, 0.9], [0.1, 0.5]], dtype=torch.float64)
        for name, fn, arg in (("fwd", P.forward, x), ("inv", P.inverse, P.forward(x).detach())):
            dynamo.reset()
            try:
                expl = dynamo.explain(fn)(arg)
                nb = expl.graph_break_count
            except Exception as ex:  # pragma: no cover
                nb = "explain failed: %s" % ex
            try:
                t0 = time.perf_counter()
                cf = torch.compile(fn, fullgraph=True)
                r = cf(arg)
                dt = time.perf_counter() - t0
                same = float((r - fn(arg)).abs().max())
                msg = "ok (%.1fs compile, max diff vs eager %.1e)" % (dt, same)
            except Exception as ex:
                msg = "FAILED: %s" % str(ex).splitlines()[0][:120]
            try:
                ag = arg.clone().requires_grad_(True)
                g1, = torch.autograd.grad(cf(ag).sum(), ag)
                ag2 = arg.clone().requires_grad_(True)
                g2, = torch.autograd.grad(fn(ag2).sum(), ag2)
                msg += "; backward max diff %.1e" % float((g1 - g2).abs().max())
            except Exception as ex:
                msg += "; backward FAILED: %s" % str(ex).splitlines()[0][:100]
            out[defn[:30] + " " + name] = (nb, msg)
            print("  %-50s %s graph breaks=%s inductor %s" % (defn[:50], name, nb, msg))
    return out


def main():
    t0 = time.time()
    gie = run_gie()
    rows = compare_pyproj()
    special_points()
    it = iterations()
    run_gradcheck()
    singular_points()
    if "--compile" in sys.argv:
        try_compile()
    print("=" * 78)
    print("SUMMARY")
    print("  gie: %d/%d passed in %d blocks" % (gie["passed"], gie["total"], gie["blocks"]))
    print("  pyproj max errors (m): fwd64 %.2e inv64 %.2e fwd32 %.2e inv32 %.2e" % (
        max(r["fwd_max_err_fp64_m"] for r in rows), max(r["inv_max_err_fp64_m"] for r in rows),
        max(r["fwd_max_err_fp32_m"] for r in rows), max(r["inv_max_err_fp32_m"] for r in rows)))
    print("  elapsed %.1fs" % (time.time() - t0))
    if FAILURES:
        print("  %d FAILURE(S):" % len(FAILURES))
        for f in FAILURES:
            print("   -", f)
        sys.exit(1)
    print("  ALL OK")


if __name__ == "__main__":
    main()
