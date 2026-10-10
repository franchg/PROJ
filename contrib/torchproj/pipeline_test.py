"""Tests for pipeline.py: gie vectors, pyproj comparisons, autograd, fp32.

Run: cd contrib/torchproj && python pipeline_test.py [-v]
Prints a summary; exit status != 0 on failure.
"""
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
torch.set_num_threads(1)

import pipeline as P  # noqa: E402
import testing  # noqa: E402
from testing import angular_error_m  # noqa: E402

import pyproj  # noqa: E402
from pyproj import Transformer  # noqa: E402
from pyproj.enums import TransformDirection  # noqa: E402

_PROJ_ROOT = Path(__file__).resolve().parents[2]  # root of the PROJ source tree

pyproj.network.set_network_enabled(False)
pyproj.datadir.append_data_dir(f"{_PROJ_ROOT}/data/tests")
VERBOSE = "-v" in sys.argv
# Transformer.from_crs("EPSG:4326", "EPSG:4277") without OSTN15 grid/network: best available candidate
# ("Inverse of OSGB36 to WGS 84 (6)"); section_pyproj checks that pyproj still returns this string.
D4277 = ("proj=pipeline step proj=axisswap order=2,1 step proj=unitconvert xy_in=deg xy_out=rad step proj=push v_3 "
         "step proj=cart ellps=WGS84 step inv proj=helmert x=446.448 y=-125.157 z=542.06 rx=0.15 ry=0.247 "
         "rz=0.842 s=-20.489 convention=position_vector step inv proj=cart ellps=airy step proj=pop v_3 "
         "step proj=unitconvert xy_in=rad xy_out=deg step proj=axisswap order=2,1")
FAILURES = []
REPORT = {}


def check(name, ok, detail=""):
    if not ok:
        FAILURES.append("%s %s" % (name, detail))
    if VERBOSE or not ok:
        print(("  ok   " if ok else "  FAIL ") + name + (" " + detail if detail else ""))


# =============================================================================
# 1. gie
# =============================================================================
GIE_DEFAULT_TOL = 0.0005


def gie_num(tok):
    if tok.upper() == "HUGE_VAL":
        return math.inf
    if tok.lower() == "nan":
        return math.nan
    return testing.parse_num(tok.replace("_", ""))


def gie_coord(args):
    vals = []
    for tok in args.split()[:4]:
        try:
            vals.append(gie_num(tok))
        except ValueError:
            break
    return vals


def op_names(op_text):
    return [t[5:] for t in P.proj_tokenize(op_text) if t.startswith("proj=")]


def classify(op_text):
    """'core' (only ops of pipeline.py), 'external' (also uses projections of other
    modules) or None (unsupported: skip)."""
    toks = P.proj_tokenize(op_text)
    if any(t.startswith(("init=", "geoidgrids=")) for t in toks) or op_text.strip().startswith("urn:"):
        return None  # needs init files / vertical grids / the EPSG database
    names = op_names(op_text)
    if not names:
        return "core"  # e.g. bogus definition: creation must fail
    kind = "core"
    for n in names:
        if n in P.REGISTRY or n == "pipeline":
            continue
        if n in P.EXTERNAL_PROJECTIONS:
            kind = "external_pipeline" if "pipeline" in names else "external_single"
            continue
        return None
    return kind


def gie_dist(T, fwd, exp, got, ndim):
    """gie.cpp:1100-1170 expect() comparison (metres)."""
    exp = list(exp) + [0.0] * (4 - len(exp))
    got = list(got)
    if ndim < 4:
        got[3] = 0.0
    if ndim < 3:
        got[2] = 0.0
    if math.isnan(got[0]) and math.isnan(exp[0]):
        return 0.0
    if T.angular_output(fwd) or T.degree_output(fwd):
        d = float(angular_error_m(exp[0], exp[1], got[0], got[1]))
        return math.hypot(d, exp[2] - got[2])
    return math.sqrt((exp[0] - got[0]) ** 2 + (exp[1] - got[1]) ** 2 + (exp[2] - got[2]) ** 2)


def to_input(T, fwd, a):
    a = list(a) + [0.0] * (4 - len(a))  # gie: proj_coord(0, 0, 0, 0)
    if T.angular_input(fwd):
        a[0], a[1] = math.radians(a[0]), math.radians(a[1])
    return torch.tensor([a], dtype=torch.float64)


def from_output(T, fwd, c):
    c = c[0].tolist()
    if T.angular_output(fwd):
        c[0], c[1] = math.degrees(c[0]), math.degrees(c[1])
    return c


def run_gie_file(path, stats, kinds=("core", "external_pipeline", "external_single")):
    T = None
    err = None
    kind = None
    fwd = True
    tol = GIE_DEFAULT_TOL
    accept = None
    skip_block = False
    for cmd, args in testing._logical_lines(path):
        if cmd == "operation":
            fwd, tol = True, GIE_DEFAULT_TOL
            kind = classify(args)
            skip_block = kind is None or kind not in kinds
            T, err = None, None
            skip_reason = "unsupported operation(s) %s" % sorted(set(op_names(args)) - set(P.REGISTRY)
                                                                  - set(P.EXTERNAL_PROJECTIONS) - {"pipeline"}) \
                if kind is None else "kind %s" % kind
            if not skip_block:
                try:
                    T = P.create(args)
                except P.NotSupported as exc:
                    skip_block, skip_reason = True, str(exc)[:60]
                except P.ProjError as exc:
                    err = exc
            if not skip_block:
                blocks = stats.setdefault("blocks", {})
                blocks[kind] = blocks.get(kind, 0) + 1
            op_text = args
            continue
        if cmd in ("crs_src", "crs_dst", "require_grid"):
            if cmd != "require_grid" or not P._find_grid_file(args.split()[0]):
                skip_block, skip_reason = True, cmd
            continue
        if skip_block and cmd in ("expect", "roundtrip"):
            sk = stats.setdefault("skipped", {})
            key = skip_reason if not skip_reason.startswith("unsupported") else "unsupported operations"
            sk[key] = sk.get(key, 0) + 1
            continue
        if skip_block or cmd in ("banner", "echo", "verbose", "use_proj4_init_rules", "ignore", "skip"):
            continue
        if cmd == "direction":
            fwd = not (args.strip()[:1].lower() in "ir")
        elif cmd == "tolerance":
            tol = testing._scaled(args)
        elif cmd == "accept":
            accept = gie_coord(args)
        elif cmd == "expect":
            key = "%s:%s" % (path.name, kind)
            st = stats.setdefault(key, [0, 0, []])
            st[0] += 1
            want_fail = args.strip().startswith("failure")
            if err is not None:
                ok = want_fail
                detail = "creation failed: %s" % err
            elif want_fail:
                out = T.transform(to_input(T, fwd, accept or [0, 0]), "fwd" if fwd else "inv")
                ok = bool(torch.isnan(out[0, 0]))
                detail = "failed to fail: got %s" % out[0].tolist()
            else:
                exp = gie_coord(args)
                if T.angular_output(fwd):
                    pass  # gie compares in radians via geodesic; we use degrees -> metres
                out = T.transform(to_input(T, fwd, accept), "fwd" if fwd else "inv")
                got = from_output(T, fwd, out)
                d = gie_dist(T, fwd, exp, got, len(exp))
                ok = d <= tol
                if ok and len(exp) == 4 and abs(exp[3] - got[3]) > 1e-4:
                    ok = False
                detail = "d=%.3g m tol=%.3g got=%s exp=%s" % (d, tol, got, exp)
            if ok:
                st[1] += 1
            else:
                st[2].append("%s [%s] %s accept=%s %s" % (path.name, "fwd" if fwd else "inv",
                                                          op_text[:90], accept, detail))
        elif cmd == "roundtrip":
            key = "%s:%s" % (path.name, kind)
            st = stats.setdefault(key, [0, 0, []])
            st[0] += 1
            toks = args.split()
            n = int(float(toks[0])) if toks else 100
            rtol = testing._scaled(" ".join(toks[1:])) if len(toks) > 1 else tol
            if err is not None or accept is None:
                ok, detail = False, "no operation/accept"
            else:
                org = to_input(T, fwd, accept)
                c = T.transform(org, "fwd" if fwd else "inv")
                for _ in range(n - 1):
                    c = T.transform(T.transform(c, "inv" if fwd else "fwd"), "fwd" if fwd else "inv")
                c = T.transform(c, "inv" if fwd else "fwd")
                if torch.isnan(org).any() and torch.isnan(c).all():
                    d = 0.0
                elif T.angular_input(fwd):
                    o, g = org[0].tolist(), c[0].tolist()
                    d = math.hypot(float(angular_error_m(math.degrees(o[0]), math.degrees(o[1]),
                                                         math.degrees(g[0]), math.degrees(g[1]))), o[2] - g[2])
                else:
                    d = float(torch.linalg.norm(org[0, :3] - c[0, :3]))
                ok = d <= rtol
                detail = "roundtrip d=%.3g tol=%.3g" % (d, rtol)
            if ok:
                st[1] += 1
            else:
                st[2].append("%s roundtrip %s accept=%s %s" % (path.name, op_text[:90], accept, detail))


_RAW_OPS = {}


def raw_operation(blk):
    """testing._clean_operation drops flags such as 'exact'/'abridged' and mangles
    'x = 1' style parameters; recover the raw operation text of the block."""
    lst = _RAW_OPS.setdefault(blk["file"], [[testing._clean_operation(a), a, False]
                                            for c, a in testing._logical_lines(testing.GIE_DIR / blk["file"])
                                            if c == "operation"])
    for item in lst:
        if item[0] == blk["operation"] and not item[2]:
            item[2] = True
            return item[1]
    return blk["operation"]


def run_gie_cases(names):
    """testing.gie_cases (single-operation blocks) for the named operations."""
    res = {}
    for name in names:
        tot = ok = 0
        fails = []
        for blk in testing.gie_cases(name):
            try:
                T = P.create(raw_operation(blk))
            except (P.ProjError, P.NotSupported) as exc:
                T = exc
            for case in blk["cases"]:
                tot += 1
                fwd = case["direction"] == "fwd"
                if isinstance(T, Exception):
                    good = case["expect"] == "failure"
                    det = "creation: %s" % T
                elif case["expect"] == "failure":
                    out = T.transform(to_input(T, fwd, case["accept"]), case["direction"])
                    good = bool(torch.isnan(out[0, 0]))
                    det = "failed to fail"
                else:
                    out = T.transform(to_input(T, fwd, case["accept"]), case["direction"])
                    got = from_output(T, fwd, out)
                    d = gie_dist(T, fwd, case["expect"], got, len(case["expect"]))
                    good = d <= case["tolerance_m"]
                    det = "d=%.3g tol=%.3g" % (d, case["tolerance_m"])
                ok += good
                if not good:
                    fails.append("%s %s %s %s" % (name, blk["operation"], case["accept"], det))
            for rt in blk["roundtrips"]:
                tot += 1
                if isinstance(T, Exception):
                    fails.append("%s roundtrip: %s" % (name, T))
                    continue
                org = to_input(T, True, rt["accept"])
                c = T.forward(org)
                for _ in range(rt["n"] - 1):
                    c = T.forward(T.inverse(c))
                c = T.inverse(c)
                if T.angular_input(True):
                    o, g = org[0].tolist(), c[0].tolist()
                    d = float(angular_error_m(math.degrees(o[0]), math.degrees(o[1]),
                                              math.degrees(g[0]), math.degrees(g[1])))
                else:
                    d = float(torch.linalg.norm(org[0, :3] - c[0, :3]))
                good = d <= rt["tolerance_m"]
                ok += good
                if not good:
                    fails.append("%s roundtrip %s d=%.3g" % (name, blk["operation"], d))
        res[name] = (tot, ok, fails)
    return res


def section_gie():
    print("== gie ==")
    gc = run_gie_cases(["unitconvert", "axisswap", "cart", "helmert", "molodensky", "hgridshift", "noop"])
    tot = sum(v[0] for v in gc.values())
    ok = sum(v[1] for v in gc.values())
    for k, (t, o, f) in gc.items():
        print("  testing.gie_cases(%-11s): %3d/%3d" % (k, o, t))
        for line in f:
            print("      ", line)
    REPORT["gie_cases"] = (tot, ok, [l for v in gc.values() for l in v[2]])
    stats = {}
    for f in sorted(testing.GIE_DIR.glob("*.gie")):
        run_gie_file(f, stats)
    core = [v for k, v in stats.items() if k.endswith(":core")]
    ext = [v for k, v in stats.items() if k.endswith(":external_pipeline")]
    ext1 = [v for k, v in stats.items() if k.endswith(":external_single")]
    skipped = stats.pop("skipped", {})
    REPORT["gie_blocks"] = stats.pop("blocks", {})
    print("  gie operation blocks executed: %s" % REPORT["gie_blocks"])
    REPORT["gie_skipped"] = skipped
    print("  skipped gie tests: %d (%s)" % (sum(skipped.values()), "; ".join(
        "%s: %d" % kv for kv in sorted(skipped.items(), key=lambda kv: -kv[1])[:8])))
    for k in sorted(stats):
        t, o, fl = stats[k]
        print("  extended runner %-40s %3d/%3d" % (k, o, t))
        for line in fl:
            print("      ", line)
    REPORT["gie_ext_core"] = (sum(v[0] for v in core), sum(v[1] for v in core), [l for v in core for l in v[2]])
    REPORT["gie_ext_external"] = (sum(v[0] for v in ext), sum(v[1] for v in ext), [l for v in ext for l in v[2]])
    REPORT["gie_ext_single"] = (sum(v[0] for v in ext1), sum(v[1] for v in ext1))
    print("  gie_cases total %d/%d; extended (pipeline.py ops only) %d/%d; pipelines using other modules' "
          "projections %d/%d; single projections of other modules (not counted) %d/%d"
          % (ok, tot, REPORT["gie_ext_core"][1], REPORT["gie_ext_core"][0], REPORT["gie_ext_external"][1],
             REPORT["gie_ext_external"][0], REPORT["gie_ext_single"][1], REPORT["gie_ext_single"][0]))
    check("gie_cases", ok == tot, "%d/%d" % (ok, tot))
    check("gie extended core", REPORT["gie_ext_core"][1] == REPORT["gie_ext_core"][0])
    check("gie extended pipelines with external projections",
          REPORT["gie_ext_external"][1] == REPORT["gie_ext_external"][0])


# =============================================================================
# 2. pyproj comparisons
# =============================================================================
def pyproj_run(defn, cols, inverse=False):
    t = Transformer.from_pipeline(defn)
    d = TransformDirection.INVERSE if inverse else TransformDirection.FORWARD
    out = t.transform(*[np.asarray(c, dtype=float) for c in cols], direction=d)
    return np.stack([np.asarray(o) for o in out], -1)


def err_metres(ref, got, kind, latlon_order=False):
    """kind: 'xyz' euclidean; 'deg' angular (lon, lat[, h]) -> metres; 'deg_latlon' (lat, lon[, h])."""
    if kind == "xyz":
        return np.sqrt(((ref[:, :3] - got[:, :3]) ** 2).sum(-1))
    lo, la = (1, 0) if kind == "deg_latlon" else (0, 1)
    d = angular_error_m(ref[:, lo], ref[:, la], got[:, lo], got[:, la])
    if ref.shape[1] > 2:
        d = np.hypot(d, ref[:, 2] - got[:, 2])
    return d


def compare(name, defn, fwd_in, fwd_kind, inv_kind, tol64, tol_inv64=None, n4=False):
    """Compare torch vs PROJ forward (fwd_in) and inverse (on PROJ's forward output)."""
    T = P.create(defn)
    ref_f = pyproj_run(defn, [fwd_in[:, i] for i in range(fwd_in.shape[1])])
    ref_i = pyproj_run(defn, [ref_f[:, i] for i in range(ref_f.shape[1])], inverse=True)
    res = {"definition": defn, "n_points": int(fwd_in.shape[0])}
    for dt, tag in ((torch.float64, "64"), (torch.float32, "32")):
        gf = T.forward(torch.tensor(fwd_in, dtype=torch.float64).to(dt)).double().numpy()
        gi = T.inverse(torch.tensor(ref_f, dtype=torch.float64).to(dt)).double().numpy()
        ef = err_metres(ref_f, gf, fwd_kind)
        ei = err_metres(ref_i, gi, inv_kind)
        res["fwd" + tag] = float(np.nanmax(ef)) if np.isfinite(ef).any() else float("nan")
        res["inv" + tag] = float(np.nanmax(ei)) if np.isfinite(ei).any() else float("nan")
        res["nan" + tag] = int(np.isnan(ef).sum() + np.isnan(ei).sum())
        if tag == "32":  # error of the fp32 computation alone: PROJ run on the fp32-rounded inputs
            q_in = fwd_in.astype(np.float32).astype(np.float64)
            q_f = ref_f.astype(np.float32).astype(np.float64)
            rq_f = pyproj_run(defn, [q_in[:, i] for i in range(q_in.shape[1])])
            rq_i = pyproj_run(defn, [q_f[:, i] for i in range(q_f.shape[1])], inverse=True)
            res["fwd32q"] = float(np.nanmax(err_metres(rq_f, gf, fwd_kind)))
            res["inv32q"] = float(np.nanmax(err_metres(rq_i, gi, inv_kind)))
    print("  %-34s n=%d fwd64 %.3g m inv64 %.3g m | fwd32 %.3g m inv32 %.3g m (nan32=%d; vs PROJ on fp32-rounded "
          "input: %.3g / %.3g m)" % (name, res["n_points"], res["fwd64"], res["inv64"], res["fwd32"], res["inv32"],
                                    res["nan32"], res["fwd32q"], res["inv32q"]))
    check(name + " fp64 fwd", res["fwd64"] <= tol64 and res["nan64"] == 0, "%.3g" % res["fwd64"])
    check(name + " fp64 inv", res["inv64"] <= (tol_inv64 or tol64), "%.3g" % res["inv64"])
    REPORT.setdefault("pyproj", []).append((name, res))
    return T, res


def section_pyproj():
    print("== pyproj comparisons ==")
    rng = np.random.default_rng(42)
    n = 2000
    # (a) EPSG:4326 -> EPSG:4978
    d1 = Transformer.from_crs("EPSG:4326", "EPSG:4978").definition
    print("  EPSG:4326->EPSG:4978:", d1)
    pts = np.stack([rng.uniform(-90, 90, n), rng.uniform(-180, 180, n), rng.uniform(-500, 9000, n)], -1)
    pts[:4] = [[90, 0, 0], [-90, 10, 100], [0, 180, 0], [0, -180, 0]]
    compare("EPSG:4326->EPSG:4978", d1, pts, "xyz", "deg_latlon", 1e-8)
    # (b) explicit 7-parameter Helmert with push/pop, exact + approximate, both conventions
    for conv, exact in (("position_vector", ""), ("coordinate_frame", "+exact")):
        d2 = ("+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=push +v_3 "
              "+step +proj=cart +ellps=GRS80 +step +proj=helmert +x=-81.0703 +y=-89.3603 +z=-115.7526 "
              "+rx=-0.48488 +ry=-0.02436 +rz=-0.41321 +s=-0.540645 +convention=%s %s "
              "+step +inv +proj=cart +ellps=intl +step +proj=pop +v_3 "
              "+step +proj=unitconvert +xy_in=rad +xy_out=deg" % (conv, exact))
        pts = np.stack([rng.uniform(-180, 180, n), rng.uniform(-89.9, 89.9, n), rng.uniform(-100, 3000, n)], -1)
        compare("7p Helmert %s%s" % (conv, exact), d2, pts, "deg", "deg", 1e-6)
    # (c) 15-parameter time dependent Helmert (ITRF2000->ITRF93 from more_builtins.gie), 4D input
    d3 = ("+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=cart +ellps=GRS80 "
          "+step +proj=helmert +convention=position_vector +x=0.0127 +dx=-0.0029 +rx=-0.00039 +drx=-0.00011 "
          "+y=0.0065 +dy=-0.0002 +ry=0.00080 +dry=-0.00019 +z=-0.0209 +dz=-0.0006 +rz=-0.00114 +drz=0.00007 "
          "+s=0.00195 +ds=0.00001 +t_epoch=1988.0 +exact +step +inv +proj=cart +ellps=GRS80 "
          "+step +proj=unitconvert +xy_in=rad +xy_out=deg")
    pts = np.stack([rng.uniform(-180, 180, n), rng.uniform(-89, 89, n), rng.uniform(-100, 3000, n),
                    rng.uniform(1980, 2030, n)], -1)
    compare("15p time-dependent Helmert (4D)", d3, pts, "deg", "deg", 1e-6)
    # (d) EPSG:4326 -> EPSG:4277 (OSGB36): no grids/network -> Helmert fallback
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from pyproj.transformer import TransformerGroup
        grp = TransformerGroup("EPSG:4326", "EPSG:4277")
        tr = grp.transformers[0]
    d4 = tr.definition
    print("  EPSG:4326->EPSG:4277 (%s):\n    %s" % (tr.description, d4))
    check("EPSG:4326->EPSG:4277 definition unchanged", d4 == D4277, d4)
    pts = np.stack([rng.uniform(49.9, 60.8, n), rng.uniform(-8.6, 1.8, n)], -1)
    compare("EPSG:4326->EPSG:4277", d4, pts, "deg_latlon", "deg_latlon", 1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t_crs = Transformer.from_crs("EPSG:4326", "EPSG:4277")
        lat2, lon2 = t_crs.transform(pts[:, 0], pts[:, 1])
    got = P.create(d4).forward(torch.tensor(pts)).numpy()
    e = angular_error_m(lon2, lat2, got[:, 1], got[:, 0])
    print("    vs Transformer.from_crs(4326, 4277).transform: max %.3g m" % e.max())
    check("EPSG:4326->EPSG:4277 vs from_crs", e.max() < 1e-6)
    # (e) NTF -> RGF93 (EPSG:4275 -> EPSG:4171), 3-parameter Helmert pipeline from the database
    # (definition printed by Transformer.from_crs("EPSG:4275", "EPSG:4171") when the test grids are not on
    # the search path; with them, PROJ prefers the fr_ign_ntf_r93.tif grid, tested as NTv2 in (f))
    d5 = ("proj=pipeline step proj=axisswap order=2,1 step proj=unitconvert xy_in=deg xy_out=rad step proj=push "
          "v_3 step proj=cart ellps=clrk80ign step proj=helmert x=-168 y=-60 z=320 step inv proj=cart ellps=GRS80 "
          "step proj=pop v_3 step proj=unitconvert xy_in=rad xy_out=deg step proj=axisswap order=2,1")
    pts = np.stack([rng.uniform(41.5, 51, n), rng.uniform(-5, 9.5, n)], -1)
    compare("EPSG:4275->EPSG:4171 (3p)", d5, pts, "deg_latlon", "deg_latlon", 1e-6)
    # (f) hgridshift NTv2
    for grid, lon_r, lat_r in ((f"{_PROJ_ROOT}/data/tests/ntf_r93.gsb", (-5.4, 9.4), (41.1, 51.9)),
                               (f"{_PROJ_ROOT}/data/tests/ntv2_0_downsampled.gsb", (-141, -45), (41, 83)),
                               ("ntf_r93.gsb,null", (-20, 20), (30, 60))):
        d6 = ("+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=hgridshift +grids=%s "
              "+step +proj=unitconvert +xy_in=rad +xy_out=deg" % grid)
        pts = np.stack([rng.uniform(*lon_r, n), rng.uniform(*lat_r, n)], -1)
        if "ntv2_0" in grid:  # densify inside the 3 child subgrids
            sub = [((-83.0, -81.75), (41.92, 42.38)), ((-112.99, -112.58), (49.33, 49.51)),
                   ((-115.58, -115.5), (51.08, 51.25))]
            for k, ((a, b), (c, d)) in enumerate(sub):
                m = slice(k * 200, (k + 1) * 200)
                pts[m] = np.stack([rng.uniform(a, b, 200), rng.uniform(c, d, 200)], -1)
        T = P.create(d6)
        ref = pyproj_run(d6, [pts[:, 0], pts[:, 1]])
        ok = np.isfinite(ref).all(-1)
        got = T.forward(torch.tensor(pts)).numpy()
        same_fail = (np.isnan(got).any(-1) == ~ok).all()
        print("  hgridshift %s: %d/%d points inside, identical failure mask: %s"
              % (grid.split("/")[-1], ok.sum(), n, same_fail))
        check("hgridshift %s failure mask" % grid, bool(same_fail))
        compare("hgridshift %s" % grid.split("/")[-1], d6, pts[ok], "deg", "deg", 1e-6)
    # (g) Molodensky (standard + abridged)
    for ab in ("", "+abridged"):
        d7 = ("+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=molodensky "
              "+a=6378160 +rf=298.25 +da=-23 +df=-8.120449e-8 +dx=-134 +dy=-48 +dz=149 %s "
              "+step +proj=unitconvert +xy_in=rad +xy_out=deg" % ab)
        pts = np.stack([rng.uniform(-180, 180, n), rng.uniform(-89, 89, n), rng.uniform(-100, 3000, n)], -1)
        compare("molodensky %s" % (ab or "standard"), d7, pts, "deg", "deg", 1e-6)
    # (h) unitconvert with time units + axisswap 4D
    d8 = ("+proj=pipeline +step +proj=axisswap +order=2,1,-3,4 +step +proj=unitconvert +xy_in=us-ft +xy_out=km "
          "+z_in=m +z_out=ft +t_in=decimalyear +t_out=gps_week")
    pts = np.stack([rng.uniform(-1e7, 1e7, n), rng.uniform(-1e7, 1e7, n), rng.uniform(-100, 3000, n),
                    rng.uniform(1900, 2100, n)], -1)
    T = P.create(d8)
    ref = pyproj_run(d8, [pts[:, i] for i in range(4)])
    got = T.forward(torch.tensor(pts)).numpy()
    e = np.abs(ref - got).max(0)
    ri = pyproj_run(d8, [ref[:, i] for i in range(4)], inverse=True)
    gi = T.inverse(torch.tensor(ref)).numpy()
    ei = np.abs(ri - gi).max(0)
    print("  unitconvert+axisswap 4D: fwd max |d| per comp %s, inv %s" % (e, ei))
    check("unitconvert time units", e.max() < 1e-6 and ei.max() < 1e-6)
    # (i) affine / geocent / latlong+towgs84 (cs2cs emulation) single operations
    d9 = ("+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=longlat +ellps=intl "
          "+towgs84=-81.0703,-89.3603,-115.7526,0.48488,0.02436,0.41321,-0.540645 +inv "
          "+step +proj=unitconvert +xy_in=rad +xy_out=deg")
    pts = np.stack([rng.uniform(-180, 180, n), rng.uniform(-89, 89, n), rng.uniform(-100, 3000, n)], -1)
    compare("latlong +towgs84 (cs2cs emulation)", d9, pts, "deg", "deg", 1e-6)


# =============================================================================
# 3. autograd
# =============================================================================
def section_autograd():
    print("== autograd ==")
    torch.manual_seed(0)
    gc_res = {}

    def gcheck(name, fn, x, s_in=1.0, s_out=1.0):
        """gradcheck in O(1) units: f(u) = fn(u * s_in) / s_out (metres would otherwise make the
        central differences of 6e6 m outputs useless at eps=1e-6)."""
        s_in = torch.as_tensor(s_in, dtype=torch.float64)
        s_out = torch.as_tensor(s_out, dtype=torch.float64)
        u = (x / s_in).clone().requires_grad_(True)
        try:
            ok = torch.autograd.gradcheck(lambda v: fn(v * s_in) / s_out, (u,), eps=1e-6, atol=1e-6,
                                          rtol=1e-5, nondet_tol=0.0)
        except Exception as exc:  # report the first line
            ok = False
            print("    gradcheck %s: %s" % (name, str(exc).splitlines()[0][:200]))
        gc_res[name] = ok
        check("gradcheck " + name, ok)

    deg = math.pi / 180
    # interior points, radians/metres
    geo = torch.stack([torch.empty(6).uniform_(-3, 3), torch.empty(6).uniform_(-1.4, 1.4),
                       torch.empty(6).uniform_(-100, 1000)], -1).double()
    cart = P.create("+proj=cart +ellps=GRS80")
    R, GEO, XYZ = 6.4e6, [1.0, 1.0, 1e3], [6.4e6] * 3
    gcheck("cart fwd", cart.forward, geo, GEO, R)
    xyz = cart.forward(geo).detach()
    gcheck("cart inv", cart.inverse, xyz, XYZ, GEO)
    h7 = ("+proj=pipeline +step +proj=push +v_3 +step +proj=cart +ellps=GRS80 +step +proj=helmert "
          "+x=-81.07 +y=-89.36 +z=-115.75 +rx=-0.48 +ry=-0.02 +rz=-0.41 +s=-0.54 +convention=coordinate_frame "
          "+exact +step +inv +proj=cart +ellps=intl +step +proj=pop +v_3")
    H = P.create(h7)
    gcheck("7p helmert pipeline fwd", H.forward, geo, GEO, GEO)
    gcheck("7p helmert pipeline inv", H.inverse, H.forward(geo).detach(), GEO, GEO)
    h15 = ("+proj=helmert +convention=position_vector +x=0.0127 +dx=-0.0029 +rx=-0.00039 +drx=-0.00011 "
           "+y=0.0065 +dy=-0.0002 +ry=0.00080 +dry=-0.00019 +z=-0.0209 +dz=-0.0006 +rz=-0.00114 +drz=0.00007 "
           "+s=0.00195 +ds=0.00001 +t_epoch=1988.0")
    H15 = P.create(h15)
    x4 = torch.cat([xyz, torch.empty(6, 1).uniform_(1990, 2020).double()], -1)
    gcheck("15p helmert fwd (incl. d/dt)", H15.forward, x4, XYZ + [1.0], XYZ + [1.0])
    gcheck("15p helmert inv", H15.inverse, H15.forward(x4).detach(), XYZ + [1.0], XYZ + [1.0])
    gcheck("molodensky fwd", P.create("+proj=molodensky +a=6378160 +rf=298.25 +da=-23 +df=-8.120449e-8 "
                                      "+dx=-134 +dy=-48 +dz=149").forward, geo, GEO, GEO)
    G = P.create(f"+proj=hgridshift +grids={_PROJ_ROOT}/data/tests/ntf_r93.gsb")
    # interior points well inside cells (cell = 0.1 deg): offset 0.03..0.07 of a cell
    ll = torch.tensor([[2.0 + 0.1 * k + 0.05, 46.0 + 0.1 * k + 0.033] for k in range(5)]).double() * deg
    gcheck("hgridshift fwd", G.forward, ll)
    gcheck("hgridshift inv (unrolled iteration)", G.inverse, G.forward(ll).detach())
    G2 = P.create(f"+proj=hgridshift +grids={_PROJ_ROOT}/data/tests/ntv2_0_downsampled.gsb")
    ll2 = torch.tensor([[-82.505, 42.105], [-112.805, 49.405], [-100.3, 55.3]]).double() * deg
    gcheck("hgridshift fwd (NTv2 subgrids)", G2.forward, ll2)
    gcheck("hgridshift inv (NTv2 subgrids)", G2.inverse, G2.forward(ll2).detach())
    d4326 = Transformer.from_crs("EPSG:4326", "EPSG:4978").definition
    T = P.create(d4326)
    gcheck("EPSG:4326->4978 fwd", T.forward, torch.tensor([[45.0, 7.0, 100.0], [-33.0, 151.0, 10.0]],
                                                          dtype=torch.float64), GEO, R)
    gcheck("EPSG:4326->4277 fwd", P.create(D4277).forward, torch.tensor([[52.1, -1.3], [55.0, -4.0]],
                                                                         dtype=torch.float64))
    gcheck("EPSG:4326->4277 inv", P.create(D4277).inverse, torch.tensor([[52.1, -1.3], [55.0, -4.0]],
                                                                         dtype=torch.float64))
    REPORT["gradcheck"] = gc_res
    print("  gradcheck (fp64, interior points): %d/%d passed: %s"
          % (sum(gc_res.values()), len(gc_res), ", ".join(k for k, v in gc_res.items() if v)))

    # singular points: what do gradients do?
    sing = {}

    def grad_at(T, x, fwd=True, comp=None):
        x = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        y = T.forward(x) if fwd else T.inverse(x)
        s = y.sum() if comp is None else y[..., comp].sum()
        g, = torch.autograd.grad(s, x, allow_unused=True)
        return y.detach().numpy().tolist(), g.numpy().tolist()

    sing["cart inv at north pole (0,0,b)"] = grad_at(cart, ([[0.0, 0.0, 6356752.314140347]]), False)
    sing["cart inv at earth centre (0,0,0)"] = grad_at(cart, ([[0.0, 0.0, 0.0]]), False)
    sing["cart fwd at pole (lat=90)"] = grad_at(cart, ([[0.3, math.pi / 2, 10.0]]))
    sing["cart fwd lon=pi (antimeridian)"] = grad_at(cart, ([[math.pi, 0.5, 0.0]]))
    sing["cart fwd lon=pi+1e-9 (adjlon wrap)"] = grad_at(cart, ([[math.pi + 1e-9, 0.5, 0.0]]))
    sing["cart fwd lat=91deg (out of domain)"] = grad_at(cart, ([[0.3, 91 * deg, 0.0]]))
    sing["hgridshift fwd on a cell edge"] = grad_at(G, ([[2.0 * deg, 46.0 * deg]]))
    sing["hgridshift fwd outside grid"] = grad_at(G, ([[30.0 * deg, 46.0 * deg]]))
    sing["hgridshift inv outside grid"] = grad_at(G, ([[30.0 * deg, 46.0 * deg]]), False)
    sing["7p pipeline fwd at pole"] = grad_at(H, ([[0.0, math.pi / 2, 0.0]]))
    # NaN hygiene: a failed point inside a batch must not poison gradients of shared parameters
    off = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    pts = torch.tensor([[2.3 * deg, 46.3 * deg], [30.0 * deg, 46.0 * deg]], dtype=torch.float64)
    for direction in ("fwd", "inv"):
        out = G.transform(pts + off, direction)
        loss = torch.where(torch.isnan(out), 0.0, out).sum()
        g, = torch.autograd.grad(loss, off)
        fin = bool(torch.isfinite(g).all())
        sing["hgridshift %s shared offset, 1 of 2 points outside grid" % direction] = (out.tolist(), g.tolist())
        check("no NaN gradient leak from failed points (hgridshift %s)" % direction, fin, str(g.tolist()))
    for k, (y, g) in sing.items():
        print("    %-40s out=%s grad=%s" % (k, np.round(y, 6).tolist(), np.round(g, 6).tolist()))
    REPORT["singular"] = sing


# =============================================================================
# 4. compile readiness (tiny input)
# =============================================================================
def section_compile():
    print("== torch.compile (small input) ==")
    d = ("+proj=pipeline +step +proj=axisswap +order=2,1 +step +proj=unitconvert +xy_in=deg +xy_out=rad "
         "+step +proj=push +v_3 +step +proj=cart +ellps=WGS84 +step +inv +proj=helmert +x=446.448 "
         "+y=-125.157 +z=542.06 +rx=0.15 +ry=0.247 +rz=0.842 +s=-20.489 +convention=position_vector "
         "+step +inv +proj=cart +ellps=airy +step +proj=pop +v_3 +step +proj=unitconvert +xy_in=rad "
         "+xy_out=deg +step +proj=axisswap +order=2,1")
    T = P.create(d)
    x = torch.tensor([[52.0, -1.0], [55.0, -3.0]], dtype=torch.float64)
    res = {}
    for name, mod, inp in (("helmert pipeline", T, x),
                           ("hgridshift inv", P.create(f"+proj=hgridshift +grids={_PROJ_ROOT}/data/tests/"
                                                       "ntf_r93.gsb"), torch.tensor([[0.04, 0.81]]).double())):
        t0 = time.time()
        try:
            f = torch.compile(mod.inverse if "inv" in name else mod.forward, fullgraph=True)
            y = f(inp)
            y_ref = mod.inverse(inp) if "inv" in name else mod.forward(inp)
            res[name] = "ok (fullgraph, %.1fs, max diff %.2g)" % (time.time() - t0, (y - y_ref).abs().max())
        except Exception as exc:
            res[name] = "FAILED: %s" % str(exc).splitlines()[0][:200]
        print("   ", name, res[name])
    REPORT["compile"] = res


if __name__ == "__main__":
    t0 = time.time()
    section_gie()
    section_pyproj()
    section_autograd()
    if "--compile" in sys.argv:
        section_compile()
    print("== summary ==")
    print("  gie_cases %d/%d, extended gie (pipeline.py only) %d/%d, extended gie pipelines using other "
          "modules' projections %d/%d, blocks %s, skipped tests %d"
          % (REPORT["gie_cases"][1], REPORT["gie_cases"][0], REPORT["gie_ext_core"][1],
             REPORT["gie_ext_core"][0], REPORT["gie_ext_external"][1], REPORT["gie_ext_external"][0],
             REPORT["gie_blocks"], sum(REPORT["gie_skipped"].values())))
    print("  %d failure(s) in %.1fs" % (len(FAILURES), time.time() - t0))
    for f in FAILURES:
        print("   -", f)
    sys.exit(1 if FAILURES else 0)
