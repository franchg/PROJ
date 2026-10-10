"""Integration tests: registry, end-to-end CRS pairs, step inventory, autograd vs PROJ factors.

Run: cd contrib/torchproj && python integ_test.py [--n 100000] [--sections reg,e2e,steps,factors]
Exit status: number of failed checks (accuracy thresholds below are generous on purpose;
the numbers printed are the result, the checks only catch breakage).
"""
import argparse
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

import pipeline as PL  # noqa: E402
import integ_registry as IR  # noqa: E402
from testing import angular_error_m  # noqa: E402

import pyproj  # noqa: E402
from pyproj import CRS, Transformer  # noqa: E402
from pyproj.transformer import TransformerGroup  # noqa: E402

_PROJ_ROOT = Path(__file__).resolve().parents[2]  # root of the PROJ source tree

pyproj.network.set_network_enabled(False)
pyproj.datadir.append_data_dir(f"{_PROJ_ROOT}/data/tests")
warnings.filterwarnings("ignore", message="Best transformation is not available")

FAIL = []


def check(name, ok, detail=""):
    if not ok:
        FAIL.append(name + " " + detail)
        print("  FAIL", name, detail)


# =============================================================================
# 1. Registry
# =============================================================================
REG_CASES = [  # (name, definition, lon range, lat range)
    ("tmerc", "+proj=tmerc +lon_0=10 +ellps=WGS84", (-20, 40), (-80, 80)),
    ("etmerc", "+proj=etmerc +lon_0=10 +ellps=WGS84", (-20, 40), (-80, 80)),
    ("utm", "+proj=utm +zone=33 +south +ellps=WGS84", (9, 21), (-80, 0)),
    ("lcc", "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +ellps=WGS84", (-140, -50), (10, 65)),
    ("stere", "+proj=stere +lat_0=45 +lon_0=10 +k=0.9999 +ellps=WGS84", (-30, 50), (10, 80)),
    ("ups", "+proj=ups +ellps=WGS84", (-180, 180), (60, 90)),
    ("laea", "+proj=laea +lat_0=52 +lon_0=10 +ellps=GRS80", (-40, 60), (20, 85)),
    ("healpix", "+proj=healpix +ellps=WGS84", (-180, 180), (-90, 90)),
    ("rhealpix", "+proj=rhealpix +ellps=WGS84", (-180, 180), (-90, 90)),
    ("moll", "+proj=moll +R=6371000", (-180, 180), (-89, 89)),
    ("wag4", "+proj=wag4 +R=6371000", (-180, 180), (-89, 89)),
    ("wag5", "+proj=wag5 +R=6371000", (-180, 180), (-89, 89)),
    ("geos", "+proj=geos +h=35786023 +lon_0=-75 +sweep=x +ellps=GRS80", (-150, 0), (-80, 80)),
    ("eqc", "+proj=eqc +ellps=WGS84 +lat_ts=30", (-180, 180), (-90, 90)),
    ("ob_tran", "+proj=ob_tran +o_proj=eqc +o_lon_p=-170 +o_lat_p=43 +lon_0=10 +R=6371229", (-180, 180), (-89, 89)),
    ("merc", "+proj=merc +lat_ts=20 +ellps=WGS84", (-180, 180), (-85, 85)),
    ("webmerc", "+proj=webmerc +ellps=WGS84", (-180, 180), (-85, 85)),
    ("ob_tran+lcc", "+proj=ob_tran +o_proj=lcc +lat_1=30 +lat_2=60 +o_lon_p=-170 +o_lat_p=43 +lon_0=10 +R=6371229",
     (-30, 50), (25, 75)),
    ("ob_tran+laea", "+proj=ob_tran +o_proj=laea +o_lon_p=20 +o_lat_p=60 +R=6371229", (-30, 50), (25, 75)),
]

# pipeline_adv_test.py defects of "projection step silently ignores ..." + two real CRSs
GENERIC_CASES = [
    ("+pm=paris", "+proj=lcc +lat_1=46.8 +lat_0=46.8 +lon_0=0 +k_0=0.99987742 +x_0=600000 +y_0=2200000 "
                  "+ellps=clrk80ign +pm=paris"),
    ("+axis=wsu", "+proj=tmerc +lon_0=15 +ellps=WGS84 +axis=wsu"),
    ("+to_meter=0.3048", "+proj=laea +lat_0=52 +lon_0=10 +ellps=GRS80 +to_meter=0.3048"),
    ("+units=us-ft", "+proj=tmerc +lon_0=15 +ellps=WGS84 +units=us-ft"),
    ("+geoc", "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +ellps=WGS84 +geoc"),
    ("+vunits=ft +z_0=10", "+proj=utm +zone=32 +ellps=WGS84 +vunits=ft +z_0=10"),
    ("+R_A", "+proj=laea +lat_0=52 +lon_0=10 +ellps=WGS84 +R_A"),
    ("+south=f", "+proj=utm +zone=32 +ellps=WGS84 +south=f"),
    ("+over=f", "+proj=tmerc +lon_0=170 +ellps=WGS84 +over=f"),
    ("shape a+rf+es (PROJ: rf)", "+proj=lcc +lat_1=20 +lat_2=60 +a=6378137 +rf=298.257223563 +es=0.1"),
    ("shape a+rf+b (PROJ: rf)", "+proj=tmerc +a=6378137 +rf=298.257223563 +b=6000000"),
    ("shape a+rf+f (PROJ: rf)", "+proj=tmerc +a=6378137 +rf=300 +f=0.01"),
    ("shape ellps+e+es (PROJ: es)", "+proj=tmerc +ellps=GRS80 +e=0.1 +es=0.006"),
    ("shape rf=0.5 (PROJ: error)", "+proj=laea +a=6378137 +rf=0.5"),
    ("shape rf, no a (PROJ: error)", "+proj=tmerc +rf=300"),
    ("R + bad rf (PROJ: R wins)", "+proj=eqc +R=6400000 +rf=0"),
    ("k_0=0 (PROJ: error)", "+proj=tmerc +ellps=WGS84 +k_0=0"),
    ("lat_0=91 (PROJ: error)", "+proj=stere +lat_0=91 +ellps=GRS80"),
    ("eqc 0..360 grid at lon_0+180", "+proj=eqc +R=6400000 +lon_0=10"),
    ("+towgs84 (cs2cs)", "+proj=tmerc +lat_0=49 +lon_0=-2 +k_0=0.9996012717 +x_0=400000 +y_0=-100000 +ellps=airy "
                         "+towgs84=446.448,-125.157,542.06,0.15,0.247,0.842,-20.489"),
]


def _rand_lonlat(n, lon, lat, rng):
    return rng.uniform(lon[0], lon[1], n), rng.uniform(lat[0], lat[1], n)


def _pipe(defn):
    return "+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step " + defn


def section_registry(n=2000):
    print("== 1. registry ==")
    rng = np.random.default_rng(1)
    for mode in ("raw", "generic"):
        reg = IR.install(mode)
        print("  mode=%s: %d module projections registered: %s" % (mode, len(reg), ", ".join(sorted(reg))))
        if IR.UNAVAILABLE:
            print("  unavailable:", IR.UNAVAILABLE)
    print("  pipeline.REGISTRY now holds %d operations: %s" % (len(PL.REGISTRY), ", ".join(sorted(PL.REGISTRY))))
    print("  per projection, inside a deg->rad pipeline, %d random points, generic mode, fp64:" % n)
    for name, defn, lon, lat in REG_CASES:
        x, y = _rand_lonlat(n, lon, lat, rng)
        d = _pipe(defn)
        try:
            T = PL.create(d)
            out = T.forward(torch.tensor(np.stack([x, y], -1)))
            rx, ry = Transformer.from_pipeline(d).transform(x, y)
            fin = np.isfinite(rx)
            err = np.hypot(out[:, 0].numpy() - rx, out[:, 1].numpy() - ry)[fin]
            mism = int((np.isfinite(out[:, 0].numpy()) != fin).sum())
            inv = T.inverse(torch.tensor(np.stack([np.where(fin, rx, 0), np.where(fin, ry, 0)], -1)))
            ie = angular_error_m(x, y, inv[:, 0].numpy(), inv[:, 1].numpy())[fin]
            print("    %-13s fwd %.2e m  roundtrip(PROJ fwd -> torch inv) %.2e m  NaN-mask mismatches %d" % (
                name, err.max(), np.nanmax(ie), mism))
            check("registry " + name, err.max() < 1e-6 and mism == 0 and np.nanmax(ie) < 1e-3)
        except Exception as exc:
            print("    %-13s ERROR %s: %s" % (name, type(exc).__name__, exc))
            check("registry " + name, False, repr(exc))
    print("  generic PROJ parameters (pipeline_adv_test defects), max |torch - PROJ| on 2000 pts, raw vs generic:")
    x, y = _rand_lonlat(n, (-10, 30), (35, 70), rng)
    z = rng.uniform(0, 3000, n)
    for label, defn in GENERIC_CASES:
        d = _pipe(defn)
        xx = np.where(np.arange(n) % 2 == 0, 190.0, x) if "0..360" in label else x
        try:
            rx, ry, rz = Transformer.from_pipeline(d).transform(xx, y, z)
        except pyproj.exceptions.ProjError as exc:
            rx = None
        res = []
        for mode in ("raw", "generic"):
            IR.install(mode)
            try:
                out = PL.create(d).forward(torch.tensor(np.stack([xx, y, z], -1))).numpy()
                if rx is None:
                    res.append("built (PROJ refuses)")
                    continue
                e = np.nanmax(np.sqrt((out[:, 0] - rx) ** 2 + (out[:, 1] - ry) ** 2 + (out[:, 2] - rz) ** 2))
                res.append("%.2e m" % e)
            except Exception as exc:
                res.append(("refused OK" if rx is None else "") + "%s(%s)" % (type(exc).__name__, str(exc)[:40]))
        print("    %-30s raw: %-34s generic: %s" % (label, res[0], res[1]))
        good = (res[1].startswith("refused OK") if rx is None
                else res[1].endswith(" m") and float(res[1].split()[0]) < 1e-5)
        check("generic " + label, good, res[1])
    # |lon| > 10 rad must fail (fwd.cpp:72-78)
    for mode in ("raw", "generic"):
        IR.install(mode)
        T = PL.create("+proj=tmerc +lon_0=10 +ellps=WGS84")
        out = T.forward(torch.tensor([[10.5, 0.2], [-12.0, 0.3]], dtype=torch.float64))
        print("    |lon|>10 rad   %-7s -> %s (PROJ: failure)" % (mode, "NaN" if torch.isnan(out).all() else "finite"))
    IR.install("generic")


# =============================================================================
# 2. End-to-end CRS pairs
# =============================================================================
WRF = "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=45 +lon_0=10 +R=6370000 +units=m"
COSMO = "+proj=ob_tran +o_proj=longlat +o_lon_p=-170 +o_lat_p=43 +lon_0=10 +R=6371229 +no_defs"
MSG = "+proj=geos +h=35785831 +a=6378169 +b=6356583.8 +lon_0=0 +sweep=y"
PAIRS = [
    # label, src, dst, sampling box (lon/lat deg; None = CRS area of use), output kind
    ("EPSG:4326->EPSG:32632", "EPSG:4326", "EPSG:32632", None, "xy"),
    ("EPSG:4326->EPSG:3035", "EPSG:4326", "EPSG:3035", None, "xy"),
    ("EPSG:4326->EPSG:3413", "EPSG:4326", "EPSG:3413", None, "xy"),
    ("EPSG:4326->EPSG:4978", "EPSG:4326", "EPSG:4978", None, "xyz"),
    ("EPSG:4326->EPSG:27700", "EPSG:4326", "EPSG:27700", None, "xy"),
    ("EPSG:4326->EPSG:3857", "EPSG:4326", "EPSG:3857", None, "xy"),
    ("WGS84->WRF lcc", "EPSG:4326", WRF, (-40, 15, 60, 80), "xy"),
    ("WGS84->COSMO ob_tran", "EPSG:4326", COSMO, (-30, 25, 50, 75), "deg6371229"),
    ("WGS84->MSG geos", "EPSG:4326", MSG, (-80, -80, 80, 80), "xy"),
    ("EPSG:3035->EPSG:4326 (inv)", "EPSG:3035", "EPSG:4326", None, "deg6378137"),
]
STEP_COUNTS = {}


def _crs(s):
    return CRS.from_user_input(s) if not s.startswith("+") else CRS.from_proj4(s)


def _steps(defn):
    """[(name, inverted)] of a pipeline definition."""
    toks = PL.proj_tokenize(defn)
    out, cur = [], None
    for t in toks + ["step"]:
        if t == "step":
            if cur is not None:
                out.append(tuple(cur))
            cur = [None, False]
        elif cur is not None and t.startswith("proj="):
            cur[0] = t[5:]
        elif cur is not None and t == "inv":
            cur[1] = True
    return [s for s in out if s[0]]


def _err(kind, out, ref):
    if kind == "xy":
        return np.hypot(out[:, 0] - ref[:, 0], out[:, 1] - ref[:, 1])
    if kind == "xyz":
        return np.sqrt(((out[:, :3] - ref[:, :3]) ** 2).sum(-1))
    a = float(kind[3:])
    return angular_error_m(ref[:, 0], ref[:, 1], out[:, 0], out[:, 1], a=a)


def _sample(src, dst, box, n, rng):
    """Uniform points over the area of use, in the source CRS's (always_xy) units."""
    if box is None:
        c = _crs(dst) if _crs(src).is_geographic else _crs(src)
        b = c.area_of_use.bounds
    else:
        b = box
    lon, lat = rng.uniform(b[0], b[2], n), rng.uniform(b[1], b[3], n)
    if _crs(src).is_geographic:
        return np.stack([lon, lat, np.zeros(n)], -1), b
    t = Transformer.from_crs("EPSG:4326", src, always_xy=True)
    xb = t.transform_bounds(*b)
    x, y = rng.uniform(xb[0], xb[2], n), rng.uniform(xb[1], xb[3], n)
    return np.stack([x, y, np.zeros(n)], -1), xb


def _group_summary(src, dst):
    g = TransformerGroup(_crs(src), _crs(dst), always_xy=True)
    lines = ["%d candidate(s), best_available=%s" % (len(g.transformers), g.best_available)]
    for t in g.transformers[:9]:
        lines.append("      cand: %-90s acc=%s steps=%s" % (
            t.description[:90], t.accuracy, ",".join(("inv " if i else "") + s for s, i in _steps(t.definition))))
    for op in g.unavailable_operations:
        lines.append("      UNAVAILABLE: %s needs grids %s" % (op.name, [gr.short_name for gr in op.grids]))
    return g, lines


def section_e2e(n=100000):
    print("== 2. end-to-end CRS pairs (n=%d per pair, generic adapter mode) ==" % n)
    rng = np.random.default_rng(2024)
    rows = []
    for label, src, dst, box, kind in PAIRS:
        print("--", label)
        t = Transformer.from_crs(_crs(src), _crs(dst), always_xy=True)
        g, lines = _group_summary(src, dst)
        defn = t.definition
        note = ""
        if defn.startswith("unavailable"):
            defn = g.transformers[0].definition
            note = "from_crs().definition is '%s'; using TransformerGroup best available" % t.definition
        print("   from_crs definition:", t.definition)
        if note:
            print("   ", note, "->", defn)
        for ln in lines:
            print("   ", ln)
        steps = _steps(defn)
        for s in steps:
            STEP_COUNTS[s] = STEP_COUNTS.get(s, 0) + 1
        pts, b = _sample(src, dst, box, n, rng)
        ref_t = Transformer.from_pipeline(defn)
        ref = np.stack(ref_t.transform(pts[:, 0], pts[:, 1], pts[:, 2]), -1)
        ref_fin = np.isfinite(ref[:, 0])
        # pipeline.py as delivered (no integ adapters) / raw adapter / generic adapter
        status = {}
        for mode in ("delivered", "raw", "generic"):
            if mode == "delivered":
                IR.uninstall()
            else:
                IR.install(mode)
            try:
                T = PL.create(defn)
                o = T.forward(torch.tensor(pts[:200])).numpy()
                e = _err(kind, o, ref[:200])
                status[mode] = "ok (%.1e m on 200 pts)" % np.nanmax(e) if np.isfinite(e).any() else "ok"
            except Exception as exc:
                status[mode] = "%s: %s" % (type(exc).__name__, str(exc)[:70])
        print("    pipeline.py as delivered: %s | integ raw: %s | integ generic: %s" % (
            status["delivered"], status["raw"], status["generic"]))
        IR.install("generic")
        try:
            T = PL.create(defn)
        except Exception as exc:
            print("    UNSUPPORTED:", type(exc).__name__, exc)
            rows.append((label, "UNSUPPORTED " + str(exc)[:60]))
            check("e2e " + label, False, str(exc))
            continue
        t0 = time.time()
        out = T.forward(torch.tensor(pts)).numpy()
        dt = time.time() - t0
        out_fin = np.isfinite(out[:, 0])
        e64 = _err(kind, out, ref)[ref_fin & out_fin]
        mism = int((out_fin != ref_fin).sum())
        o32 = T.forward(torch.tensor(pts, dtype=torch.float32)).double().numpy()
        f32 = np.isfinite(o32[:, 0])
        e32 = _err(kind, o32, ref)[ref_fin & f32]
        mism32 = int((f32 != ref_fin).sum())
        pts32 = pts.astype(np.float32).astype(np.float64)
        ref_r = np.stack(ref_t.transform(pts32[:, 0], pts32[:, 1], pts32[:, 2]), -1)
        floor32 = np.nanmax(_err(kind, ref_r, ref)[ref_fin & np.isfinite(ref_r[:, 0])])
        # inverse: PROJ forward output -> torch inverse -> compare to the input
        src_kind = "deg6378137" if _crs(src).is_geographic else "xy"
        inv = T.inverse(torch.tensor(np.where(ref_fin[:, None], ref, 0.0))).numpy()
        ie = _err(src_kind, inv, pts)[ref_fin]
        inv_ref = np.stack(ref_t.transform(*[np.where(ref_fin, ref[:, k], 0.0) for k in range(3)],
                                           direction="INVERSE"), -1)
        ie_proj = _err(src_kind, inv_ref, pts)[ref_fin]
        # vs Transformer.from_crs().transform (operation chosen per point by PROJ)
        fc = np.stack(t.transform(pts[:, 0], pts[:, 1], pts[:, 2]), -1)
        efc = _err(kind, out, fc)[ref_fin & out_fin & np.isfinite(fc[:, 0])]
        # gradients through the whole pipeline (fp64, 10k points)
        gi = torch.tensor(pts[:10000], requires_grad=True)
        go = T.forward(gi)
        torch.where(torch.isfinite(go), go, torch.zeros_like(go)).sum().backward()
        gbad = int((~torch.isfinite(gi.grad)).any(-1).sum())
        failed = ~torch.isfinite(go[:, 0]).detach()
        gfail_nz = int((gi.grad[failed] != 0).any(-1).sum())
        print("    steps: %s" % " -> ".join(("inv " if i else "") + s for s, i in steps))
        print("    sample box %s; PROJ failures %d/%d" % (np.round(b, 2).tolist(), int((~ref_fin).sum()), n))
        print("    fp64: max %.2e m  p99 %.2e m  NaN-mask mismatches %d   (%.2f s single thread)" % (
            e64.max(), np.percentile(e64, 99), mism, dt))
        print("    fp32: max %.2e m  p99 %.2e m  NaN-mask mismatches %d   (input rounding alone: %.2e m)" % (
            e32.max(), np.percentile(e32, 99), mism32, floor32))
        print("    inverse (PROJ fwd output -> torch inv, vs original input): max %.2e m (PROJ's own round trip %.2e m)" % (
            np.nanmax(ie), np.nanmax(ie_proj)))
        print("    vs from_crs().transform (per-point operation choice): max %.2e m, %d pts > 1 mm" % (
            efc.max() if efc.size else float("nan"), int((efc > 1e-3).sum())))
        print("    grad (10k pts): non-finite grads %d, failed points with non-zero grad %d" % (gbad, gfail_nz))
        rows.append((label, "fp64 %.1e / fp32 %.1e m, mism %d/%d, inv %.1e m, grad-bad %d" % (
            e64.max(), e32.max(), mism, mism32, np.nanmax(ie), gbad)))
        check("e2e " + label, e64.max() < 1e-3 and mism == 0 and gbad == 0, rows[-1][1])
    print("== E2E summary ==")
    for r in rows:
        print("   %-28s %s" % r)


def section_steps():
    print("== 3. step inventory over the 10 exported pipelines (always_xy=True) ==")
    for (name, inv), c in sorted(STEP_COUNTS.items(), key=lambda kv: -kv[1]):
        print("    %-12s %-4s x%d" % (name, "inv" if inv else "", c))
    extra = {}
    for label, src, dst, box, kind in PAIRS:
        for always in (True, False):
            g = TransformerGroup(_crs(src), _crs(dst), always_xy=always)
            for t in g.transformers:
                for s in _steps(t.definition):
                    extra.setdefault(s[0] if always else s[0] + " (always_xy=False only)" if s[0] == "axisswap" else s[0],
                                     set()).add(label)
            for op in g.unavailable_operations:
                extra.setdefault("(grid) " + ",".join(gr.short_name for gr in op.grids), set()).add(label)
    print("    all steps in all TransformerGroup candidates (incl. always_xy=False):")
    for k in sorted(extra):
        print("      %-40s in %d pair(s)" % (k, len(extra[k])))


# =============================================================================
# 4. Autograd Jacobian vs PROJ's numerical factors (factors.cpp / deriv.cpp)
# =============================================================================
FACTOR_CASES = [
    ("lcc WRF sphere", "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=45 +lon_0=10 +R=6370000", (-40, 60), (15, 80)),
    ("lcc CONUS WGS84", "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +ellps=WGS84", (-140, -50), (10, 65)),
    ("utm 32N", "+proj=utm +zone=32 +ellps=WGS84", (3, 15), (0, 84)),
    ("tmerc exact wide", "+proj=tmerc +lon_0=10 +ellps=WGS84", (-30, 50), (-80, 80)),
    ("tmerc approx", "+proj=tmerc +lon_0=10 +ellps=WGS84 +approx", (4, 16), (-80, 80)),
    ("stere EPSG:3413", "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +ellps=WGS84", (-180, 180), (30, 89.99)),
    ("stere oblique", "+proj=stere +lat_0=45 +lon_0=10 +k=0.9999 +ellps=WGS84", (-30, 50), (10, 80)),
    ("laea EPSG:3035", "+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80", (-35, 45), (25, 84)),
    ("laea sphere", "+proj=laea +lat_0=45 +lon_0=10 +R=6371000", (-60, 80), (-20, 89)),
]


def _jacobian(P, lp):
    """[N,2,2] d(x,y)/d(lam,phi) in metres/rad. torch.func first, autograd fallback."""
    def f(p):
        return P.forward(p.unsqueeze(0)).squeeze(0)[:2]
    try:
        J = torch.func.vmap(torch.func.jacrev(f))(lp)
        return J, "vmap(jacrev)"
    except Exception as exc:
        how = "autograd (vmap failed: %s)" % type(exc).__name__
    x = lp.clone().requires_grad_(True)
    out = P.forward(x)
    gx, = torch.autograd.grad(out[:, 0].sum(), x, retain_graph=True)
    gy, = torch.autograd.grad(out[:, 1].sum(), x)
    return torch.stack([gx, gy], 1), how


def _proj_factors(der, phi, es):
    """factors.cpp:91-125 on unit-ellipsoid derivatives der = (x_l, x_p, y_l, y_p)."""
    x_l, x_p, y_l, y_p = der
    cosphi = np.cos(phi)
    h = np.hypot(x_p, y_p)
    k = np.hypot(x_l, y_l) / cosphi
    if es != 0:
        t = 1 - es * np.sin(phi) ** 2
        nn = np.sqrt(t)
        h = h * t * nn / (1 - es)
        k = k * nn
        r = t * t / (1 - es)
    else:
        r = 1.0
    conv = -np.arctan2(x_p, y_p)
    s = (y_p * x_l - x_p * y_l) * r / cosphi
    t2 = k * k + h * h
    A = np.sqrt(t2 + 2 * s)
    B = np.sqrt(np.maximum(t2 - 2 * s, 0))
    a_, b_ = 0.5 * (A + B), 0.5 * (A - B)
    omega = 2 * np.arcsin(np.clip((a_ - b_) / (a_ + b_), -1, 1))
    return dict(h=h, k=k, s=s, conv=np.degrees(conv), omega=np.degrees(omega))


def section_factors(n=1000):
    print("== 4. autograd Jacobian -> h, k, areal scale, convergence vs pyproj get_factors (PROJ numeric, h=1e-5) ==")
    rng = np.random.default_rng(7)
    for label, defn, lon, lat in FACTOR_CASES:
        params = PL.ParamList(PL.proj_tokenize(defn)).as_dict()
        P = IR.AVAILABLE[params["proj"]](params)
        lo, la = _rand_lonlat(n, lon, lat, rng)
        lp = torch.tensor(np.radians(np.stack([lo, la], -1)))
        J, how = _jacobian(P, lp)
        J = J.detach().numpy() / P.a  # unit ellipsoid, like pj_deriv on P->fwd
        der_ad = (J[:, 0, 0], J[:, 0, 1], J[:, 1, 0], J[:, 1, 1])
        # torch forward with PROJ's own 4-point stencil (deriv.cpp:8-70), h = 1e-5
        hh = 1e-5
        def fw(dl, dp):
            q = lp + torch.tensor([dl, dp], dtype=torch.float64)
            return P.forward(q)[:, :2].numpy() / P.a
        a1, a2, a3, a4 = fw(hh, hh), fw(hh, -hh), fw(-hh, -hh), fw(-hh, hh)
        der_fd = ((a1[:, 0] + a2[:, 0] - a3[:, 0] - a4[:, 0]) / (4 * hh), (a1[:, 0] - a2[:, 0] - a3[:, 0] + a4[:, 0]) / (4 * hh),
                  (a1[:, 1] + a2[:, 1] - a3[:, 1] - a4[:, 1]) / (4 * hh), (a1[:, 1] - a2[:, 1] - a3[:, 1] + a4[:, 1]) / (4 * hh))
        f = pyproj.Proj(defn).get_factors(lo, la)
        ref = dict(h=np.asarray(f.meridional_scale), k=np.asarray(f.parallel_scale), s=np.asarray(f.areal_scale),
                   conv=np.asarray(f.meridian_convergence), omega=np.asarray(f.angular_distortion))
        rder = (np.asarray(f.dx_dlam), np.asarray(f.dx_dphi), np.asarray(f.dy_dlam), np.asarray(f.dy_dphi))
        phi = np.radians(la)
        mine = _proj_factors(der_ad, phi, P.es)
        fdm = _proj_factors(der_fd, phi, P.es)
        ok = np.isfinite(ref["h"])
        rel = lambda a, b: np.max(np.abs(a[ok] - b[ok]) / np.abs(b[ok]))
        scale = np.hypot(np.hypot(rder[0], rder[1]), np.hypot(rder[2], rder[3]))[ok]
        dmax = max(np.max(np.abs(der_ad[i][ok] - rder[i][ok]) / scale) for i in range(4))
        dfd = max(np.max(np.abs(der_fd[i][ok] - rder[i][ok]) / scale) for i in range(4))
        dad_fd = max(np.max(np.abs(der_ad[i][ok] - der_fd[i][ok]) / scale) for i in range(4))
        print("  %-17s [%s] n=%d" % (label, how, int(ok.sum())))
        print("     autograd vs PROJ: rel h %.1e  k %.1e  areal %.1e | abs conv %.1e deg  omega %.1e deg | partials %.1e" % (
            rel(mine["h"], ref["h"]), rel(mine["k"], ref["k"]), rel(mine["s"], ref["s"]),
            np.max(np.abs(mine["conv"] - ref["conv"])[ok]), np.max(np.abs(mine["omega"] - ref["omega"])[ok]), dmax))
        print("     torch fwd + PROJ stencil vs PROJ: partials %.1e (forward equality) | autograd vs that FD: %.1e (FD truncation)"
              % (dfd, dad_fd))
        print("     conformal check (autograd): max |h/k-1| %.1e, max omega %.1e deg;  PROJ's own: max |h/k-1| %.1e, omega %.1e deg"
              % (np.max(np.abs(mine["h"] / mine["k"] - 1)[ok]), np.max(mine["omega"][ok]),
                 np.max(np.abs(ref["h"] / ref["k"] - 1)[ok]), np.max(ref["omega"][ok])))
        print("     equal-area check: max |areal-1| autograd %.1e, PROJ %.1e" % (
            np.max(np.abs(mine["s"] - 1)[ok]), np.max(np.abs(ref["s"] - 1)[ok])))
        check("factors " + label, rel(mine["h"], ref["h"]) < 1e-6 and rel(mine["k"], ref["k"]) < 1e-6
              and rel(mine["s"], ref["s"]) < 1e-6)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--sections", default="reg,e2e,steps,factors")
    a = ap.parse_args()
    t0 = time.time()
    IR.install("generic")
    secs = a.sections.split(",")
    if "reg" in secs:
        section_registry()
    if "e2e" in secs or "steps" in secs:
        section_e2e(a.n)
    if "steps" in secs:
        section_steps()
    if "factors" in secs:
        section_factors()
    print("== %d failed check(s) in %.1f s ==" % (len(FAIL), time.time() - t0))
    for f in FAIL:
        print("   ", f)
    sys.exit(len(FAIL))
