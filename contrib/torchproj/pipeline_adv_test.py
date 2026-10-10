"""Adversarial verification of pipeline.py against PROJ (pyproj 3.8.0 / PROJ 9.8.1).

Run: cd contrib/torchproj && python pipeline_adv_test.py [-v] [--no-dynamo]
Every check prints PASS/FAIL. A FAIL tagged [DEFECT] is a confirmed deviation of the
port from PROJ; untagged checks re-derive the implementer's headline claims.
Exit status = number of failed checks (0 = no defect found).

Raw-value comparisons: pyproj converts angular I/O itself depending on `radians=`; we
pick the mode in which pyproj passes the numbers through unchanged (see `ref`).
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
RESULTS = []
HP = math.pi / 2
D4978 = "proj=pipeline step proj=axisswap order=2,1 step proj=unitconvert xy_in=deg xy_out=rad step proj=cart ellps=WGS84"
D4277 = ("proj=pipeline step proj=axisswap order=2,1 step proj=unitconvert xy_in=deg xy_out=rad step proj=push v_3 "
         "step proj=cart ellps=WGS84 step inv proj=helmert x=446.448 y=-125.157 z=542.06 rx=0.15 ry=0.247 "
         "rz=0.842 s=-20.489 convention=position_vector step inv proj=cart ellps=airy step proj=pop v_3 "
         "step proj=unitconvert xy_in=rad xy_out=deg step proj=axisswap order=2,1")


def check(name, ok, detail="", defect=False):
    RESULTS.append((name, bool(ok), detail, defect))
    if VERBOSE or not ok:
        tag = "PASS" if ok else ("FAIL [DEFECT]" if defect else "FAIL")
        print("  %-14s %s %s" % (tag, name, ("| " + detail) if detail else ""))


# ----------------------------------------------------------------------------- helpers
def ref(defn, pts, inverse=False):
    """PROJ output on raw numbers (inf -> nan)."""
    pts = np.asarray(pts, dtype=float)
    T = P.create(defn)
    fwd = not inverse
    rad = T.angular_input(fwd) or T.angular_output(fwd)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = Transformer.from_pipeline(defn)
    d = TransformDirection.INVERSE if inverse else TransformDirection.FORWARD
    out = t.transform(*[pts[:, i].copy() for i in range(pts.shape[1])], direction=d, errcheck=False, radians=rad)
    o = np.stack([np.asarray(c) for c in out], -1)
    o[np.isinf(o)] = np.nan
    return o


def mine(defn, pts, inverse=False, dtype=torch.float64):
    T = P.create(defn)
    x = torch.tensor(np.asarray(pts, dtype=float), dtype=dtype)
    return (T.inverse(x) if inverse else T.forward(x)).double().numpy()


def diff(defn, pts, inverse=False, ang_scale=None):
    """(max abs diff in metres-ish units, n nan-mask mismatches)."""
    r, m = ref(defn, pts, inverse), mine(defn, pts, inverse)
    nr, nm = np.isnan(r[:, :3]).any(-1), np.isnan(m[:, :3]).any(-1)
    ok = ~nr & ~nm
    d = np.abs(r - m)[:, :3]
    if ang_scale is not None:
        d = d * np.asarray(ang_scale)
    return (float(d[ok].max()) if ok.any() else 0.0), int((nr != nm).sum()), r, m


# ============================================================================= 1. claims
def section_claims():
    print("== 1. re-derived headline numbers (own random points) ==")
    rng = np.random.default_rng(2024)
    n = 2000
    pts = np.stack([rng.uniform(-90, 90, n), rng.uniform(-180, 180, n), rng.uniform(-500, 9000, n)], -1)
    pts[:6] = [[90, 0, 0], [-90, 0, 0], [90, 180, 10], [-90, -180, 0], [0, 180, 0], [0, -180, 0]]
    r = ref(D4978, pts)
    for dt, tol in ((torch.float64, 1e-8), (torch.float32, 3.0)):
        m = mine(D4978, pts, dtype=dt)
        e = np.sqrt(((r - m) ** 2).sum(-1)).max()
        ri, mi = ref(D4978, r, True), mine(D4978, r, True, dtype=dt)
        ei = np.hypot(angular_error_m(ri[:, 1], ri[:, 0], mi[:, 1], mi[:, 0]), ri[:, 2] - mi[:, 2]).max()
        check("4326->4978 %s fwd %.3g m inv %.3g m" % (str(dt)[6:], e, ei), e < tol and ei < tol)
    defs = {
        "4326->4277 helmert": (D4277, np.stack([rng.uniform(49.9, 60.8, n), rng.uniform(-8.6, 1.8, n), rng.uniform(0, 500, n)], -1),
                               [111319.5, 111319.5, 1.0]),
        "7p helmert cf exact": ("+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=push +v_3 +step +proj=cart "
                                "+ellps=GRS80 +step +proj=helmert +x=-81.07 +y=-89.36 +z=-115.75 +rx=-0.48 +ry=-0.02 +rz=-0.41 "
                                "+s=-0.54 +convention=coordinate_frame +exact +step +inv +proj=cart +ellps=intl +step +proj=pop +v_3 "
                                "+step +proj=unitconvert +xy_in=rad +xy_out=deg",
                                np.stack([rng.uniform(-180, 180, n), rng.uniform(-89.9, 89.9, n), rng.uniform(-100, 3000, n)], -1),
                                [111319.5, 111319.5, 1.0]),
        "15p helmert 4D": ("+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=cart +ellps=GRS80 "
                           "+step +proj=helmert +convention=position_vector +x=0.0127 +dx=-0.0029 +rx=-0.00039 +drx=-0.00011 "
                           "+y=0.0065 +dy=-0.0002 +ry=0.00080 +dry=-0.00019 +z=-0.0209 +dz=-0.0006 +rz=-0.00114 +drz=0.00007 "
                           "+s=0.00195 +ds=0.00001 +t_epoch=1988.0 +exact +step +inv +proj=cart +ellps=GRS80 "
                           "+step +proj=unitconvert +xy_in=rad +xy_out=deg",
                           np.stack([rng.uniform(-180, 180, n), rng.uniform(-89, 89, n), rng.uniform(-100, 3000, n),
                                     rng.uniform(1980, 2030, n)], -1), [111319.5, 111319.5, 1.0]),
        "molodensky": ("+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=molodensky +a=6378160 "
                       "+rf=298.25 +da=-23 +df=-8.120449e-8 +dx=-134 +dy=-48 +dz=149 +step +proj=unitconvert +xy_in=rad +xy_out=deg",
                       np.stack([rng.uniform(-180, 180, n), rng.uniform(-89, 89, n), rng.uniform(-100, 3000, n)], -1),
                       [111319.5, 111319.5, 1.0]),
        "towgs84 cs2cs": ("+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=longlat +ellps=intl "
                          "+towgs84=-81.0703,-89.3603,-115.7526,0.48488,0.02436,0.41321,-0.540645 +inv "
                          "+step +proj=unitconvert +xy_in=rad +xy_out=deg",
                          np.stack([rng.uniform(-180, 180, n), rng.uniform(-89, 89, n), rng.uniform(-100, 3000, n)], -1),
                          [111319.5, 111319.5, 1.0]),
    }
    for name, (d, p, sc) in defs.items():
        e, nm, r, _ = diff(d, p, False, sc)
        ei, nmi, _, _ = diff(d, r, True, sc)
        check("%s fp64 fwd %.3g inv %.3g (m)" % (name, e, ei), e < 1e-6 and ei < 1e-6 and nm == 0 and nmi == 0)
    for g, lon, lat in (("ntf_r93.gsb", (-5.4, 9.4), (41.1, 51.9)), ("ntv2_0_downsampled.gsb", (-141, -45), (41, 83))):
        d = ("+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=hgridshift +grids=%s "
             "+step +proj=unitconvert +xy_in=rad +xy_out=deg" % g)
        p = np.stack([rng.uniform(*lon, n), rng.uniform(*lat, n)], -1)
        e, nm, r, _ = diff(d, p)
        ok = ~np.isnan(r).any(-1)
        ei, nmi, _, _ = diff(d, r[ok], True)
        check("hgridshift %s bit-exact fwd %.3g inv %.3g, nan-mask mismatches %d/%d" % (g, e, ei, nm, nmi),
              e == 0 and ei == 0 and nm == 0 and nmi == 0)


# ============================================================================= 2. gie
def section_gie():
    """Every gie accept point of blocks using only pipeline.py ops: port vs PROJ (proj_create path)."""
    print("== 2. gie accept points: port vs pyproj ==")

    def coord(args):
        v = []
        for t in args.split()[:4]:
            try:
                v.append(math.inf if t.upper() == "HUGE_VAL" else testing.parse_num(t.replace("_", "")))
            except ValueError:
                break
        return v
    n_blocks = n_pts = n_bad = 0
    for f in sorted(testing.GIE_DIR.glob("*.gie")):
        T = Tp = None
        fwd, accept = True, None
        for cmd, args in testing._logical_lines(f):
            if cmd == "operation":
                fwd, T = True, None
                toks = P.proj_tokenize(args)
                names = [t[5:] for t in toks if t.startswith("proj=")]
                if not names or any(nm not in P.REGISTRY and nm != "pipeline" for nm in names) or \
                        any(t.startswith(("init=", "geoidgrids=")) for t in toks):
                    continue
                try:
                    T = P.create(args)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        Tp = Transformer.from_pipeline(args)
                except Exception:
                    T = None
                    continue
                n_blocks += 1
            elif cmd == "accept":
                accept = coord(args)
            elif T is None:
                continue
            elif cmd == "direction":
                fwd = not (args.strip()[:1].lower() in "ir")
            elif cmd == "expect" and accept and len(accept) >= 2:
                if (T.angular_input(fwd) and T.degree_output(fwd)) or (T.degree_input(fwd) and T.angular_output(fwd)):
                    continue  # pyproj cannot be asked for raw numbers here
                a = list(accept) + [0.0] * (4 - len(accept))
                if T.angular_input(fwd):
                    a[0], a[1] = math.radians(a[0]), math.radians(a[1])
                m = (T.forward if fwd else T.inverse)(torch.tensor([a], dtype=torch.float64))[0].numpy()
                rad = T.angular_input(fwd) or T.angular_output(fwd)
                r = np.array(Tp.transform(*[np.array([v]) for v in a], errcheck=False, radians=rad,
                                          direction=TransformDirection.FORWARD if fwd else TransformDirection.INVERSE)).ravel()
                r[np.isinf(r)] = np.nan
                n_pts += 1
                sc = np.array([6378137.0, 6378137.0, 1.0]) if T.angular_output(fwd) else (
                    np.array([111319.5, 111319.5, 1.0]) if T.degree_output(fwd) else np.ones(3))
                bad = np.isnan(r[:3]).any() != np.isnan(m[:3]).any() or \
                    (not np.isnan(r[:3]).any() and np.abs((r[:3] - m[:3]) * sc).max() > 1e-6)
                n_bad += bool(bad)
                if bad and VERBOSE:
                    print("     mismatch", f.name, args[:80], a, r, m)
    check("gie: %d blocks / %d expect points with only pipeline.py ops agree with PROJ to 1e-6 m (%d bad)"
          % (n_blocks, n_pts, n_bad), n_bad == 0)


# ============================================================================= 3. edge cases
def section_edges():
    print("== 3. edge cases vs pyproj ==")
    cart = "+proj=cart +ellps=GRS80"
    # poles
    _, nm, r, m = diff(cart, [[0.3, HP, 0], [0.3, -HP, 0], [0.3, HP + 0.5e-12, 0], [0.3, HP + 2e-12, 0],
                              [0.3, HP - 1e-12, 0], [0.3, -HP + 1e-12, 0], [0.3, -HP - 2e-12, 0]])
    check("north pole / exact poles / beyond-eps latitudes: same values and failures", nm == 0)
    _, nm, r, m = diff(cart, [[0.3, -HP - 1e-15, 0], [0.3, -HP - 0.5e-12, 0]])
    check("south pole within PJ_EPS_LAT below -90 deg is clamped by PROJ 9.8.1 (port: NaN)", nm == 0,
          "pyproj %s / port %s" % (r[:, 2].tolist(), m[:, 2].tolist()), defect=True)
    dd = "+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=cart +ellps=GRS80"
    _, nm, r, m = diff(dd, [[10, -90.00000000000001, 0], [10, -90 - 5e-11, 0]])
    check("lat=-90.00000000000001 deg through deg pipeline (port NaN, PROJ -90)", nm == 0,
          "pyproj z %s port z %s" % (r[:, 2].tolist(), m[:, 2].tolist()), defect=True)
    # antimeridian / lon range / over / lon_0
    lons = [[math.pi, 0.5, 0], [-math.pi, 0.5, 0], [math.pi + 1e-12, 0.5, 0], [math.pi + 1e-9, 0.5, 0], [3.5, 0.5, 0],
            [9.99, 0.5, 0], [10.01, 0.5, 0], [-10.01, 0.5, 0], [-9.99, -0.5, 0]]
    for extra in ("", "+over", "+lon_0=180", "+lon_0=-180 +over", "+lon_0=33"):
        e, nm, _, _ = diff(cart + " " + extra, lons)
        check("antimeridian / |lon|>pi / |lon|>10 with '%s': %.3g m, %d nan mismatches" % (extra, e, nm), e < 1e-6 and nm == 0)
    # ellipsoids, units, offsets on core ops
    rng = np.random.default_rng(5)
    geo = np.stack([rng.uniform(-math.pi, math.pi, 300), rng.uniform(-1.5, 1.5, 300), rng.uniform(-500, 9000, 300)], -1)
    for extra in ("+R=6371000", "+a=6378137 +rf=298.257223563", "+a=6378137 +b=6356752.314245", "+a=6378137 +es=0.00669438",
                  "+a=6378137 +e=0.0818191908426", "+ellps=sphere", "+ellps=clrk66", "+ellps=airy +R_A", "+ellps=WGS84 +R_h",
                  "+datum=OSGB36", "+ellps=GRS80 +units=km", "+ellps=GRS80 +to_meter=1/3", "+ellps=GRS80 +z_0=100",
                  "+ellps=GRS80 +pm=paris", "+ellps=GRS80 +geoc", "+ellps=GRS80 +axis=nwu", "+ellps=GRS80 +x_0=1000 +y_0=-2000",
                  "+ellps=GRS80 +k_0=0.5"):
        e, nm, r, _ = diff("+proj=cart " + extra, geo)
        ei, nmi, _, _ = diff("+proj=cart " + extra, r, True, [6378137.0, 6378137.0, 1.0])
        check("cart %s fwd %.3g inv %.3g" % (extra, e, ei), e < 1e-6 and ei < 1e-6 and nm == 0 and nmi == 0)
    for d in ("+proj=latlong +ellps=GRS80 +pm=paris +lon_0=3", "+proj=latlong +ellps=GRS80 +lon_wrap=180",
              "+proj=latlong +ellps=GRS80 +z_0=10 +vunits=ft", "+proj=latlong +datum=OSGB36 +axis=neu",
              "+proj=geocent +R=6371000 +units=km"):
        e, nm, _, _ = diff(d, geo, False, None if "geocent" in d else [6378137.0, 6378137.0, 1.0])
        check("%s fwd %.3g" % (d, e), e < 1e-6 and nm == 0)
    e, _, _, _ = diff("+proj=geocent +no_defs", geo)
    check("+no_defs: no default ellps=GRS80 (PROJ falls back to WGS84) -> %.3g m" % e, e < 1e-6, defect=True)
    try:
        diff("+proj=geocent +ellps=intl +towgs84=-87,-98,-121", geo[:3])
        ok, det = True, ""
    except Exception as exc:
        ok, det = False, "%s: %s" % (type(exc).__name__, exc)
    check("geocent with non-zero towgs84 does not crash at transform time", ok, det, defect=True)
    # single-op +inv (proj_create honours it)
    e, nm, r, m = diff("+proj=helmert +x=100 +inv", [[1.0, 2.0, 3.0]])
    check("single (non-pipeline) op with +inv is applied inverted", e < 1e-9 and nm == 0,
          "pyproj x=%s port x=%s" % (r[0, 0], m[0, 0]), defect=True)
    e, nm, r, m = diff("+proj=unitconvert +xy_in=km +xy_out=m +inv", [[1.0, 2.0, 3.0]])
    check("'+proj=unitconvert +xy_in=km +xy_out=m +inv'", e < 1e-9, "pyproj %s port %s" % (r[0, :2], m[0, :2]), defect=True)
    # proj_create pipeline normalisation (noop removed by PROJ)
    d = ("+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=noop +lon_0=10 "
         "+step +proj=unitconvert +xy_in=rad +xy_out=deg")
    e, _, r, m = diff(d, [[-170.0, 10.0, 0.0]])
    check("noop step with generic params (PROJ's proj_create drops it)", e < 1e-9, "pyproj %s port %s" % (r[0, 0], m[0, 0]),
          defect=True)
    # NaN in t: PROJ returns all NaN
    _, nm, r, m = diff(D4277, [[52, -1, 10, math.nan]])
    check("NaN in t -> PROJ makes the whole coordinate NaN", nm == 0, "port %s" % m[0].tolist(), defect=True)
    # null grid with out-of-range input
    _, nm, r, m = diff("+proj=hgridshift +grids=null", [[0.1, HP + 0.3], [12.0, 0.2]], True)
    check("null grid inverse with |lat|>90 deg or |lon|>pi passes through in PROJ", nm == 0, "port %s" % m.tolist(), defect=True)
    _, nm, r, m = diff("+proj=hgridshift +grids=null +over", [[9.9, 0.1], [-9.9, 0.1]])
    check("null grid forward with +over and |lon|>3pi passes through in PROJ", nm == 0, "port %s" % m.tolist(), defect=True)
    # hgridshift edges of every subgrid (incl. eps band) for all NTv2 test files
    for g in ("BETA2007.gsb", "test_hgrid_big_endian.gsb", "ntv2_0_downsampled.gsb"):
        nodes = P.create("+proj=hgridshift +grids=" + g).steps[0].op.nodes
        pts = []
        for nd in nodes:
            for lon in (nd.west, nd.east, nd.west - nd.eps * 0.5, nd.east + nd.eps * 0.5, nd.west - nd.eps * 1.5,
                        nd.east + nd.eps * 1.5, nd.west + nd.resX * (1 - 9e-5), nd.east - nd.resX * 9e-5):
                for lat in (nd.south, nd.north, nd.south - nd.eps * 0.5, nd.north + nd.eps * 0.5, 0.5 * (nd.south + nd.north)):
                    pts.append([lon, lat])
        e, nm, _, _ = diff("+proj=hgridshift +grids=" + g, pts)
        ei, nmi, _, _ = diff("+proj=hgridshift +grids=" + g, pts, True)
        check("hgridshift %s subgrid edges/eps band: fwd %.3g inv %.3g, nan mismatches %d/%d" % (g, e, ei, nm, nmi),
              e == 0 and ei == 0 and nm == 0 and nmi == 0)
    # pipeline driver semantics
    geo2 = [[0.1, 0.7, 100.0], [-2.0, -0.5, 0.0]]
    for d, inv in (("+proj=pipeline +inv +step +inv +proj=cart +ellps=GRS80", False),
                   ("+proj=pipeline +ellps=intl +step +proj=cart +step +inv +proj=cart +ellps=GRS80", False),
                   ("+proj=pipeline +step +proj=cart +ellps=GRS80 +step +proj=helmert +x=100 +omit_fwd +step +inv +proj=cart +ellps=GRS80", True),
                   ("+proj=pipeline +step +proj=cart +ellps=GRS80 +step +proj=helmert +x=100 +omit_inv +step +inv +proj=cart +ellps=GRS80", True),
                   ("+proj=pipeline +step +proj=push +v_1 +v_2 +step +proj=cart +ellps=GRS80 +step +proj=helmert +x=100 +step +inv "
                    "+proj=cart +ellps=GRS80 +step +proj=pop +v_1 +v_2", False),
                   ("+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +lon_wrap=180 +step +proj=unitconvert "
                    "+xy_in=rad +xy_out=deg", False)):
        pts = geo2 if "lon_wrap" not in d else [[-170.0, 10, 0]]
        e, nm, _, _ = diff(d, pts, inv)
        check("driver: %s (%s)" % (d[:70], "inv" if inv else "fwd"), e < 1e-6 and nm == 0)
    # projection steps (ProjectionStep wrapper) with generic PROJ parameters
    pts = np.stack([np.radians(rng.uniform(3, 15, 200)), np.radians(rng.uniform(-60, 70, 200)), rng.uniform(0, 100, 200)], -1)
    base = "+proj=tmerc +lon_0=9 +ellps=GRS80 "
    for extra in ("+k_0=0.9996 +x_0=500000 +y_0=-100", "+datum=OSGB36"):
        e, nm, _, _ = diff(base + extra, pts)
        check("projection step %s: %.3g m" % (extra, e), e < 1e-6 and nm == 0)
    for extra in ("+pm=paris", "+axis=wsu", "+to_meter=0.3048", "+geoc", "+vunits=ft +z_0=10", "+R_A"):
        e, nm, _, _ = diff(base + extra, pts)
        check("projection step silently ignores %s: %.3g m off" % (extra, e), e < 1e-6 and nm == 0, defect=True)
    _, nm, r, m = diff(base, [[10.5, 0.5, 0], [-10.5, 0.5, 0]])
    check("projection step: |lon| > 10 rad must fail (PROJ fwd_prepare)", nm == 0, "port %s" % m[:, :2].tolist(), defect=True)
    # real database pipelines that put +pm / +axis inside the projection step
    d27572 = ("proj=pipeline step proj=axisswap order=2,1 step proj=unitconvert xy_in=deg xy_out=rad step proj=push v_3 "
              "step proj=cart ellps=WGS84 step proj=helmert x=168 y=60 z=-320 step inv proj=cart ellps=clrk80ign step proj=pop v_3 "
              "step proj=lcc lat_1=46.8 lat_0=46.8 lon_0=0 k_0=0.99987742 x_0=600000 y_0=2200000 ellps=clrk80ign pm=paris")
    d22275 = ("proj=pipeline step proj=axisswap order=2,1 step proj=unitconvert xy_in=deg xy_out=rad step proj=tmerc axis=wsu "
              "lat_0=0 lon_0=15 k=1 x_0=0 y_0=0 a=6378249.145 rf=293.4663077")
    for name, d, la, lo in (("EPSG:4326->27572 (NTF Paris Lambert II)", d27572, (43, 50), (-4, 7)),
                            ("EPSG:4326->22275 (Lo15, axis=wsu)", d22275, (-33, -25), (14, 16))):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from pyproj.transformer import TransformerGroup
            dst = name.split("->")[1].split()[0]
            same = any(t.definition == d for t in TransformerGroup("EPSG:4326", "EPSG:" + dst).transformers)
        p = np.stack([rng.uniform(*la, 200), rng.uniform(*lo, 200)], -1)
        e, nm, _, _ = diff(d, p)
        check("%s [pyproj emits this string: %s]: %.3g m off" % (name, same, e), e < 1e-6, defect=True)
    # shapes, passthrough, dtypes
    T = P.create(D4277)
    torch.manual_seed(0)
    x = torch.stack([torch.empty(2, 3, 5).uniform_(50, 58), torch.empty(2, 3, 5).uniform_(-6, 1)], -1).double()
    y = T.forward(x)
    check("[B,H,W,2] batch equals flattened", tuple(y.shape) == (2, 3, 5, 2) and torch.equal(y.reshape(-1, 2), T.forward(x.reshape(-1, 2))))
    x4 = torch.cat([x.reshape(-1, 2), torch.linspace(-100, 100, 30).double()[:, None], torch.linspace(2000, 2020, 30).double()[:, None]], -1)
    y4 = T.forward(x4)
    r4 = ref(D4277, x4.numpy())
    check("[N,4] z restored by pop and t passed through; matches pyproj",
          torch.equal(y4[:, 2:], x4[:, 2:]) and np.abs(r4 - y4.numpy()).max() < 1e-9)
    yf = T.forward(x.float())
    check("float32 in -> float32 out, finite", yf.dtype == torch.float32 and bool(torch.isfinite(yf).all()))


# ============================================================================= 4. gradients
def jac_auto(f, x):
    x = x.clone().requires_grad_(True)
    y = f(x)
    J = torch.zeros(x.shape[0], y.shape[-1], x.shape[-1], dtype=torch.float64)
    for k in range(y.shape[-1]):
        g, = torch.autograd.grad(y[:, k].sum(), x, retain_graph=True)
        J[:, k, :] = g
    return J


def jac_fd(f, x, h):
    cols = []
    with torch.no_grad():
        for i in range(x.shape[-1]):
            e = torch.zeros(x.shape[-1], dtype=torch.float64)
            e[i] = h[i]
            cols.append((f(x + e) - f(x - e)) / (2 * h[i]))
    return torch.stack(cols, -1)


def section_grad():
    print("== 4. autograd Jacobian vs central differences; NaN-free gradients at singular points ==")
    torch.manual_seed(1)
    N = 30
    deg = torch.stack([torch.empty(N).uniform_(-170, 170), torch.empty(N).uniform_(-85, 85), torch.empty(N).uniform_(-100, 3000)], -1).double()
    latlon = deg[:, [1, 0, 2]].contiguous()
    cases = [("4326->4978 fwd", P.create(D4978).forward, latlon, [1e-5, 1e-5, 1e-2]),
             ("4326->4277 fwd", P.create(D4277).forward, latlon, [1e-5, 1e-5, 1e-2]),
             ("4326->4277 inv", P.create(D4277).inverse, latlon, [1e-5, 1e-5, 1e-2]),
             ("molodensky fwd", P.create("+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=molodensky "
                                         "+a=6378160 +rf=298.25 +da=-23 +df=-8.120449e-8 +dx=-134 +dy=-48 +dz=149 +step "
                                         "+proj=unitconvert +xy_in=rad +xy_out=deg").forward, deg, [1e-5, 1e-5, 1e-2]),
             ("towgs84 cs2cs inv", P.create("+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=longlat "
                                            "+ellps=intl +towgs84=-81.07,-89.36,-115.75,0.48,0.02,0.41,-0.54 +inv +step "
                                            "+proj=unitconvert +xy_in=rad +xy_out=deg").inverse, deg, [1e-5, 1e-5, 1e-2])]
    xyz = P.create(D4978).forward(latlon).detach()
    cases.append(("4326->4978 inv", P.create(D4978).inverse, xyz, [1e-2] * 3))
    fr = torch.stack([torch.empty(N).uniform_(-4, 8), torch.empty(N).uniform_(43, 50)], -1).double()
    dh = ("+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=hgridshift +grids=ntf_r93.gsb "
          "+step +proj=unitconvert +xy_in=rad +xy_out=deg")
    cases += [("hgridshift fwd", P.create(dh).forward, fr, [1e-7, 1e-7]), ("hgridshift inv", P.create(dh).inverse, fr, [1e-7, 1e-7])]
    x4 = torch.stack([torch.zeros(N), torch.zeros(N), torch.zeros(N), torch.empty(N).uniform_(1900, 2100)], -1).double()
    cases.append(("unitconvert decimalyear->gps_week", P.create("+proj=unitconvert +t_in=decimalyear +t_out=gps_week").forward,
                  x4, [1, 1, 1, 1e-6]))
    for name, f, x, h in cases:
        # FD of heights through ECEF round trips is rounding-noise limited (~1e-9 m / 2h): take the
        # better of two step sizes per entry (truncation error grows, rounding noise shrinks with h)
        Ja = jac_auto(f, x)
        Jf1, Jf2 = jac_fd(f, x, h), jac_fd(f, x, [10 * v for v in h])
        err = torch.minimum((Ja - Jf1).abs(), (Ja - Jf2).abs())
        rel = (err.amax(-1) / Jf1.abs().amax(-1).clamp_min(1e-12)).max().item()
        check("Jacobian %s: max row-relative error %.2e" % (name, rel), rel < 1e-4 and not torch.isnan(Ja).any())

    def gfinite(defn, pts, inverse=False):
        T = P.create(defn)
        x = torch.tensor(pts, dtype=torch.float64, requires_grad=True)
        y = T.inverse(x) if inverse else T.forward(x)
        g, = torch.autograd.grad(torch.where(torch.isnan(y), 0.0, y).sum(), x)
        return bool(torch.isfinite(g).all())
    b = 6356752.314140356
    sing = [("+proj=cart +ellps=GRS80", [[0.3, HP, 0], [0.3, -HP, 0], [0.3, HP + 0.5e-12, 0], [math.pi, 0.0, 0.0]], False),
            ("+proj=cart +ellps=GRS80", [[0, 0, b], [0, 0, -b], [0, 0, 0], [1e-300, 0, b], [1e-3, 0, 0]], True),
            ("+proj=molodensky +a=6378160 +rf=298.25 +da=-23 +df=-8.12e-8 +dx=-134 +dy=-48 +dz=149", [[0.2, HP, 0], [0.2, 0.0, 0]], False),
            ("+proj=geoc +ellps=GRS80", [[0.2, HP, 0], [0.2, HP - 1e-9, 0]], False),
            ("+proj=hgridshift +grids=ntf_r93.gsb", [[0.1, 0.8], [1.0, 0.8], [-0.1, 0.72]], True),
            ("+proj=helmert +x=1 +dx=1 +rx=1 +drx=1 +t_epoch=2000 +convention=position_vector", [[4e6, 1e6, 4e6, math.inf]], False),
            ("+proj=unitconvert +t_in=decimalyear +t_out=mjd", [[0, 0, 0, math.inf], [0, 0, 0, 10000.5]], False),
            (D4277, [[91.0, 0.0, 0.0], [90.0, 0.0, 0.0], [52.0, 600.0, 0.0]], False)]
    for d, pts, inv in sing:
        check("finite gradients at singular/failed points: %s %s" % (d[:50], "inv" if inv else "fwd"), gfinite(d, pts, inv))
    # shared learnable offset with one failing point in the batch
    off = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    pts = torch.tensor([[0.04, 0.81], [0.52, 0.80]], dtype=torch.float64)
    out = P.create("+proj=hgridshift +grids=ntf_r93.gsb").inverse(pts + off)
    g, = torch.autograd.grad(torch.where(torch.isnan(out), 0.0, out).sum(), off)
    check("shared parameter gradient finite with an outside-grid point in the batch", bool(torch.isfinite(g).all()))


# ============================================================================= 5. compile
def section_dynamo():
    print("== 5. torch._dynamo graph breaks (tracing only) ==")
    import torch._dynamo as dyn
    for name, fn, x in (("4277 fwd", P.create(D4277).forward, torch.tensor([[52.0, -1.0, 3.0]], dtype=torch.float64)),
                        ("ntv2 hgridshift inv", P.create("+proj=hgridshift +grids=ntv2_0_downsampled.gsb").inverse,
                         torch.tensor([[-1.44, 0.73]], dtype=torch.float64))):
        dyn.reset()
        ex = dyn.explain(fn)(x)
        check("%s: %d graph(s), %d break(s)" % (name, ex.graph_count, ex.graph_break_count), ex.graph_break_count == 0)


if __name__ == "__main__":
    t0 = time.time()
    section_claims()
    section_gie()
    section_edges()
    section_grad()
    if "--no-dynamo" not in sys.argv:
        section_dynamo()
    fails = [r for r in RESULTS if not r[1]]
    print("== summary: %d checks, %d failed (%d tagged DEFECT) in %.1fs ==" % (
        len(RESULTS), len(fails), sum(r[3] for r in fails), time.time() - t0))
    for name, _, det, defect in fails:
        print("   %s %s %s" % ("[DEFECT]" if defect else "[FAIL]  ", name, ("| " + det[:160]) if det else ""))
    sys.exit(len(fails))
