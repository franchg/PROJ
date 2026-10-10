"""Adversarial verification of lcc_stere.py (independent of lcc_stere_test.py).

  cd contrib/torchproj && python lcc_stere_adv_test.py [section ...]

Sections: claims, gie, random, edge, inv_edge, eps, setup, shapes, fp32_pole,
          grad_fd, grad_nan, scan, compile
All PROJ references come from pyproj (PROJ 9.8.1) with radians=True so that the
exact same float64 radians go into PROJ and the port.
"""
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from pyproj import Transformer
from pyproj.enums import TransformDirection

import lcc_stere as L
import testing

torch.set_num_threads(1)
D2R = L.DEG_TO_RAD
FWD, INV = TransformDirection.FORWARD, TransformDirection.INVERSE
SECTIONS = set(sys.argv[1:]) or {"gie", "random", "edge", "inv_edge", "eps", "setup", "shapes",
                                 "grad_fd", "grad_nan", "scan", "compile", "claims", "fp32_pole"}
FINDINGS = []


def finding(msg):
    FINDINGS.append(msg)
    print("  ** FINDING:", msg)


_TC = {}


def _tr(defn):
    if defn not in _TC:
        _TC[defn] = Transformer.from_pipeline(defn)
    return _TC[defn]


def proj_fwd(defn, lam, phi, z=None, t=None):
    tr = _tr(defn)
    if z is None:
        return [np.asarray(v) for v in tr.transform(np.asarray(lam, float), np.asarray(phi, float),
                                                     radians=True, direction=FWD)]
    return [np.asarray(v) for v in tr.transform(np.asarray(lam, float), np.asarray(phi, float),
                                                 np.asarray(z, float), np.asarray(t, float) if t is not None else None,
                                                 radians=True, direction=FWD)]


def proj_inv(defn, x, y, z=None, t=None):
    tr = _tr(defn)
    if z is None:
        return [np.asarray(v) for v in tr.transform(np.asarray(x, float), np.asarray(y, float),
                                                     radians=True, direction=INV)]
    return [np.asarray(v) for v in tr.transform(np.asarray(x, float), np.asarray(y, float),
                                                 np.asarray(z, float), np.asarray(t, float) if t is not None else None,
                                                 radians=True, direction=INV)]


def port_fwd(P, lam, phi, dtype=torch.float64):
    inp = torch.stack([torch.as_tensor(np.asarray(lam, float)), torch.as_tensor(np.asarray(phi, float))], -1).to(dtype)
    with torch.no_grad():
        o = P.forward(inp).double().numpy()
    return o[..., 0], o[..., 1]


def port_inv(P, x, y, dtype=torch.float64):
    inp = torch.stack([torch.as_tensor(np.asarray(x, float)), torch.as_tensor(np.asarray(y, float))], -1).to(dtype)
    with torch.no_grad():
        o = P.inverse(inp).double().numpy()
    return o[..., 0], o[..., 1]


def ang_err_m(l1, p1, l2, p2, a=6378137.0):
    """Angular error in metres; inputs in RADIANS."""
    return testing.angular_error_m(np.degrees(l1), np.degrees(p1), np.degrees(l2), np.degrees(p2), a=a)


def bad(*arrs):
    m = np.zeros(np.shape(arrs[0]), bool)
    for a in arrs:
        m |= ~np.isfinite(a)
    return m


# =============================================================================
# 1. gie, independently re-derived
# =============================================================================
def sec_gie():
    print("=" * 78)
    print("1. gie re-derivation (independent loop; PROJ itself evaluated on every case)")
    tot = passed = 0
    nblocks = 0
    setup_blocks = []
    worst_vs_proj = 0.0
    for name in ("lcc", "stere", "ups"):
        for b in testing.gie_cases(name):
            nblocks += 1
            op = b["operation"]
            try:
                P = L.make(op)
                perr = None
            except Exception as ex:  # noqa
                P, perr = None, ex
            try:
                _tr(op)
                proj_ok = True
            except Exception:
                proj_ok = False
            if not b["cases"] and not b["roundtrips"]:
                setup_blocks.append((op, P is not None, proj_ok))
                continue
            if P is None:
                finding("gie block %s: port setup raised %s" % (op, perr))
                tot += len(b["cases"]) + len(b["roundtrips"])
                continue
            for c in b["cases"]:
                tot += 1
                a = c["accept"]
                if c["direction"] == "fwd":
                    lam, phi = np.array([a[0] * D2R]), np.array([a[1] * D2R])
                    x, y = port_fwd(P, lam, phi)
                    X, Y = proj_fwd(op, lam, phi)
                    if c["expect"] == "failure":
                        ok = bool(bad(x, y)[0])
                    else:
                        e = math.hypot(x[0] - c["expect"][0], y[0] - c["expect"][1])
                        ok = e <= c["tolerance_m"]
                        worst_vs_proj = max(worst_vs_proj, math.hypot(x[0] - X[0], y[0] - Y[0]))
                else:
                    xx, yy = np.array([a[0]]), np.array([a[1]])
                    lam, phi = port_inv(P, xx, yy)
                    RL, RP = proj_inv(op, xx, yy)
                    if c["expect"] == "failure":
                        ok = bool(bad(lam, phi)[0])
                    else:
                        e = float(ang_err_m(lam, phi, np.radians(c["expect"][0]), np.radians(c["expect"][1]), a=P.a)[0])
                        ok = e <= c["tolerance_m"]
                        worst_vs_proj = max(worst_vs_proj, float(ang_err_m(lam, phi, RL, RP, a=P.a)[0]))
                passed += ok
                if not ok:
                    finding("gie case fails: %s %s %s" % (op, c["direction"], a))
            for r in b["roundtrips"]:
                tot += 1
                a = r["accept"]
                cur = torch.tensor([[a[0] * D2R, a[1] * D2R]], dtype=torch.float64)
                with torch.no_grad():
                    for _ in range(r["n"]):
                        cur = P.inverse(P.forward(cur))
                e = float(ang_err_m(cur[0, 0].numpy(), cur[0, 1].numpy(), a[0] * D2R, a[1] * D2R, a=P.a))
                ok = e <= r["tolerance_m"]
                passed += ok
                if not ok:
                    finding("gie roundtrip fails: %s %s err %.3g" % (op, a, e))
    # setup-failure blocks: compare PROJ vs port setup outcome directly
    for op, port_ok, proj_ok in setup_blocks:
        tot += 1
        same = port_ok == proj_ok
        passed += same
        print("  setup-only block: %-58s PROJ creates=%s port creates=%s %s" % (op[:58], proj_ok, port_ok,
                                                                               "" if same else "MISMATCH"))
        if not same:
            finding("setup outcome differs from PROJ for %s" % op)
        if port_ok and proj_ok:
            P = L.make(op)
            lam = np.radians(np.array([0.0, 10.0, -45.0, 120.0, 3.0]))
            phi = np.radians(np.array([0.0, 30.0, -60.0, 80.0, 2.5]))
            x, y = port_fwd(P, lam, phi)
            X, Y = proj_fwd(op, lam, phi)
            print("     values vs PROJ on 5 pts: port %s PROJ %s" % (np.c_[x, y].round(3).tolist(),
                                                                     np.c_[X, Y].round(3).tolist()))
    print("  gie: %d/%d (blocks %d); worst |port - PROJ| on gie inputs: %.2e m" % (passed, tot, nblocks, worst_vs_proj))
    return passed, tot, nblocks


# =============================================================================
# 2. random points incl. out-of-domain, fp64 and fp32; failure pattern exact
# =============================================================================
RAND_DEFS = [
    # (def, lon box, lat box) -- boxes deliberately include failing areas
    ("+proj=lcc +lat_1=30 +lat_2=60 +lat_0=45 +lon_0=10 +R=6370000", (-180, 180), (-90, 90)),
    ("+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +ellps=WGS84", (-180, 180), (-90, 90)),
    ("+proj=lcc +lat_1=-30 +lat_2=-60 +lat_0=-45 +lon_0=140 +x_0=500000 +y_0=1000000 +ellps=WGS84", (-180, 180), (-90, 90)),
    ("+proj=lcc +lat_1=46.8 +lat_0=46.8 +lon_0=2.337229167 +k_0=0.99987742 +x_0=600000 +y_0=2200000 +ellps=clrk80ign", (-180, 180), (-90, 90)),
    ("+proj=lcc +lat_1=25 +lon_0=180 +a=6378137 +rf=298.257223563 +k_0=0.9996", (-180, 180), (-89, 89)),
    ("+proj=lcc +lat_1=-20 +lat_2=10 +lat_0=-5 +lon_0=-170 +a=6378137 +b=6356752.314245", (-180, 180), (-89, 89)),
    ("+proj=lcc +lat_1=60 +lat_2=89.9 +lat_0=90 +lon_0=0 +ellps=WGS84", (-180, 180), (-89, 90)),
    ("+proj=lcc +lat_1=-89 +lat_0=-90 +R=1", (-180, 180), (-90, 89)),
    ("+proj=lcc +ellps=GRS80 +lat_1=0.5 +lat_2=2", (-180, 180), (-80, 80)),
    ("+proj=lcc +lat_1=33 +lat_2=45 +lon_0=-96 +ellps=GRS80 +to_meter=0.3048", (-150, -40), (0, 80)),
    ("+proj=lcc +lat_1=33 +lat_2=45 +lon_0=-96 +ellps=bessel +units=km", (-150, -40), (0, 80)),
    ("+proj=lcc +lat_1=33 +lat_2=45 +a=6378137 +es=0.0066943799901413165", (-90, 90), (0, 80)),
    ("+proj=lcc +lat_1=33 +lat_2=45 +a=6378137 +e=0.08181919084262149", (-90, 90), (0, 80)),
    ("+proj=lcc +lat_1=33 +lat_2=45 +a=6378137 +f=0.0033528106647474805", (-90, 90), (0, 80)),
    ("+proj=lcc +lat_1=33 +lat_2=45 +ellps=intl", (-90, 90), (0, 80)),
    ("+proj=lcc +lat_1=33 +lat_2=45 +ellps=sphere", (-90, 90), (0, 80)),
    ("+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +ellps=WGS84", (-180, 180), (-90, 90)),
    ("+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +ellps=WGS84", (-180, 180), (-90, 90)),
    ("+proj=stere +lat_0=90 +lon_0=0 +k=0.994 +x_0=2000000 +y_0=2000000 +ellps=WGS84", (-180, 180), (-90, 90)),
    ("+proj=stere +lat_0=45 +lon_0=10 +k=0.9999 +x_0=100000 +y_0=200000 +ellps=WGS84", (-180, 180), (-90, 90)),
    ("+proj=stere +lat_0=-30 +lon_0=150 +k=0.9 +ellps=intl", (-180, 180), (-90, 90)),
    ("+proj=stere +lat_0=0 +lon_0=0 +ellps=GRS80", (-180, 180), (-90, 90)),
    ("+proj=stere +lat_0=60 +lon_0=-100 +R=6371229", (-180, 180), (-90, 90)),
    ("+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-105 +R=6371229", (-180, 180), (-90, 90)),
    ("+proj=stere +lat_0=-90 +lat_ts=-60 +lon_0=0 +R=6371229", (-180, 180), (-90, 90)),
    ("+proj=stere +lat_0=-90 +k_0=0.97 +R=6371229", (-180, 180), (-90, 90)),
    ("+proj=stere +lat_0=0 +lon_0=30 +R=6371000", (-180, 180), (-90, 90)),
    ("+proj=stere +lat_0=90 +lat_ts=-70 +ellps=WGS84", (-180, 180), (0, 90)),
    ("+proj=stere +lat_0=90 +lat_ts=70 +k_0=0.5 +ellps=WGS84", (-180, 180), (0, 90)),
    ("+proj=ups +ellps=WGS84", (-180, 180), (-90, 90)),
    ("+proj=ups +south +ellps=WGS84", (-180, 180), (-90, 90)),
    ("+proj=ups +ellps=WGS84 +k_0=0.5 +lon_0=30 +x_0=5 +y_0=7", (-180, 180), (0, 90)),
]


def sec_random(n=40000):
    print("=" * 78)
    print("2. random points over the whole globe (incl. out-of-domain); %d pts/def" % n)
    rng = np.random.default_rng(20261009)
    hdr = "  %-60s %9s %9s %9s %9s %6s %6s" % ("definition", "fwd64", "inv64", "fwd32", "inv32", "fwdNaN", "invNaN")
    print(hdr)
    rows = []
    for defn, (lo0, lo1), (la0, la1) in RAND_DEFS:
        P = L.make(defn)
        lam = np.radians(rng.uniform(lo0, lo1, n))
        phi = np.radians(rng.uniform(la0, la1, n))
        X, Y = proj_fwd(defn, lam, phi)
        x, y = port_fwd(P, lam, phi)
        rb, ob = bad(X, Y), bad(x, y)
        fmis = int((rb != ob).sum())
        ok = ~rb & ~ob
        ef = np.hypot(x - X, y - Y)[ok] * P.to_meter
        # metres error measured relative to the magnitude (far-field values are huge)
        mag = np.maximum(1.0, np.hypot(X, Y)[ok] * P.to_meter)
        efr = ef / mag * 1e7  # error scaled to a 1e4 km magnitude
        # inverse: of PROJ's forward output (finite ones) AND of random xy in a big box
        xs = np.concatenate([X[~rb], rng.uniform(-3e7, 3e7, n // 4) * P.fr_meter + P.x0 * P.fr_meter])
        ys = np.concatenate([Y[~rb], rng.uniform(-3e7, 3e7, n // 4) * P.fr_meter + P.y0 * P.fr_meter])
        RL, RP = proj_inv(defn, xs, ys)
        il, ip = port_inv(P, xs, ys)
        rbi, obi = bad(RL, RP), bad(il, ip)
        imis = int((rbi != obi).sum())
        oki = ~rbi & ~obi
        ei = ang_err_m(il[oki], ip[oki], RL[oki], RP[oki], a=P.a)
        # fp32
        x32, y32 = port_fwd(P, lam, phi, torch.float32)
        ok32 = ok & ~bad(x32, y32)
        dom = ok32 & (np.hypot(X, Y) * P.to_meter < 2e7)   # bounded region only for fp32 stats
        ef32 = np.hypot(x32 - X, y32 - Y)[dom] * P.to_meter
        il32, ip32 = port_inv(P, xs[:int((~rb).sum())], ys[:int((~rb).sum())], torch.float32)
        okk = oki[:int((~rb).sum())] & ~bad(il32, ip32)
        ei32 = ang_err_m(il32[okk], ip32[okk], RL[:int((~rb).sum())][okk], RP[:int((~rb).sum())][okk], a=P.a)
        r = dict(defn=defn, fwd64=float(ef.max()) if ef.size else 0.0, fwd64_rel=float(efr.max()) if efr.size else 0.0,
                 inv64=float(ei.max()) if ei.size else 0.0, fwd32=float(ef32.max()) if ef32.size else 0.0,
                 inv32=float(ei32.max()) if ei32.size else 0.0, fmis=fmis, imis=imis,
                 nref_fail=int(rb.sum()), nref_inv_fail=int(rbi.sum()))
        rows.append(r)
        print("  %-60s %9.2e %9.2e %9.2e %9.2e %6d %6d  (PROJ fails fwd %d inv %d; fwd rel-scaled %.1e)" % (
            defn[:60], r["fwd64"], r["inv64"], r["fwd32"], r["inv32"], fmis, imis, r["nref_fail"],
            r["nref_inv_fail"], r["fwd64_rel"]))
        if fmis or imis:
            # show a few of the mismatching points
            k = np.nonzero(rb != ob)[0][:3]
            for i in k:
                print("     fwd mismatch lon %.6f lat %.6f: PROJ (%g, %g) port (%g, %g)" % (
                    np.degrees(lam[i]), np.degrees(phi[i]), X[i], Y[i], x[i], y[i]))
            k = np.nonzero(rbi != obi)[0][:3]
            for i in k:
                print("     inv mismatch x %.3f y %.3f: PROJ (%g, %g) port (%g, %g)" % (
                    xs[i], ys[i], RL[i], RP[i], il[i], ip[i]))
            finding("failure-pattern mismatch for %s: fwd %d inv %d" % (defn, fmis, imis))
        if r["fwd64_rel"] > 1e-6 or r["inv64"] > 1e-6:
            # locate
            j = int(np.argmax(ei)) if ei.size else None
            if r["inv64"] > 1e-6 and j is not None:
                print("     worst inv at x %.3f y %.3f: PROJ (%.12f, %.12f) port (%.12f, %.12f)" % (
                    xs[oki][j], ys[oki][j], RL[oki][j], RP[oki][j], il[oki][j], ip[oki][j]))
            finding("fp64 discrepancy for %s: fwd(rel-scaled) %.2e inv %.2e" % (defn, r["fwd64_rel"], r["inv64"]))
    return rows


# =============================================================================
# 3. edge points in the forward direction
# =============================================================================
EDGE_DEFS = [
    "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +ellps=WGS84",
    "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +ellps=WGS84 +over",
    "+proj=lcc +lat_1=-30 +lat_2=-60 +lat_0=-45 +lon_0=140 +ellps=WGS84",
    "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=45 +lon_0=10 +R=6370000",
    "+proj=lcc +lat_1=25 +lon_0=180 +a=6378137 +rf=298.257223563 +k_0=0.9996 +x_0=1000 +y_0=-500",
    "+proj=lcc +lat_1=60 +lat_2=89.9 +lat_0=90 +lon_0=0 +ellps=WGS84",
    "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +ellps=WGS84",
    "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +ellps=WGS84 +over",
    "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +ellps=WGS84",
    "+proj=stere +lat_0=45 +lon_0=170 +k=0.9999 +x_0=100000 +y_0=200000 +ellps=WGS84",
    "+proj=stere +lat_0=0 +lon_0=-180 +ellps=GRS80",
    "+proj=stere +lat_0=60 +lon_0=-100 +R=6371229",
    "+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-105 +R=6371229",
    "+proj=stere +lat_0=-90 +lat_ts=-60 +lon_0=0 +R=6371229",
    "+proj=stere +lat_0=0 +lon_0=30 +R=6371000",
    "+proj=ups +ellps=WGS84",
    "+proj=ups +south +ellps=WGS84",
]


def edge_points(P):
    h = math.pi / 2
    lam0 = P.lam0
    lats = [h, -h, h - 1e-12, -h + 1e-12, h + 0.5e-12, -h - 0.5e-12, h + 2e-12, -h - 2e-12,
            h - 1e-15, h - 2e-15, h - 1e-10, h - 0.9e-10, h - 1.1e-10, h - 1e-8, h - 0.99e-8,
            0.0, -0.0, P.phi0, -P.phi0, 1e-300, math.radians(89.9999999), math.radians(-89.9999999),
            math.radians(45.0), math.radians(-45.0)]
    lons = [lam0, lam0 + math.pi, lam0 - math.pi, lam0 + math.pi + 1e-12, lam0 + math.pi - 1e-12,
            lam0 + math.pi + 2e-12, math.pi, -math.pi, math.radians(181.0), math.radians(-181.0),
            math.radians(360.0), math.radians(540.0), math.radians(444.0), math.radians(-276.0),
            9.99, 10.0, 10.01, -10.01, 12.0, 0.0, -0.0, 1e-300]
    LA, PH = np.meshgrid(np.array(lons), np.array(lats))
    return LA.ravel(), PH.ravel()


def sec_edge():
    print("=" * 78)
    print("3. forward edge points (poles, +-1e-12, antimeridian, |lon|>pi, |lon|>10, +over)")
    for defn in EDGE_DEFS:
        P = L.make(defn)
        lam, phi = edge_points(P)
        X, Y = proj_fwd(defn, lam, phi)
        x, y = port_fwd(P, lam, phi)
        rb, ob = bad(X, Y), bad(x, y)
        mis = np.nonzero(rb != ob)[0]
        ok = ~rb & ~ob
        d = np.hypot(x - X, y - Y)
        tol = np.maximum(1e-6, 1e-13 * np.hypot(X, Y))
        val = np.nonzero(ok & (d > tol))[0]
        print("  %-66s %4d pts: fail-pattern mismatches %d, value mismatches %d" % (defn[:66], len(lam), len(mis), len(val)))
        groups = {}
        for i in mis:
            key = "PROJ %s / port %s" % ("fail" if rb[i] else "ok", "fail" if ob[i] else "ok")
            groups.setdefault(key, []).append(i)
        for key, idx in groups.items():
            lonset = sorted(set(round(float(lam[i]), 13) for i in idx))
            latset = sorted(set(round(float(phi[i]), 16) for i in idx))
            i = idx[0]
            print("     %s: %d pts, lons(rad) %s, lats(rad) %s; e.g. lam=%r phi=%r PROJ=(%g,%g) port=(%g,%g)" % (
                key, len(idx), lonset[:6], latset[:6], lam[i], phi[i], X[i], Y[i], x[i], y[i]))
        for i in val[:4]:
            print("     value diff %.3e m at lam=%r phi=%r: PROJ (%.6f, %.6f) port (%.6f, %.6f)" % (
                d[i], lam[i], phi[i], X[i], Y[i], x[i], y[i]))
        if len(mis):
            finding("edge fwd fail-pattern mismatch: %s (%d pts; %s)" % (defn, len(mis), "; ".join(groups.keys())))
        if len(val):
            finding("edge fwd value mismatch: %s (%d pts, max %.3e m)" % (defn, len(val), d[val].max()))


# =============================================================================
# 4. inverse edge points
# =============================================================================
def sec_inv_edge():
    print("=" * 78)
    print("4. inverse edge points (centre, apex, axes, huge, inf/nan)")
    for defn in EDGE_DEFS + ["+proj=lcc +ellps=GRS80 +lat_1=0.5 +lat_2=2"]:
        P = L.make(defn)
        A = P.a
        cx, cy = P.x0 * P.fr_meter, P.y0 * P.fr_meter
        pts = [(cx, cy), (cx + 1e-300, cy), (cx, cy + 1e-9), (cx + 1e-4, cy + 2e-4), (cx + 1.0, cy)]
        if isinstance(P, L.LambertConformalConic):
            ay = cy + P.rho0 * P.k0 * A * P.fr_meter
            pts += [(cx, ay), (cx + 1e-9, ay), (cx, ay + 1e-9), (cx + 1.0, ay), (cx, ay + 1.0), (cx + 30.0, ay),
                    (cx, ay + 50 * A), (cx, ay - 50 * A), (cx - 1e5, ay + 1e3)]
        else:
            r = A * P.akm1
            pts += [(cx + r, cy), (cx, cy + r), (cx, cy - r), (cx + 3 * r, cy), (cx + 1e3 * r, cy), (cx, cy + 1e3 * r),
                    (cx + 0.6 * A * 1e-10, cy + 0.8 * A * 1e-10), (cx + 0.7 * r, cy + 0.7 * r)]
        pts += [(1e20, 1e20), (-1e20, 3.0), (cx, 1e300), (math.inf, 0.0), (0.0, math.nan)]
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        RL, RP = proj_inv(defn, xs, ys)
        il, ip = port_inv(P, xs, ys)
        rb, ob = bad(RL, RP), bad(il, ip)
        mis = np.nonzero(rb != ob)[0]
        ok = ~rb & ~ob
        e = np.zeros_like(xs)
        e[ok] = ang_err_m(il[ok], ip[ok], RL[ok], RP[ok], a=A)
        dl = np.zeros_like(xs)
        dl[ok] = np.abs(il[ok] - RL[ok])
        val = np.nonzero(ok & ((e > 1e-6) | (dl > 1e-9) & (np.abs(np.abs(RP) - math.pi / 2) > 1e-9)))[0]
        print("  %-66s %3d pts: fail mismatches %d, value mismatches %d" % (defn[:66], len(xs), len(mis), len(val)))
        for i in list(mis) + list(val):
            print("     x=%r y=%r: PROJ (%r, %r) port (%r, %r)" % (xs[i], ys[i], RL[i], RP[i], il[i], ip[i]))
        if len(mis) or len(val):
            finding("inverse edge mismatch for %s: %d fail-pattern, %d value" % (defn, len(mis), len(val)))


# =============================================================================
# 5. setup parameters at PROJ's EPS10 / TOL thresholds
# =============================================================================
EPS_DEFS = [
    # lcc |lat_1 + lat_2| < EPS10 (5.7296e-9 deg): just below / just above
    "+proj=lcc +lat_1=30 +lat_2=-29.999999996 +ellps=GRS80",
    "+proj=lcc +lat_1=30 +lat_2=-29.99999999 +ellps=GRS80",
    "+proj=lcc +lat_1=30 +lat_2=-29.99999999 +R=6400000",
    # secant switch |lat_1 - lat_2| >= EPS10
    "+proj=lcc +lat_1=45 +lat_2=45.000000005 +ellps=GRS80",
    "+proj=lcc +lat_1=45 +lat_2=45.000000006 +ellps=GRS80",
    "+proj=lcc +lat_1=45 +lat_2=45.00000001 +R=6400000",
    # cos(phi) < EPS10 limits
    "+proj=lcc +lat_1=89.999999994 +lat_2=60 +ellps=GRS80",
    "+proj=lcc +lat_1=89.99999999 +lat_2=60 +ellps=GRS80",
    "+proj=lcc +lat_1=60 +lat_2=-89.99999999 +ellps=GRS80",
    # rho0 pole test for lat_0
    "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=89.999999995 +ellps=GRS80",
    "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=89.99999999 +ellps=GRS80",
    "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=-89.999999995 +R=1",
    # tiny n (sphere)  "+proj=lcc +a=1 +lat_2=.0000001"
    "+proj=lcc +a=1 +lat_2=.0000001",
    "+proj=lcc +R=1 +lat_1=1e-7 +lat_2=2e-7",
    # stere mode thresholds
    "+proj=stere +lat_0=89.999999995 +ellps=WGS84",
    "+proj=stere +lat_0=89.99999999 +ellps=WGS84",
    "+proj=stere +lat_0=-89.999999995 +R=6371000",
    "+proj=stere +lat_0=-89.99999999 +R=6371000",
    "+proj=stere +lat_0=5e-9 +ellps=WGS84",
    "+proj=stere +lat_0=6e-9 +ellps=WGS84",
    "+proj=stere +lat_0=-6e-9 +R=6371000",
    "+proj=stere +lat_0=90 +lat_ts=89.999999995 +ellps=WGS84",
    "+proj=stere +lat_0=90 +lat_ts=89.99999999 +ellps=WGS84",
    "+proj=stere +lat_0=90 +lat_ts=89.99999999 +R=6371000",
    "+proj=stere +lat_0=90 +lat_ts=0 +ellps=WGS84",
    "+proj=stere +lat_0=-90 +lat_ts=1e-9 +R=6371000",
    # odd ellipsoids
    "+proj=stere +lat_0=50 +a=6378137 +b=6378137",
    "+proj=stere +lat_0=50 +a=6378137 +b=6000000",
    "+proj=lcc +lat_1=20 +lat_2=60 +a=6378137 +b=3000000",
    "+proj=stere +lat_0=90 +a=6378137 +e=0.5",
    "+proj=lcc +lat_1=20 +lat_2=60 +a=6378137 +rf=298.257223563 +es=0.1",
    "+proj=lcc +lat_1=20 +lat_2=60 +a=6378137 +f=0.003 +rf=200",
    "+proj=lcc +lat_1=20 +lat_2=60 +ellps=WGS84 +a=6400000",
    "+proj=lcc +lat_1=20 +lat_2=60 +ellps=WGS84 +R=6400000",
    "+proj=stere +lat_0=-90 +lat_ts=-70 +ellps=airy +b=6356000",
    "+proj=lcc +lat_1=44d11'N +lat_2=45d42'N +lat_0=43d19'N +lon_0=84d20'W +ellps=clrk66",
    "+proj=lcc +lat_1=44.5 +lat_2=46 +lon_0=0.5r +ellps=GRS80",
    "+proj=lcc +lat_1=44.5 +lat_2=46 +lon_0=-1.5e1 +ellps=GRS80",
]


def sec_eps(n=4000):
    print("=" * 78)
    print("5. parameters at PROJ's thresholds / unusual ellipsoids: setup outcome and values vs PROJ")
    rng = np.random.default_rng(5)
    for defn in EPS_DEFS:
        try:
            _tr(defn)
            proj_ok = True
        except Exception:
            proj_ok = False
        try:
            P = L.make(defn)
            port_ok = True
        except Exception as ex:
            port_ok, perr = False, ex
        line = "  %-62s PROJ %-4s port %-4s" % (defn[:62], "ok" if proj_ok else "FAIL", "ok" if port_ok else "FAIL")
        if proj_ok != port_ok:
            print(line + "  MISMATCH" + ("" if port_ok else " (%s)" % perr))
            finding("setup outcome differs for %s (PROJ ok=%s, port ok=%s)" % (defn, proj_ok, port_ok))
            continue
        if not proj_ok:
            print(line)
            continue
        lam = np.radians(rng.uniform(-180, 180, n))
        phi = np.radians(rng.uniform(-90, 90, n))
        X, Y = proj_fwd(defn, lam, phi)
        x, y = port_fwd(P, lam, phi)
        rb, ob = bad(X, Y), bad(x, y)
        ok = ~rb & ~ob
        d = (np.hypot(x - X, y - Y) / np.maximum(1.0, np.hypot(X, Y)))[ok]
        RL, RP = proj_inv(defn, X[~rb], Y[~rb])
        il, ip = port_inv(P, X[~rb], Y[~rb])
        rbi, obi = bad(RL, RP), bad(il, ip)
        oki = ~rbi & ~obi
        ei = ang_err_m(il[oki], ip[oki], RL[oki], RP[oki], a=P.a)
        fm, im = int((rb != ob).sum()), int((rbi != obi).sum())
        md, mi = (float(d.max()) if d.size else 0.0), (float(ei.max()) if ei.size else 0.0)
        flag = fm or im or md > 1e-12 or mi > 1e-6
        print(line + "  fwd rel %.1e inv %.1e m, fail mism fwd %d inv %d  (PROJ fails %d/%d)%s" % (
            md, mi, fm, im, int(rb.sum()), int(rbi.sum()), "  <--" if flag else ""))
        if flag:
            finding("threshold/ellipsoid def %s: fwd rel %.2e inv %.2e m, mismatches fwd %d inv %d" % (defn, md, mi, fm, im))


# =============================================================================
# 6. setup checks inherited from init.cpp / pj_param
# =============================================================================
SETUP_DEFS = [
    "+proj=lcc +lat_1=33 +lat_2=45 +ellps=GRS80 +k_0=0",
    "+proj=lcc +lat_1=33 +lat_2=45 +ellps=GRS80 +k_0=-1",
    "+proj=stere +lat_0=90 +ellps=GRS80 +k=-0.5",
    "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=95 +ellps=GRS80",
    "+proj=stere +lat_0=91 +ellps=GRS80",
    "+proj=stere +lat_0=-90.0000001 +R=6400000",
    "+proj=ups +south=f +ellps=WGS84",
    "+proj=ups +south=F +ellps=WGS84",
    "+proj=lcc +lat_1=33 +lat_2=45 +ellps=GRS80 +over=f",
    "+proj=lcc +lat_1=33 +lat_2=45 +ellps=GRS80 +pm=paris",
    "+proj=lcc +lat_1=33 +lat_2=45 +ellps=GRS80 +units=us-ch",
    "+proj=lcc +lat_1=33 +lat_2=45 +ellps=GRS80 +to_meter=1/3",
    "+proj=lcc +lat_1=33 +lat_2=45 +ellps=GRS80 +rf=0",
    "+proj=lcc +lat_1=33 +lat_2=45 +ellps=GRS80 +es=1.5",
    "+proj=lcc +lat_1=33 +lat_2=45 +ellps=WGS84 +axis=neu",
    "+proj=stere +lat_0=90 +lat_ts=95 +ellps=WGS84",
]


def sec_setup():
    print("=" * 78)
    print("6. generic PROJ setup semantics (init.cpp, pj_param) vs port")
    rng = np.random.default_rng(6)
    lam = np.radians(rng.uniform(-250, 250, 64))  # |lon|>180 exposes +over parsing
    phi = np.radians(rng.uniform(20, 80, 64))
    for defn in SETUP_DEFS:
        try:
            _tr(defn)
            proj_ok = True
        except Exception:
            proj_ok = False
        try:
            P = L.make(defn)
            port_ok, perr = True, ""
        except Exception as ex:
            port_ok, perr = False, "%s: %s" % (type(ex).__name__, ex)
        note = ""
        if proj_ok and port_ok:
            X, Y = proj_fwd(defn, lam, phi)
            x, y = port_fwd(P, lam, phi)
            okm = ~bad(X, Y) & ~bad(x, y)
            d = float(np.hypot(x - X, y - Y)[okm].max()) if okm.any() else 0.0
            note = "max |port-PROJ| %.3e m, fail mism %d" % (d, int((bad(X, Y) != bad(x, y)).sum()))
            if d > 1e-6 or (bad(X, Y) != bad(x, y)).any():
                finding("%s: port silently differs from PROJ (%s)" % (defn, note))
        elif proj_ok != port_ok:
            note = "MISMATCH " + perr[:80]
            finding("%s: PROJ creates=%s, port creates=%s %s" % (defn, proj_ok, port_ok, perr[:80]))
        print("  %-60s PROJ %-4s port %-4s %s" % (defn[:60], "ok" if proj_ok else "FAIL", "ok" if port_ok else "FAIL", note))


# =============================================================================
# 7. shapes, z/t passthrough, dtypes, requires_grad
# =============================================================================
def sec_shapes():
    print("=" * 78)
    print("7. shapes [B,H,W,2], [N,3], [N,4]; z/t vs PROJ; float32; requires_grad")
    rng = np.random.default_rng(7)
    for defn in ["+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +ellps=WGS84",
                 "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +ellps=WGS84",
                 "+proj=lcc +ellps=clrk66 +lat_1=44d11'N +lat_2=45d42'N +x_0=609601.2192 +lon_0=84d20'W +lat_0=43d19'N +k_0=1.0000382 +units=us-ft",
                 "+proj=stere +lat_0=45 +lon_0=10 +ellps=WGS84 +z_0=100",
                 "+proj=ups +ellps=WGS84"]:
        P = L.make(defn)
        lon = np.radians(rng.uniform(-100, -80, (2, 3, 4)))
        lat = np.radians(rng.uniform(60, 85, (2, 3, 4)))
        c4 = torch.stack([torch.as_tensor(lon), torch.as_tensor(lat)], -1)
        with torch.no_grad():
            o4 = P.forward(c4)
            o2 = P.forward(c4.reshape(-1, 2)).reshape(2, 3, 4, 2)
            b4 = P.inverse(o4)
        same = bool(torch.equal(o4, o2))
        rt = float((b4 - c4).abs().max())
        # z, t
        z = rng.uniform(-100, 3000, 24)
        tt = rng.uniform(2000, 2030, 24)
        c3 = torch.stack([torch.as_tensor(lon.ravel()), torch.as_tensor(lat.ravel()), torch.as_tensor(z)], -1)
        cc4 = torch.cat([c3, torch.as_tensor(tt)[:, None]], -1)
        with torch.no_grad():
            o3 = P.forward(c3).numpy()
            oo4 = P.forward(cc4).numpy()
            i3 = P.inverse(torch.as_tensor(o3)).numpy()
        X, Y, Z, T = proj_fwd(defn, lon.ravel(), lat.ravel(), z, tt)
        RL, RP, RZ, RT = proj_inv(defn, X, Y, Z, T)
        dz = float(np.abs(o3[:, 2] - Z).max())
        dzi = float(np.abs(i3[:, 2] - RZ).max())
        dt = float(np.abs(oo4[:, 3] - T).max())
        dxy = float(np.hypot(o3[:, 0] - X, o3[:, 1] - Y).max())
        # dtype / grad
        c32 = c4.float().requires_grad_(True)
        o32 = P.forward(c32)
        o32.sum().backward()
        g32 = c32.grad
        cg = c4.clone().requires_grad_(True)
        og = P.forward(cg)
        ig = P.inverse(og)
        ig.sum().backward()
        print("  %-60s [B,H,W,2] == flat: %s, roundtrip %.1e rad; out dtype32=%s grad32 dtype=%s finite=%s; "
              "xy diff %.1e; z diff fwd %.3g inv %.3g; t diff %.3g" % (
                  defn[:60], same, rt, o32.dtype, g32.dtype, bool(torch.isfinite(g32).all()), dxy, dz, dzi, dt))
        if dz > 1e-9 or dzi > 1e-9:
            finding("z not handled like PROJ for %s: PROJ scales/offsets z (vfr_meter/z_0), port passes it "
                    "through; max diff fwd %.4g inv %.4g (units of output)" % (defn, dz, dzi))
        if not same:
            finding("batched shape result differs from flat for %s" % defn)
    # fp32 at poles / special points: dtype stays fp32 and NaN pattern equals fp64's
    for defn in EDGE_DEFS:
        P = L.make(defn)
        lam, phi = edge_points(P)
        x64, y64 = port_fwd(P, lam, phi)
        x32, y32 = port_fwd(P, lam, phi, torch.float32)
        m = bad(x64, y64) != bad(x32, y32)
        if m.any():
            i = np.nonzero(m)[0]
            print("  fp32 vs fp64 failure pattern differs for %-50s at %d pts, e.g. lam=%r phi=%r: fp64 (%g,%g) fp32 (%g,%g)" % (
                defn[:50], len(i), lam[i[0]], phi[i[0]], x64[i[0]], y64[i[0]], x32[i[0]], y32[i[0]]))


# =============================================================================
# 8. autograd vs central finite differences (float64, random interior points)
# =============================================================================
FD_DEFS = [
    ("+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +ellps=WGS84", (-140, -50), (5, 80)),
    ("+proj=lcc +lat_1=30 +lat_2=60 +lat_0=45 +lon_0=10 +R=6370000", (-60, 80), (5, 85)),
    ("+proj=lcc +lat_1=-30 +lat_2=-60 +lat_0=-45 +lon_0=140 +ellps=WGS84", (80, 200), (-85, -5)),
    ("+proj=lcc +lat_1=46.8 +lat_0=46.8 +lon_0=2.337229167 +k_0=0.99987742 +x_0=600000 +y_0=2200000 +ellps=clrk80ign", (-30, 30), (20, 80)),
    ("+proj=lcc +lat_1=-89 +lat_0=-90 +R=1", (-170, 170), (-89.9, 60)),
    ("+proj=lcc +ellps=GRS80 +lat_1=0.5 +lat_2=2", (-60, 60), (-70, 70)),
    ("+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +ellps=WGS84", (-180, 180), (0, 89.999)),
    ("+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +ellps=WGS84", (-180, 180), (-89.999, 0)),
    ("+proj=stere +lat_0=45 +lon_0=10 +k=0.9999 +x_0=100000 +y_0=200000 +ellps=WGS84", (-60, 80), (-20, 89)),
    ("+proj=stere +lat_0=-30 +lon_0=150 +k=0.9 +ellps=intl", (90, 210), (-85, 20)),
    ("+proj=stere +lat_0=0 +lon_0=0 +ellps=GRS80", (-80, 80), (-80, 80)),
    ("+proj=stere +lat_0=60 +lon_0=-100 +R=6371229", (-160, -40), (0, 89)),
    ("+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-105 +R=6371229", (-180, 180), (0, 89.99)),
    ("+proj=stere +lat_0=-90 +lat_ts=-60 +lon_0=0 +R=6371229", (-180, 180), (-89.99, 0)),
    ("+proj=stere +lat_0=0 +lon_0=30 +R=6371000", (-50, 110), (-80, 80)),
    ("+proj=ups +ellps=WGS84", (-180, 180), (60, 89.999)),
    ("+proj=ups +south +ellps=WGS84", (-180, 180), (-89.999, -60)),
]


def jac_autograd(fn, X):
    X = X.clone().requires_grad_(True)
    Y = fn(X)
    rows = []
    for k in range(2):
        g, = torch.autograd.grad(Y[..., k].sum(), X, retain_graph=True)
        rows.append(g[..., :2])
    return Y.detach(), torch.stack(rows, -2)  # [N, out, in]


def jac_fd(fn, X, h):
    cols = []
    for j in range(2):
        e = torch.zeros_like(X)
        e[..., j] = h
        with torch.no_grad():
            cols.append((fn(X + e)[..., :2] - fn(X - e)[..., :2]) / (2 * h))
    return torch.stack(cols, -1)  # [N, out, in]


def sec_grad_fd(n=400):
    print("=" * 78)
    print("8. autograd Jacobian vs central FD (fp64). rel err = |J_ad - J_fd| / |J_fd| (Frobenius, per point)")
    rng = np.random.default_rng(8)
    worst_all = 0.0
    for defn, (lo0, lo1), (la0, la1) in FD_DEFS:
        for mode_kw in ({}, {"iter_mode": "fixed", "n_iter": 6} if "stere" in defn or "ups" in defn else {"iter_mode": "fixed", "n_iter": 2}):
            P = L.make(defn, **mode_kw)
            lam = torch.as_tensor(np.radians(rng.uniform(lo0, lo1, n)))
            phi = torch.as_tensor(np.radians(rng.uniform(la0, la1, n)))
            X = torch.stack([lam, phi], -1)
            # forward: output in units of a (O(1)); h = 1e-6 rad
            f = lambda c: P.forward(c) / P.a  # noqa: E731
            Yf, Jad = jac_autograd(f, X)
            Jfd = jac_fd(f, X, 1e-6)
            ok = torch.isfinite(Yf).all(-1) & torch.isfinite(Jfd).flatten(1).all(-1)
            rel = ((Jad - Jfd).flatten(1).norm(dim=-1) / Jfd.flatten(1).norm(dim=-1))[ok]
            nanad = int((~torch.isfinite(Jad).flatten(1).all(-1) & ok).sum())
            # inverse at the forward images: h = 1 m (in output units)
            with torch.no_grad():
                XY = P.forward(X)
            XY = XY[ok]
            g = lambda c: P.inverse(c)  # noqa: E731
            Yi, Jadi = jac_autograd(g, XY)
            Jfdi = jac_fd(g, XY, 1.5e-7 * P.a * P.fr_meter)  # ~1 m on the Earth
            oki = torch.isfinite(Yi).all(-1) & torch.isfinite(Jfdi).flatten(1).all(-1)
            # exclude points where FD straddles the lon cut (|dlam| ~ 2pi)
            oki &= (Jfdi.abs().flatten(1).max(-1).values * P.a < 1e6)
            reli = ((Jadi - Jfdi).flatten(1).norm(dim=-1) / Jfdi.flatten(1).norm(dim=-1))[oki]
            nanadi = int((~torch.isfinite(Jadi).flatten(1).all(-1) & oki).sum())
            w = max(float(rel.max()), float(reli.max()))
            worst_all = max(worst_all, w)
            k = int(torch.argmax(reli)) if reli.numel() else 0
            tag = "" if not mode_kw else " [%s %d]" % (mode_kw["iter_mode"], mode_kw["n_iter"])
            print("  %-60s%s fwd max %.1e med %.1e | inv max %.1e med %.1e | NaN-grad fwd %d inv %d" % (
                defn[:60], tag, float(rel.max()), float(rel.median()), float(reli.max()), float(reli.median()), nanad, nanadi))
            if w > 1e-5 or nanad or nanadi:
                Yi_ok = Yi[oki]
                print("     worst inverse point: lon %.6f lat %.6f" % (math.degrees(float(Yi_ok[k, 0])), math.degrees(float(Yi_ok[k, 1]))))
                finding("autograd vs FD for %s%s: max rel fwd %.2e inv %.2e, NaN grads %d/%d" % (
                    defn, tag, float(rel.max()), float(reli.max()), nanad, nanadi))
    print("  worst relative Jacobian error overall: %.2e" % worst_all)


# =============================================================================
# 9. NaN / inf gradients at special and failing points
# =============================================================================
def sec_grad_nan():
    print("=" * 78)
    print("9. gradient hygiene at special / failing points (loss = sum of finite outputs)")
    for defn in EDGE_DEFS + ["+proj=lcc +ellps=GRS80 +lat_1=0.5 +lat_2=2",
                             "+proj=stere +lat_0=0 +lon_0=0 +R=6400000"]:
        P = L.make(defn)
        lam, phi = edge_points(P)
        X = torch.stack([torch.as_tensor(lam), torch.as_tensor(phi)], -1).requires_grad_(True)
        out = P.forward(X)
        m = torch.isfinite(out)
        loss = torch.where(m, out, torch.zeros_like(out)).sum() / P.a
        g, = torch.autograd.grad(loss, X)
        nf = ~torch.isfinite(g).all(-1)
        # inverse at the forward images + special xy
        xy = torch.where(m, out, torch.zeros_like(out)).detach()
        A = P.a
        extra = [[P.x0 * P.fr_meter, P.y0 * P.fr_meter]]
        if isinstance(P, L.LambertConformalConic):
            ay = P.y0 * P.fr_meter + P.rho0 * P.k0 * A * P.fr_meter
            extra += [[P.x0 * P.fr_meter, ay], [P.x0 * P.fr_meter + 1e-6, ay], [P.x0 * P.fr_meter + 20.0, ay],
                      [P.x0 * P.fr_meter, ay + 1e-3]]
        xy = torch.cat([xy, torch.tensor(extra, dtype=torch.float64)]).requires_grad_(True)
        back = P.inverse(xy)
        mb = torch.isfinite(back)
        g2, = torch.autograd.grad(torch.where(mb, back, torch.zeros_like(back)).sum(), xy)
        nf2 = ~torch.isfinite(g2).all(-1)
        print("  %-66s non-finite grads: fwd %d / %d, inv %d / %d" % (defn[:66], int(nf.sum()), len(lam), int(nf2.sum()), xy.shape[0]))
        if nf.any() or nf2.any():
            for i in torch.nonzero(nf).ravel()[:3].tolist():
                print("     fwd lam=%r phi=%r out=%s grad=%s" % (lam[i], phi[i], out[i].tolist(), g[i].tolist()))
            for i in torch.nonzero(nf2).ravel()[:3].tolist():
                print("     inv xy=%s out=%s grad=%s" % (xy[i].tolist(), back[i].tolist(), g2[i].tolist()))
            finding("non-finite gradients for %s: fwd %d inv %d" % (defn, int(nf.sum()), int(nf2.sum())))
    # gradient at points where the forward value is finite but grad is 0 due to constant branches
    print("  zero-Jacobian (constant-branch) points where the output is finite and not a singularity:")
    for defn, pt in [("+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-105 +R=6371229", [3e-4, 4e-4]),
                     ("+proj=stere +lat_0=-90 +lat_ts=-60 +lon_0=0 +R=6371229", [3e-4, 4e-4])]:
        P = L.make(defn)
        Xp = torch.tensor([pt], dtype=torch.float64)
        Y, J = jac_autograd(P.inverse, Xp)
        Jfd = jac_fd(P.inverse, Xp, 1e-4)
        print("     %s inv at %s m: dphi/dxy autograd %s vs FD %s" % (defn[:50], pt, J[0, 1].tolist(), Jfd[0, 1].tolist()))


# =============================================================================
# 10. static scan for host syncs / data-dependent Python control flow
# =============================================================================
def sec_scan():
    print("=" * 78)
    print("10. static scan of lcc_stere.py")
    src = Path(__file__).with_name("lcc_stere.py").read_text().splitlines()
    pats = [r"\.item\(", r"\.numpy\(", r"\.tolist\(", r"\.cpu\(", r"\bbool\(", r"\bfloat\(.*(phi|lam|x|y|rho|tau)",
            r"\bint\(.*(phi|lam|x|y|rho|tau)", r"\.any\(", r"\.all\(", r"nonzero", r"masked_select", r"\bwhile\b",
            r"torch\.unique", r"\[.*(mask|bad|done|active|origin).*\]"]
    hits = []
    for i, line in enumerate(src, 1):
        code = line.split("#", 1)[0]
        for p in pats:
            if re.search(p, code):
                hits.append((i, p, line.strip()))
    for h in hits:
        print("  line %d [%s]: %s" % h)
    # Python `if` statements: list those that reference tensor-ish names
    for i, line in enumerate(src, 1):
        s = line.strip()
        if s.startswith(("if ", "elif ")) or " if " in s.split("#", 1)[0]:
            if re.search(r"\b(phi|lam|x|y|rho|rh|tau|taup|ts0|coo|dtau|cosc|sinc)\b(?!\w)", s.split("#", 1)[0]) and \
                    not re.search(r"self\.|mode|dtype|stats|count|iter|n_iter|es|smooth|fr_meter|to_meter|isinstance|key|params|value|s\[|c in|nl|tv|sign", s):
                print("  possible data-dependent if, line %d: %s" % (i, s))
    print("  %d pattern hits (inspect: only setup / Python-float code should appear)" % len(hits))


# =============================================================================
# 11. torch.compile on modes the implementer did not compile
# =============================================================================
def sec_compile():
    print("=" * 78)
    print("11. torch.compile(fullgraph=True) on modes not covered by the implementer's smoke test")
    import torch._dynamo as dynamo
    for defn in ["+proj=stere +lat_0=0 +lon_0=0 +ellps=GRS80",
                 "+proj=ups +south +ellps=WGS84",
                 "+proj=stere +lat_0=0 +lon_0=30 +R=6371000",
                 "+proj=lcc +ellps=clrk66 +lat_1=44d11'N +lat_2=45d42'N +x_0=609601.2192 +lon_0=84d20'W +lat_0=43d19'N +k_0=1.0000382 +units=us-ft"]:
        P = L.make(defn)
        x = torch.tensor([[-0.3, 0.7], [0.2, 0.9], [0.1, -0.5], [0.0, 1.5707963267948966]], dtype=torch.float64)
        for name, fn, arg in (("fwd", P.forward, x), ("inv", P.inverse, torch.nan_to_num(P.forward(x).detach()))):
            dynamo.reset()
            try:
                t0 = time.perf_counter()
                cf = torch.compile(fn, fullgraph=True, dynamic=True)
                r = cf(arg)
                r2 = cf(torch.cat([arg, arg]))  # different size: dynamic shape, no recompile crash
                d = float(torch.nan_to_num(r - fn(arg)).abs().max())
                ag = arg.clone().requires_grad_(True)
                g1, = torch.autograd.grad(torch.nan_to_num(cf(ag)).sum(), ag)
                ag2 = arg.clone().requires_grad_(True)
                g2, = torch.autograd.grad(torch.nan_to_num(fn(ag2)).sum(), ag2)
                msg = "ok %.1fs, max diff %.1e, grad diff %.1e" % (time.perf_counter() - t0, d,
                                                                  float(torch.nan_to_num(g1 - g2).abs().max()))
            except Exception as ex:
                msg = "FAILED: %s" % str(ex).splitlines()[0][:150]
                finding("torch.compile failed for %s %s: %s" % (defn, name, msg))
            print("  %-55s %s %s" % (defn[:55], name, msg))


def main():
    t0 = time.time()
    order = ["claims", "gie", "random", "edge", "inv_edge", "eps", "setup", "shapes", "fp32_pole", "grad_fd", "grad_nan", "scan", "compile"]
    fns = dict(gie=sec_gie, random=sec_random, edge=sec_edge, inv_edge=sec_inv_edge, eps=sec_eps, setup=sec_setup,
               shapes=sec_shapes, claims=sec_claims, fp32_pole=sec_fp32_pole, grad_fd=sec_grad_fd, grad_nan=sec_grad_nan, scan=sec_scan, compile=sec_compile)
    for s in order:
        if s in SECTIONS:
            fns[s]()
    print("=" * 78)
    print("FINDINGS (%d):" % len(FINDINGS))
    for f in FINDINGS:
        print("  -", f)
    print("elapsed %.1fs" % (time.time() - t0))



# =============================================================================
# 12. re-derive the implementer's headline table (same boxes, new seed, own code)
# =============================================================================
CLAIMED = {  # definition: (fwd64, inv64, fwd32, inv32) from the implementer report
    "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=45 +lon_0=10 +R=6370000": (2.33e-9, 4.24e-9, 2.81, 2.55),
    "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +ellps=WGS84": (3.55e-9, 5.36e-9, 2.43, 1.8),
    "+proj=lcc +lat_1=-30 +lat_2=-60 +lat_0=-45 +lon_0=140 +x_0=500000 +y_0=1000000 +ellps=WGS84": (3.98e-9, 5.45e-9, 2.45, 1.78),
    "+proj=lcc +lat_1=46.8 +lat_0=46.8 +lon_0=2.337229167 +k_0=0.99987742 +x_0=600000 +y_0=2200000 +ellps=clrk80ign": (2.8e-9, 2.83e-9, 2.0, 1.57),
    "+proj=lcc +ellps=clrk66 +lat_1=44d11'N +lat_2=45d42'N +x_0=609601.2192 +lon_0=84d20'W +lat_0=43d19'N +k_0=1.0000382 +units=us-ft": (2.91e-9, 4.25e-9, 2.13, 1.75),
    "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +ellps=WGS84": (4.67e-9, 2.83e-9, 3.52, 2.73),
    "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +ellps=WGS84": (2.33e-9, 2.83e-9, 1.57, 1.56),
    "+proj=stere +lat_0=90 +lon_0=0 +k=0.994 +x_0=2000000 +y_0=2000000 +ellps=WGS84": (2.33e-9, 2.83e-9, 1.5, 1.52),
    "+proj=ups +ellps=WGS84": (2.08e-9, 2.83e-9, 1.44, 1.5),
    "+proj=ups +south +ellps=WGS84": (1.92e-9, 2.83e-9, 1.23, 1.4),
    "+proj=stere +lat_0=45 +lon_0=10 +k=0.9999 +x_0=100000 +y_0=200000 +ellps=WGS84": (4.59e-9, 1.13e-8, 2.85, 5.64),
    "+proj=stere +lat_0=0 +lon_0=0 +ellps=GRS80": (5.61e-9, 6.55e-9, 3.47, 3.27),
    "+proj=stere +lat_0=60 +lon_0=-100 +R=6371229": (2.98e-9, 2.83e-8, 1.61, 38),
    "+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-105 +R=6371229": (2.08e-9, 3.25e-8, 8.16, 1630),
    "+proj=stere +lat_0=0 +lon_0=30 +R=6371000": (1.92e-9, 5.09e-9, 1.97, 1.95),
}
BOXES = {  # from lcc_stere_test.DEFS
    0: ((-40, 60), (15, 80)), 1: ((-140, -50), (10, 65)), 2: ((100, 180), (-80, -10)), 3: ((-6, 10), (41, 52)),
    4: ((-91, -82), (41, 48.5)), 5: ((-180, 180), (30, 90)), 6: ((-180, 180), (-90, -40)), 7: ((-180, 180), (50, 90)),
    8: ((-180, 180), (60, 90)), 9: ((-180, 180), (-90, -60)), 10: ((-30, 50), (10, 80)), 11: ((-60, 60), (-60, 60)),
    12: ((-150, -50), (20, 89)), 13: ((-180, 180), (0, 90)), 14: ((-30, 90), (-60, 60)),
}


def sec_claims(n=100000):
    print("=" * 78)
    print("12. headline table re-derived (implementer's boxes, seed 777, %d pts); claimed -> measured" % n)
    rng = np.random.default_rng(777)
    for i, (defn, cl) in enumerate(CLAIMED.items()):
        (lo0, lo1), (la0, la1) = BOXES[i]
        P = L.make(defn)
        lam = np.radians(rng.uniform(lo0, lo1, n))
        phi = np.radians(rng.uniform(la0, la1, n))
        X, Y = proj_fwd(defn, lam, phi)
        rb = bad(X, Y)
        RL, RP = proj_inv(defn, X[~rb], Y[~rb])
        res = []
        for dt in (torch.float64, torch.float32):
            x, y = port_fwd(P, lam, phi, dt)
            ok = ~rb & ~bad(x, y)
            ef = float((np.hypot(x - X, y - Y)[ok] * P.to_meter).max())
            il, ip = port_inv(P, X[~rb], Y[~rb], dt)
            oki = ~bad(il, ip)
            ei = float(ang_err_m(il[oki], ip[oki], RL[oki], RP[oki], a=P.a).max())
            res += [ef, ei]
            if dt == torch.float64 and (int((bad(x, y) != rb).sum()) or int((~oki).sum())):
                finding("NaN pattern differs in claimed box for %s" % defn)
        ratio = [r / c for r, c in zip(res, cl)]
        flag = any(rt > 3 for rt in ratio)
        print("  %-55s fwd64 %.2e->%.2e inv64 %.2e->%.2e fwd32 %.3g->%.3g inv32 %.3g->%.3g%s" % (
            defn[:55], cl[0], res[0], cl[1], res[1], cl[2], res[2], cl[3], res[3], "  <-- >3x claim" if flag else ""))
        if flag:
            finding("claimed max error understated for %s: claimed %s measured %s" % (defn, cl, ["%.3g" % r for r in res]))


# =============================================================================
# 13. fp32 near the poles (oblique/equatorial ellipsoidal stere inverse) and
#     NaN-gradient radius around the apex of small-n LCC cones
# =============================================================================
def sec_fp32_pole():
    print("=" * 78)
    print("13a. fp32 inverse near the poles (PROJ fp64 forward images, lat 85..90 / -90..-85)")
    for defn in ["+proj=stere +lat_0=45 +lon_0=10 +k=0.9999 +x_0=100000 +y_0=200000 +ellps=WGS84",
                 "+proj=stere +lat_0=0 +lon_0=0 +ellps=GRS80", "+proj=stere +lat_0=-30 +lon_0=150 +k=0.9 +ellps=intl",
                 "+proj=stere +lat_0=60 +lon_0=-100 +R=6371229", "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +ellps=WGS84"]:
        P = L.make(defn)
        for sgn in (1.0, -1.0):
            lat = sgn * np.linspace(85, 90, 50001)
            lon = np.full_like(lat, math.degrees(P.lam0) + 20.0)
            X, Y = proj_fwd(defn, np.radians(lon), np.radians(lat))
            ok = ~bad(X, Y)
            RL, RP = proj_inv(defn, X[ok], Y[ok])
            il, ip = port_inv(P, X[ok], Y[ok], torch.float32)
            out_of_range = np.abs(ip) > math.pi / 2 + 1e-6
            nan = bad(il, ip)
            fin = ~nan & ~out_of_range
            e = ang_err_m(il[fin], ip[fin], RL[fin], RP[fin], a=P.a)
            print("  %-60s pole %+d: |lat|>90 in %d pts (min |lat| %.4f deg), NaN %d, max err of the rest %.3g m" % (
                defn[:60], int(sgn * 90), int(out_of_range.sum()),
                float(np.abs(lat[ok][out_of_range]).min()) if out_of_range.any() else float("nan"), int(nan.sum()),
                float(e.max()) if e.size else 0.0))
            if out_of_range.any():
                finding("fp32 inverse returns |lat| > 90 deg (e.g. %.3f deg) for %s near pole %+d (band from %.4f deg)" % (
                    math.degrees(float(ip[out_of_range][0])), defn, int(sgn * 90), float(np.abs(lat[ok][out_of_range]).min())))
            if nan.any():
                finding("fp32 inverse NaN at %d valid near-pole points for %s pole %+d" % (int(nan.sum()), defn, int(sgn * 90)))
    print("13b. NaN-gradient radius around the LCC apex (inverse, distance along the central meridian)")
    for defn in ["+proj=lcc +ellps=GRS80 +lat_1=0.5 +lat_2=2", "+proj=lcc +R=6371000 +lat_1=0.5 +lat_2=2",
                 "+proj=lcc +ellps=GRS80 +lat_1=-0.5 +lat_2=-2", "+proj=lcc +ellps=GRS80 +lat_1=5 +lat_2=10"]:
        P = L.make(defn)
        sgn = 1.0 if P.n > 0 else -1.0
        ay = P.rho0 * P.k0 * P.a
        d = np.logspace(-6, 7, 131)
        xy = torch.tensor(np.c_[np.zeros_like(d), ay - sgn * d], dtype=torch.float64, requires_grad=True)
        out = P.inverse(xy)
        g, = torch.autograd.grad(torch.where(torch.isfinite(out), out, torch.zeros_like(out)).sum(), xy)
        nan = ~torch.isfinite(g).all(-1)
        rmax = float(d[nan.numpy()].max()) if nan.any() else 0.0
        print("  %-50s n=%+.4f: NaN gradient up to %.3g m from the apex (%d/%d probes)" % (defn, P.n, rmax, int(nan.sum()), len(d)))
        if nan.any():
            finding("NaN inverse gradient within %.3g m of the apex for %s (outputs finite)" % (rmax, defn))


if __name__ == "__main__":
    main()
