"""Integration layer: plug every torch projection of the spike into pipeline.py.

    import integ_registry
    integ_registry.install()                 # mode="generic" (default) or "raw"
    T = pipeline.create(Transformer.from_crs(...).definition)

What it registers (pipeline.REGISTRY[name] -> Operation subclass):
  * tmerc.py           : tmerc, etmerc, utm
  * lcc_stere.py       : lcc, stere, ups
  * laea_healpix_moll.py: laea, healpix, rhealpix, moll, wag4, wag5
  * geos_obtran.py     : geos, eqc, ob_tran   (geos_obtran's own moll/latlong are
                         NOT registered: laea_healpix_moll's moll is the more complete
                         one and pipeline.py has its own latlong family)
  * this file          : merc, webmerc  -- integrator-added minimal port of
                         src/projections/merc.cpp:13-84, because EPSG:3857 needs
                         webmerc and no module agent ported it.
  pipeline.py's own operations (unitconvert, axisswap, cart, geocent, helmert,
  molodensky, hgridshift, push/pop, affine, ...) are already in pipeline.REGISTRY.
  Every module projection is also registered as an ob_tran link (+o_proj=<name>)
  in geos_obtran.REGISTRY.

Two adapter modes (both are small wrappers; no module is modified):

  mode="raw"     pipeline.ProjectionStep as written by the pipeline agent (the
                 module's forward()/inverse() do ALL of fwd.cpp/inv.cpp), plus the
                 one fix needed for end-to-end pipelines: ob_tran over a latlong
                 link has a PJ_IO_UNITS_WHATEVER right side (ob_tran.cpp:297-298).
                 Without it pipeline.py rejects the COSMO rotated-pole pipeline
                 with "Mismatched units between step 2 and 3".

  mode="generic" (default) the generic PROJ layer is done ONCE, by pipeline.py's
                 _fwd_prepare/_fwd_finalize/_inv_prepare/_inv_finalize (ports of
                 fwd.cpp:40-189, inv.cpp:39-142), around the module's projection
                 kernel; the module only sees lon_0/x_0/y_0/k_0/ellipsoid.
                 This restores, for every module at once, what the adversarial
                 verifications found silently ignored by core.Projection-based
                 modules: +pm, +axis, +units/+to_meter, +vunits/+vto_meter, +z_0,
                 +geoc, +R_A/+R_V/+R_a/+R_g/+R_h spherification, PROJ's shape-key
                 priority and range checks (the module receives a canonical
                 R=... or a=... es=... computed by pipeline.pj_ellipsoid), the |lon| > 10 rad
                 rejection, the pre-adjlon of the raw longitude (fwd.cpp:80-82),
                 PROJ's latitude clamp/exact eps test, PROJ boolean parsing
                 (+south=f, +over=f, +approx=f), and the cs2cs +towgs84/+nadgrids
                 helpers. Order of operations is PROJ's:
                   fwd: [pipeline] lat check+clamp, |lon|>10, adjlon, geoc, datum
                        helpers, lam -= pm, adjlon  ->  [module] lam -= lon_0, adjlon,
                        kernel, *a + x_0  ->  [pipeline] *fr_meter, z = vfr*(z+z0), axis
                   inv: the mirror image.
                 lon_0 and x_0/y_0 stay in the module (P.lam0 = P.x0 = P.y0 = 0 in
                 the pipeline Operation) because some modules use them in
                 non-generic ways (ob_tran WHATEVER output ignores x_0/y_0, HEALPix
                 rescales a, UPS forces x_0/y_0/lon_0, UTM derives lon_0 from zone).
"""
from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pipeline as PL  # noqa: E402
from core import Projection  # noqa: E402

# ---------------------------------------------------------------------------
# Integrator-added projection: merc / webmerc (src/projections/merc.cpp)
# ---------------------------------------------------------------------------


class Mercator(Projection):
    """merc.cpp:13-74 (+proj=merc, ellipsoid and sphere, lat_ts / k_0)."""

    def __init__(self, params):
        super().__init__(params)
        p = self.params
        self.spherical = self.es == 0.0
        if "lat_ts" in p:  # merc.cpp:49-57
            phits = abs(math.radians(float(p["lat_ts"])))
            if phits >= math.pi / 2:
                raise ValueError("Invalid value for lat_ts: |lat_ts| should be <= 90")
            if self.spherical:  # merc.cpp:67-68
                self.k0 = math.cos(phits)
            else:  # merc.cpp:60-61, pj_msfn (msfn.cpp:5-7)
                s, c = math.sin(phits), math.cos(phits)
                self.k0 = c / math.sqrt(1.0 - self.es * s * s)

    def _fwd(self, lam, phi):
        x = self.k0 * lam  # merc.cpp:15 / :26
        if self.spherical:  # merc_s_forward, merc.cpp:24-29
            y = self.k0 * torch.asinh(torch.tan(phi))
        else:  # merc_e_forward, merc.cpp:13-22 (sin/cos instead of tan, as PROJ)
            sphi, cphi = torch.sin(phi), torch.cos(phi)
            y = self.k0 * (torch.asinh(sphi / cphi) - self.e * torch.atanh(self.e * sphi))
        return x, y

    def _inv(self, x, y):
        lam = x / self.k0  # merc.cpp:34 / :41
        if self.spherical:  # merc_s_inverse, merc.cpp:38-43
            phi = torch.atan(torch.sinh(y / self.k0))
        else:  # merc_e_inverse, merc.cpp:31-36: pj_sinhpsi2tanphi (masked Newton, cap 5)
            from lcc_stere import sinhpsi2tanphi
            tau, failed = sinhpsi2tanphi(torch.sinh(y / self.k0), self.e)
            phi = torch.where(failed, torch.full_like(tau, float("nan")), torch.atan(tau))
        return lam, phi


class WebMercator(Mercator):
    """merc.cpp:76-84: spherical formulas on the ellipsoid's a, k_0 forced to 1."""

    def __init__(self, params):
        Projection.__init__(self, params)
        self.k0 = 1.0
        self.spherical = True


def _make_merc(params):
    name = params.get("proj")
    return WebMercator(params) if name == "webmerc" else Mercator(params)


# ---------------------------------------------------------------------------
# What is available
# ---------------------------------------------------------------------------
# name -> (module, factory attribute) ; factories take a core-style params dict
MODULE_PROJECTIONS: Dict[str, Tuple[str, str]] = {
    "tmerc": ("tmerc", "make"), "etmerc": ("tmerc", "make"), "utm": ("tmerc", "make"),
    "lcc": ("lcc_stere", "make"), "stere": ("lcc_stere", "make"), "ups": ("lcc_stere", "make"),
    "laea": ("laea_healpix_moll", "make"), "healpix": ("laea_healpix_moll", "make"),
    "rhealpix": ("laea_healpix_moll", "make"), "moll": ("laea_healpix_moll", "make"),
    "wag4": ("laea_healpix_moll", "make"), "wag5": ("laea_healpix_moll", "make"),
    "geos": ("geos_obtran", "make"), "eqc": ("geos_obtran", "make"), "ob_tran": ("geos_obtran", "make"),
    "merc": ("integ_registry", "_make_merc"), "webmerc": ("integ_registry", "_make_merc"),
}

# ob_tran links (+o_proj=<name>): name -> (module, class attribute)
OBTRAN_LINKS: Dict[str, Tuple[str, str]] = {
    "tmerc": ("tmerc", "TransverseMercator"), "etmerc": ("tmerc", "ExtendedTransverseMercator"),
    "utm": ("tmerc", "UTM"),
    "lcc": ("lcc_stere", "LambertConformalConic"), "stere": ("lcc_stere", "Stereographic"),
    "ups": ("lcc_stere", "UniversalPolarStereographic"),
    "laea": ("laea_healpix_moll", "LambertAzimuthalEqualArea"),
    "healpix": ("laea_healpix_moll", "HEALPix"), "rhealpix": ("laea_healpix_moll", "HEALPix"),
    "wag4": ("laea_healpix_moll", "Mollweide"), "wag5": ("laea_healpix_moll", "Mollweide"),
    "merc": ("integ_registry", "Mercator"), "webmerc": ("integ_registry", "WebMercator"),
}

# keys the generic layer handles itself (stripped before calling the module)
_GENERIC_KEYS = {"pm", "geoc", "axis", "units", "to_meter", "vunits", "vto_meter", "z_0",
                 "towgs84", "nadgrids", "datum", "geoidgrids", "lon_wrap", "inv",
                 "omit_fwd", "omit_inv", "break_cs2cs_recursion", "type", "no_defs", "wktext"}
_BOOL_KEYS = ("over", "south", "approx")
_SHAPE_KEYS = ("ellps", "a", "b", "rf", "f", "e", "es", "R_A", "R_V", "R_a", "R_g", "R_h",
               "R_lat_a", "R_lat_g", "R_C")
_SPHERIFY = ("R_A", "R_V", "R_a", "R_g", "R_h", "R_lat_a", "R_lat_g", "R_C")

AVAILABLE: Dict[str, Callable] = {}
UNAVAILABLE: Dict[str, str] = {}
_ORIGINAL_REGISTRY = dict(PL.REGISTRY)  # pipeline.py as delivered (projections via EXTERNAL_PROJECTIONS)


def uninstall() -> None:
    """Back to pipeline.py as delivered (lazy EXTERNAL_PROJECTIONS -> ProjectionStep)."""
    PL.REGISTRY.clear()
    PL.REGISTRY.update(_ORIGINAL_REGISTRY)


def _resolve(spec: Tuple[str, str]):
    modname, attr = spec
    mod = sys.modules[__name__] if modname == "integ_registry" else importlib.import_module(modname)
    return getattr(mod, attr)


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------
class RawStep(PL.ProjectionStep):
    """pipeline.ProjectionStep + ob_tran-over-latlong WHATEVER right side."""

    def setup(self):
        super().setup()
        if getattr(self.proj, "whatever_units", False):  # ob_tran.cpp:297-298
            self.right = PL.WHATEVER


class GenericStep(PL.ProjectionStep):
    """Module projection kernel wrapped in pipeline.py's fwd.cpp/inv.cpp port."""

    need_ellps = True  # projections require an ellipsoid (PJ_PROJECTION macro): no WGS84 fallback

    def module_params(self) -> dict:
        pl = self.params
        d = pl.as_dict()
        out = {k: v for k, v in d.items() if k not in _GENERIC_KEYS}
        for k in _BOOL_KEYS:  # pj_param 'b' semantics (param.cpp:195-210)
            if k in out:
                if pl.b(k):
                    out[k] = True
                else:
                    del out[k]
        # Canonical ellipsoid: pipeline.pj_ellipsoid (ell_set.cpp: size, shape with PROJ's
        # rf > f > es > e > b priority, b -> es via f, range checks, R_A/R_V/... spherification)
        # has already run in Operation._pj_init; the module gets exactly that ellipsoid.
        for k in _SHAPE_KEYS + ("R",):
            out.pop(k, None)
        if self.es == 0.0:
            out["R"] = repr(self.a)
        else:
            out["a"], out["es"] = repr(self.a), repr(self.es)
        return out

    def setup(self):
        self.left, self.right = PL.RADIANS, PL.PROJECTED
        try:
            self.proj = self.factory(self.module_params())
        except NotImplementedError as exc:
            raise PL.NotSupported(str(exc))
        except ValueError as exc:
            raise PL.ProjError(str(exc))
        except PL.ProjError:
            raise
        except Exception as exc:
            raise PL.NotSupported("%s: %r" % (self.name, exc))
        if getattr(self.proj, "whatever_units", False):  # ob_tran.cpp:297-298
            self.right = PL.WHATEVER
        # lon_0, x_0, y_0 are applied by the module (see module docstring)
        self.lam0 = 0.0
        self.x0 = self.y0 = 0.0
        self.has_inv = hasattr(self.proj, "inverse")
        # generic prepare/finalize run (unlike ProjectionStep)
        self.skip_fwd_prepare = self.skip_fwd_finalize = False
        self.skip_inv_prepare = self.skip_inv_finalize = False

    def fwd4d(self, c, ctx):
        out = self.proj.forward(torch.stack(c[:2], -1))
        return (out[..., 0], out[..., 1], c[2], c[3])

    def inv4d(self, c, ctx):
        out = self.proj.inverse(torch.stack(c[:2], -1))
        return (out[..., 0], out[..., 1], c[2], c[3])


def _subclass(base: type, name: str, factory: Callable) -> type:
    return type("Integ_%s_%s" % (base.__name__, name), (base,), {"factory": staticmethod(factory)})


def install(mode: str = "generic", verbose: bool = False) -> Dict[str, str]:
    """Register every importable torch projection in pipeline.REGISTRY (and as
    ob_tran links). Returns {name: 'module.factory'} of what was registered."""
    if mode not in ("generic", "raw"):
        raise ValueError(mode)
    uninstall()
    base = GenericStep if mode == "generic" else RawStep
    AVAILABLE.clear()
    UNAVAILABLE.clear()
    done = {}
    for name, spec in MODULE_PROJECTIONS.items():
        try:
            fac = _resolve(spec)
        except Exception as exc:  # module missing / broken: leave unregistered
            UNAVAILABLE[name] = "%s.%s: %r" % (spec[0], spec[1], exc)
            PL.REGISTRY.pop(name, None)
            continue
        PL.REGISTRY[name] = _subclass(base, name, fac)
        AVAILABLE[name] = fac
        done[name] = "%s.%s" % spec
    try:
        import geos_obtran
        for name, spec in OBTRAN_LINKS.items():
            try:
                geos_obtran.register_projection(name, _resolve(spec))
            except Exception as exc:
                UNAVAILABLE["o_proj=" + name] = repr(exc)
    except Exception as exc:
        UNAVAILABLE["ob_tran links"] = repr(exc)
    if verbose:
        print("registered (%s): %s" % (mode, ", ".join(sorted(done))))
        if UNAVAILABLE:
            print("unavailable:", UNAVAILABLE)
    return done


def registered_operations() -> Dict[str, str]:
    """Everything pipeline.create() can build after install()."""
    out = {}
    for name, cls in PL.REGISTRY.items():
        out[name] = cls.__name__
    return out
