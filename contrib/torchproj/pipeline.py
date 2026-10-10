"""PROJ execution layer (pipelines + basic conversions/transformations) in PyTorch.

Faithful port of (file:line refer to src/ of this PROJ tree):
  * internal.cpp:156-372 pj_shrink/pj_trim_argc/argv  (PROJ string tokenizer)
  * param.cpp:66-218      pj_param / pj_param_exists   (ParamList)
  * dmstor.cpp:27-118     dmstor                        (DMS angles)
  * ell_set.cpp, ellps.cpp, datums.cpp, datum_set.cpp, init.cpp:330-790 (generic op setup)
  * fwd.cpp:40-311, inv.cpp:39-142  (prepare / finalize around every operation)
  * create.cpp:49-205     cs2cs emulation (+towgs84 / +nadgrids / +axis / geocent helpers)
  * pipeline.cpp          pipeline driver, +step/+inv/+omit_fwd/+omit_inv, push/pop
  * conversions/{unitconvert,axisswap,cart,geocent,geoc,noop,set}.cpp, projections/latlong.cpp
  * transformations/{helmert,molodensky,affine,hgridshift}.cpp, grids.cpp (NTv2 + hgrid apply)

Conventions (see core.py): coordinates are tensors [..., D] (2 <= D <= 4), geodetic
input in RADIANS, linear in metres. Missing z is 0, missing t is +inf (HUGE_VAL, as
proj_trans_generic does when no time array is given). A failed point is NaN in all
components (PROJ: HUGE_VAL + errno). forward()/inverse() contain no data dependent
Python control flow; all data dependent branches are torch.where (with the "double
where" trick where an untaken branch could produce inf/NaN), iterative algorithms run a
fixed number of masked iterations equal to PROJ's cap.
Projections written by other modules (tmerc.py, lcc_stere.py, ...) are plugged in via
EXTERNAL_PROJECTIONS (lazy import) or register_operation().

Usage:
    T = pipeline.create(pyproj.Transformer.from_crs("EPSG:4326", "EPSG:4978").definition)
    xyz = T.forward(torch.tensor([[52.0, 4.0, 10.0]], dtype=torch.float64))  # lat, lon (deg), h
    llh = T.inverse(xyz)
    T.angular_input(), T.has_inverse, ...   # PROJ's left/right unit bookkeeping
Errors: ProjError (PROJ would refuse the definition), NotSupported (not ported).
"""
from __future__ import annotations

import importlib
import math
import os
import re
import struct
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

_PROJ_ROOT = Path(__file__).resolve().parents[2]  # root of the PROJ source tree

Tensor = torch.Tensor
Coord = Tuple[Tensor, Tensor, Tensor, Tensor]

# proj_internal.h / proj.h constants
M_PI = math.pi
M_HALFPI = math.pi / 2
M_TWOPI = 2 * math.pi
DEG_TO_RAD = 0.017453292519943296
PJ_EPS_LAT = 1e-12
ARCSEC_TO_RAD = DEG_TO_RAD / 3600.0          # helmert.cpp:477
SEC_TO_RAD = 4.84813681109535993589914102357e-6  # datum_set.cpp:34
INF = math.inf
NAN = math.nan

# enum pj_io_units (proj_internal.h:192-201)
WHATEVER, CLASSIC, PROJECTED, CARTESIAN, RADIANS, DEGREES = range(6)
UNIT_NAMES = ["whatever", "classic", "projected", "cartesian", "radians", "degrees"]


class ProjError(ValueError):
    """PROJ would refuse to create the operation (errno name in .errno)."""

    def __init__(self, msg: str, errno: str = "invalid_op_illegal_arg_value"):
        super().__init__(msg)
        self.errno = errno


class NotSupported(NotImplementedError):
    """Valid PROJ, but not ported in this spike."""


# =============================================================================
# String handling: internal.cpp:156-372
# =============================================================================
def pj_shrink(c: str) -> str:
    """internal.cpp:156-277 (pj_chomp + pj_shrink)."""
    c = c.split("#", 1)[0]
    c = c.rstrip(" \t\n\r\f\v;").lstrip(" \t\n\r\f\v;")
    out: List[str] = []
    ws = False
    in_string = False
    j, n = 0, len(c)
    while j < n:
        ch = c[j]
        if in_string:
            if ch == '"' and j + 1 < n and c[j + 1] == '"':
                out.append(ch)
                j += 1
            elif ch == '"':
                in_string = False
            out.append(c[j])
            j += 1
            continue
        if out and ch == "+" and ws:
            ch = " "
        if not out and ch == "+":
            ch = " "
        if ch == '"' and out and out[-1] == "=":
            in_string, ws = True, False
            out.append(ch)
            j += 1
            continue
        if ch.isspace() or ch == ";":
            if not ws and out:
                out.append(" ")
            ws = True
            j += 1
            continue
        ws = False
        out.append(ch)
        j += 1
    res: List[str] = []
    for ch in out:  # make ',' and '=' greedy
        if not res:
            res.append(ch)
        elif ch in "=,":
            if res[-1] == " ":
                res[-1] = ch
            else:
                res.append(ch)
        elif ch == " " and res[-1] in "=,":
            continue
        else:
            res.append(ch)
    return "".join(res)


def proj_tokenize(definition: str) -> List[str]:
    """pj_trim_argc + pj_trim_argv (internal.cpp:281-372): string -> argv list."""
    s = pj_shrink(definition)
    if not s:
        return []
    args, cur, in_string, i = [], [], False, 0
    while i < len(s):
        ch = s[i]
        if in_string:
            if ch == '"' and i + 1 < len(s) and s[i + 1] == '"':
                cur.append(ch)
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == "=" and i + 1 < len(s) and s[i + 1] == '"':
            cur.append(ch)
            i += 1
            ch = s[i]
            in_string = True
        elif ch == " ":
            args.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    args.append("".join(cur))
    out = []
    for a in args:  # unquote_string (internal.cpp:307-330)
        k = a.find('="')
        if k >= 0 and a.endswith('"') and len(a) > k + 2:
            body = a[k + 2:-1].replace('""', '"')
            a = a[:k + 1] + body
        out.append(a)
    return out


# =============================================================================
# Numbers / angles: strtod.cpp, dmstor.cpp
# =============================================================================
_STRTOD = re.compile(r"\s*[+-]?(?:(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|inf(?:inity)?|nan)", re.I)


def pj_strtod(s: str) -> Tuple[float, int]:
    """C strtod on the longest numeric prefix -> (value, n_chars_consumed)."""
    m = _STRTOD.match(s)
    if not m:
        return 0.0, 0
    return float(m.group(0)), m.end()


def pj_atof(s: str) -> float:
    return pj_strtod(s)[0]


_DMS_VM = (DEG_TO_RAD, .0002908882086657216, .0000048481368110953599)


def dmstor(s: str) -> float:
    """dmstor.cpp:27-118: DMS string (or plain degrees, or 'r' radians) -> radians."""
    work = "".join(ch for ch in s.lstrip() if ch.isprintable() and not ch.isspace())
    i, sign = 0, "+"
    if work[:1] in ("+", "-"):
        sign, i = work[0], 1
    v, nl = 0.0, 0
    while nl < 3:
        if i >= len(work) or not (work[i].isdigit() or work[i] == "."):
            break
        rest = work[i:]
        cut = min([k for k in (rest.find("d"), rest.find("D")) if k >= 0] or [len(rest)])
        tv, n = pj_strtod(rest[:cut])
        if tv == INF:
            return tv
        i += n
        ch = work[i] if i < len(work) else ""
        if ch in ("D", "d", "\xb0"):
            nn = 0
        elif ch == "'":
            nn = 1
        elif ch == '"':
            nn = 2
        elif ch in ("r", "R"):
            if nl:
                raise ProjError("invalid DMS value %r" % s)
            i += 1
            v = tv
            nl = 5
            continue
        else:
            v += tv * _DMS_VM[nl]
            nl = 5
            continue
        if nn < nl:
            raise ProjError("invalid DMS value %r" % s)
        v += tv * _DMS_VM[nn]
        i += 1
        nl = nn + 1
    if i < len(work) and work[i] in "NnEeSsWw":
        sign = "-" if "NnEeSsWw".index(work[i]) >= 4 else "+"
    return -v if sign == "-" else v


# =============================================================================
# Parameter lists: param.cpp:66-218
# =============================================================================
class ParamList:
    """A PROJ paralist: first occurrence of a key wins (pj_param_exists)."""

    def __init__(self, items: List[str]):
        self.items = [it[1:] if it.startswith("+") else it for it in items]

    def find(self, key: str) -> Optional[str]:
        n = len(key)
        for it in self.items:
            if it.startswith(key) and (len(it) == n or it[n] == "="):
                return it
        return None

    def exists(self, key: str) -> bool:
        return self.find(key) is not None

    def s(self, key: str) -> Optional[str]:
        it = self.find(key)
        if it is None:
            return None
        return it[len(key) + 1:]

    def d(self, key: str, default: float = 0.0) -> float:
        v = self.s(key)
        return default if v is None else pj_atof(v)

    def r(self, key: str) -> float:
        v = self.s(key)
        return 0.0 if v is None else dmstor(v)

    def b(self, key: str) -> bool:
        v = self.s(key)
        if v is None:
            return False
        if v[:1] in ("F", "f"):
            return False
        if v == "" or v[:1] in ("T", "t"):
            return True
        raise ProjError("invalid boolean value for %s" % key)

    def append(self, item: str) -> None:
        self.items.append(item[1:] if item.startswith("+") else item)

    def as_dict(self) -> Dict[str, object]:
        """core.parse_proj_string-style dict (first occurrence wins, flags -> True)."""
        out: Dict[str, object] = {}
        for it in self.items:
            k, eq, v = it.partition("=")
            if k not in out:
                out[k] = v if eq else True
        return out


# =============================================================================
# Units (units.cpp), ellipsoids (ellps.cpp), datums & prime meridians (datums.cpp)
# =============================================================================
LINEAR_UNITS = [  # id, to_meter string, name, factor   (units.cpp:12-35)
    ("km", "1000", "Kilometer", 1000.0), ("m", "1", "Meter", 1.0),
    ("dm", "1/10", "Decimeter", 0.1), ("cm", "1/100", "Centimeter", 0.01),
    ("mm", "1/1000", "Millimeter", 0.001), ("kmi", "1852", "International Nautical Mile", 1852.0),
    ("in", "0.0254", "International Inch", 0.0254), ("ft", "0.3048", "International Foot", 0.3048),
    ("yd", "0.9144", "International Yard", 0.9144), ("mi", "1609.344", "International Statute Mile", 1609.344),
    ("fath", "1.8288", "International Fathom", 1.8288), ("ch", "20.1168", "International Chain", 20.1168),
    ("link", "0.201168", "International Link", 0.201168),
    ("us-in", "1/39.37", "U.S. Surveyor's Inch", 100 / 3937.0),
    ("us-ft", "0.304800609601219", "U.S. Surveyor's Foot", 1200 / 3937.0),
    ("us-yd", "0.914401828803658", "U.S. Surveyor's Yard", 3600 / 3937.0),
    ("us-ch", "20.11684023368047", "U.S. Surveyor's Chain", 79200 / 3937.0),
    ("us-mi", "1609.347218694437", "U.S. Surveyor's Statute Mile", 6336000 / 3937.0),
    ("ind-yd", "0.91439523", "Indian Yard", 0.91439523), ("ind-ft", "0.30479841", "Indian Foot", 0.30479841),
    ("ind-ch", "20.11669506", "Indian Chain", 20.11669506)]
ANGULAR_UNITS = [("rad", "1.0", "Radian", 1.0), ("deg", "0.017453292519943296", "Degree", DEG_TO_RAD),
                 ("grad", "0.015707963267948967", "Grad", 0.015707963267948967)]

ELLPS = {  # ellps.cpp:9-55
    "MERIT": ("a=6378137.0", "rf=298.257"), "SGS85": ("a=6378136.0", "rf=298.257"),
    "GRS80": ("a=6378137.0", "rf=298.257222101"), "IAU76": ("a=6378140.0", "rf=298.257"),
    "airy": ("a=6377563.396", "rf=299.3249646"), "APL4.9": ("a=6378137.0", "rf=298.25"),
    "NWL9D": ("a=6378145.0", "rf=298.25"), "mod_airy": ("a=6377340.189", "b=6356034.446"),
    "andrae": ("a=6377104.43", "rf=300.0"), "danish": ("a=6377019.2563", "rf=300.0"),
    "aust_SA": ("a=6378160.0", "rf=298.25"), "GRS67": ("a=6378160.0", "rf=298.2471674270"),
    "GSK2011": ("a=6378136.5", "rf=298.2564151"), "bessel": ("a=6377397.155", "rf=299.1528128"),
    "bess_nam": ("a=6377483.865", "rf=299.1528128"), "clrk66": ("a=6378206.4", "b=6356583.8"),
    "clrk80": ("a=6378249.145", "rf=293.4663"), "clrk80ign": ("a=6378249.2", "rf=293.4660212936269"),
    "CPM": ("a=6375738.7", "rf=334.29"), "delmbr": ("a=6376428.", "rf=311.5"),
    "engelis": ("a=6378136.05", "rf=298.2566"), "evrst30": ("a=6377276.345", "rf=300.8017"),
    "evrst48": ("a=6377304.063", "rf=300.8017"), "evrst56": ("a=6377301.243", "rf=300.8017"),
    "evrst69": ("a=6377295.664", "rf=300.8017"), "evrstSS": ("a=6377298.556", "rf=300.8017"),
    "fschr60": ("a=6378166.", "rf=298.3"), "fschr60m": ("a=6378155.", "rf=298.3"),
    "fschr68": ("a=6378150.", "rf=298.3"), "helmert": ("a=6378200.", "rf=298.3"),
    "hough": ("a=6378270.0", "rf=297."), "intl": ("a=6378388.0", "rf=297."),
    "krass": ("a=6378245.0", "rf=298.3"), "kaula": ("a=6378163.", "rf=298.24"),
    "lerch": ("a=6378139.", "rf=298.257"), "mprts": ("a=6397300.", "rf=191."),
    "new_intl": ("a=6378157.5", "b=6356772.2"), "plessis": ("a=6376523.", "b=6355863."),
    "PZ90": ("a=6378136.0", "rf=298.25784"), "SEasia": ("a=6378155.0", "b=6356773.3205"),
    "walbeck": ("a=6376896.0", "b=6355834.8467"), "WGS60": ("a=6378165.0", "rf=298.3"),
    "WGS66": ("a=6378145.0", "rf=298.25"), "WGS72": ("a=6378135.0", "rf=298.26"),
    "WGS84": ("a=6378137.0", "rf=298.257223563"), "sphere": ("a=6370997.0", "b=6370997.0")}

DATUMS = {  # datums.cpp:38-59: id -> (defn, ellps)
    "WGS84": ("towgs84=0,0,0", "WGS84"), "GGRS87": ("towgs84=-199.87,74.79,246.62", "GRS80"),
    "NAD83": ("towgs84=0,0,0", "GRS80"),
    "NAD27": ("nadgrids=@conus,@alaska,@ntv2_0.gsb,@ntv1_can.dat", "clrk66"),
    "potsdam": ("nadgrids=@BETA2007.gsb", "bessel"), "carthage": ("towgs84=-263.0,6.0,431.0", "clrk80ign"),
    "hermannskogel": ("towgs84=577.326,90.129,463.919,5.137,1.474,5.297,2.4232", "bessel"),
    "ire65": ("towgs84=482.530,-130.596,564.557,-1.042,-0.214,-0.631,8.15", "mod_airy"),
    "nzgd49": ("towgs84=59.47,-5.04,187.44,0.47,-0.1,1.024,-4.5993", "intl"),
    "OSGB36": ("towgs84=446.448,-125.157,542.060,0.1502,0.2470,0.8421,-20.4894", "airy")}

PRIME_MERIDIANS = {  # datums.cpp:69-87
    "greenwich": "0dE", "lisbon": "9d07'54.862\"W", "paris": "2d20'14.025\"E",
    "bogota": "74d04'51.3\"W", "madrid": "3d41'16.58\"W", "rome": "12d27'8.4\"E",
    "bern": "7d26'22.5\"E", "jakarta": "106d48'27.79\"E", "ferro": "17d40'W",
    "brussels": "4d22'4.71\"E", "stockholm": "18d3'29.8\"E", "athens": "23d42'58.815\"E",
    "oslo": "10d43'22.5\"E", "copenhagen": "12d34'40.35\"E"}


class Ellipsoid:
    """Result of pj_ellipsoid + pj_calc_ellipsoid_params (ell_set.cpp:22-134, 550-630)."""

    def __init__(self):
        self.a = self.es = self.e = self.f = self.b = self.rf = 0.0

    def calc(self, a: float, es: float) -> None:  # ell_set.cpp:550-630
        self.a, self.es = a, es
        if self.e == 0:
            self.e = math.sqrt(self.es)
        self.alpha = math.asin(self.e)
        self.e2 = math.tan(self.alpha)
        self.e2s = self.e2 * self.e2
        if self.f == 0:
            self.f = 1 - math.cos(self.alpha)
        if not (0.0 <= self.f < 1.0):
            raise ProjError("Invalid eccentricity")
        self.rf = 1.0 / self.f if self.f != 0.0 else INF
        self.n = math.tan(self.alpha / 2) ** 2
        if self.b == 0:
            self.b = (1 - self.f) * self.a
        self.rb = 1.0 / self.b
        self.ra = 1.0 / self.a
        self.one_es = 1.0 - self.es
        if self.one_es == 0.0:
            raise ProjError("Invalid eccentricity")
        self.rone_es = 1.0 / self.one_es


def _ellps_size(E: Ellipsoid, pl: ParamList) -> None:  # ell_set.cpp:198-240
    key = "R" if pl.exists("R") else ("a" if pl.exists("a") else None)
    if key is None:
        if E.a != 0:
            return
        raise ProjError("Major axis not given", "invalid_op_missing_arg")
    E.a = pj_atof(pl.s(key))
    if not (E.a > 0) or E.a == INF:
        raise ProjError("Invalid value for major axis")
    if key == "R":
        E.es = E.f = E.e = E.rf = 0.0
        E.b = E.a


def _ellps_shape(E: Ellipsoid, pl: ParamList) -> None:  # ell_set.cpp:243-346
    key = next((k for k in ("rf", "f", "es", "e", "b") if pl.exists(k)), None)
    if key is None:
        if E.es != 0:
            return
        E.es = E.f = 0.0
        E.b = E.a
        return
    E.es = E.f = E.b = E.e = E.rf = 0.0
    v = pj_atof(pl.s(key))
    if key == "rf":
        if v == INF or v <= 0:
            raise ProjError("Invalid value for rf. Should be > 0")
        E.rf = v
        E.f = 1 / v
        E.es = 2 * E.f - E.f * E.f
    elif key == "f":
        if v == INF or v < 0:
            raise ProjError("Invalid value for f. Should be >= 0")
        E.f = v
        E.rf = 1.0 / v if v != 0.0 else INF
        E.es = 2 * E.f - E.f * E.f
    elif key == "es":
        if v == INF or v < 0 or v >= 1:
            raise ProjError("Invalid value for es. Should be in [0,1[ range")
        E.es = v
    elif key == "e":
        if v == INF or v < 0 or v >= 1:
            raise ProjError("Invalid value for e. Should be in [0,1[ range")
        E.e = v
        E.es = v * v
    else:
        if v == INF or v <= 0:
            raise ProjError("Invalid value for b. Should be > 0")
        E.b = v
        if v != E.a:
            E.f = (E.a - v) / E.a
            E.es = 2 * E.f - E.f * E.f
    if not (E.es >= 0):
        raise ProjError("Invalid eccentricity")


def pj_ellipsoid(pl: ParamList) -> Ellipsoid:
    """ell_set.cpp:22-134 (size, shape and the simple spherifications)."""
    E = Ellipsoid()
    if pl.exists("R"):
        _ellps_size(E, pl)
        E.calc(E.a, 0.0)
        return E
    name = pl.s("ellps")
    if name is not None:  # ellps_ellps, ell_set.cpp:137-195
        if name not in ELLPS:
            raise ProjError("Unrecognized value for +ellps")
        tmp = ParamList(list(ELLPS[name]))
        _ellps_size(E, tmp)
        _ellps_shape(E, tmp)
    _ellps_size(E, pl)
    _ellps_shape(E, pl)
    E.calc(E.a, E.es)
    sph = next((k for k in ("R_A", "R_V", "R_a", "R_g", "R_h", "R_lat_a", "R_lat_g", "R_C")
                if pl.exists(k)), None)
    if sph is not None:  # ell_set.cpp:356-466
        es = E.es
        if sph == "R_A":
            E.a *= 1. - es * (1 / 6. + es * (17 / 360. + es * 67 / 3024.))
        elif sph == "R_V":
            E.a *= 1. - es * (1 / 6. + es * (5 / 72. + es * 55 / 1296.))
        elif sph == "R_a":
            E.a = (E.a + E.b) / 2
        elif sph == "R_g":
            E.a = math.sqrt(E.a * E.b)
        elif sph == "R_h":
            E.a = (2 * E.a * E.b) / (E.a + E.b)
        else:
            raise NotSupported("spherification " + sph)
        E.es = E.e = E.f = 0.0
        E.rf = INF
        E.b = E.a
        E.calc(E.a, 0.0)
    return E


# =============================================================================
# Tensor helpers
# =============================================================================
def adjlon(lon: Tensor) -> Tensor:
    """adjlon.cpp:9-21 (identity, and identity gradient, inside +-(pi+1e-12))."""
    w = lon + M_PI
    w = w - M_TWOPI * torch.floor(w / M_TWOPI)
    w = w - M_PI
    return torch.where(lon.abs() < M_PI + 1e-12, lon, w)


def _err(mask: Tensor, c: Coord) -> Coord:
    """proj_coord_error() for the points in mask."""
    return tuple(torch.where(mask, NAN, v) for v in c)  # type: ignore


def _sqrt0(x: Tensor) -> Tensor:
    """sqrt with a zero (instead of inf) gradient at x == 0 (double where)."""
    pos = x > 0
    return torch.where(pos, torch.sqrt(torch.where(pos, x, 1.0)), x * 0.0)


def _nonzero_div(num: Tensor, den: Tensor) -> Tuple[Tensor, Tensor]:
    """(num/den, den == 0) with den == 0 kept out of the gradient (double where)."""
    zero = den == 0
    return num / torch.where(zero, 1.0, den), zero


class _ExecContext:
    """Per call state of a pipeline: the 4 coordinate stacks (pipeline.cpp:135-140)."""

    def __init__(self):
        self.stack: List[List[Tensor]] = [[], [], [], []]


# =============================================================================
# Generic operation: init.cpp:540-790 + create.cpp:49-205 + fwd.cpp/inv.cpp
# =============================================================================
class Operation(torch.nn.Module):
    """A PROJ PJ object. Subclasses implement setup(), fwd4d(c, ctx), inv4d(c, ctx)."""

    need_ellps = False
    has_fwd = True
    has_inv = True

    def __init__(self, params: ParamList):
        super().__init__()
        self.params = params
        self.name = params.s("proj") or ""
        self.inverted = False
        self.left = WHATEVER
        self.right = WHATEVER
        self.skip_fwd_prepare = self.skip_fwd_finalize = False
        self.skip_inv_prepare = self.skip_inv_finalize = False
        self.is_geocent = self.is_latlong = False
        self.h_helmert: Optional[Operation] = None
        self.h_cart: Optional[Operation] = None
        self.h_cart_wgs84: Optional[Operation] = None
        self.h_hgridshift: Optional[Operation] = None
        self.h_axisswap: Optional[Operation] = None
        self._pj_init()
        self.setup()
        self._cs2cs_emulation_setup()

    # ---- init.cpp:540-790 -------------------------------------------------
    def _pj_init(self) -> None:
        pl = self.params
        dname = pl.s("datum")
        if dname is not None:  # datum_set.cpp:56-108
            if dname not in DATUMS:
                raise ProjError("Unknown value for datum")
            defn, ellps = DATUMS[dname]
            pl.append("ellps=" + ellps)
            pl.append(defn)
        self.datum_params = [0.0] * 7
        self.datum_type = "unknown"
        if pl.s("nadgrids") is not None:
            self.datum_type = "gridshift"
        elif pl.s("towgs84") is not None:  # datum_set.cpp:121-158
            vals = pl.s("towgs84").split(",")[:7]
            for k, v in enumerate(vals):
                self.datum_params[k] = pj_atof(v)
            dp = self.datum_params
            if dp[3] != 0.0 or dp[4] != 0.0 or dp[5] != 0.0 or dp[6] != 0.0:
                self.datum_type = "7param"
                dp[3] *= SEC_TO_RAD
                dp[4] *= SEC_TO_RAD
                dp[5] *= SEC_TO_RAD
                dp[6] = (dp[6] / 1000000.0) + 1
            else:
                self.datum_type = "3param"
        try:
            self.ell = pj_ellipsoid(pl)
        except ProjError:
            if self.need_ellps:
                raise
            self.ell = Ellipsoid()
            self.ell.f = 1.0 / 298.257223563
            self.ell.calc(6378137.0, self.ell.f * (2 - self.ell.f))
        E = self.ell
        self.a, self.es, self.e, self.f, self.ra = E.a, E.es, E.e, E.f, E.ra
        self.a_orig, self.es_orig = E.a, E.es
        self.geoc = E.es != 0.0 and pl.b("geoc")
        self.over = pl.b("over")
        self.is_long_wrap_set = pl.exists("lon_wrap")
        self.long_wrap_center = 0.0
        if self.is_long_wrap_set:
            self.long_wrap_center = pl.r("lon_wrap")
            if not (abs(self.long_wrap_center) < 10 * M_TWOPI):
                raise ProjError("Invalid value for lon_wrap")
        self.axis = "enu"
        ax = pl.s("axis")
        if ax is not None:
            if len(ax) != 3 or any(ch not in "ewnsud" for ch in ax):
                raise ProjError("Invalid value for axis")
            self.axis = ax
        self.lam0 = pl.r("lon_0")
        self.phi0 = pl.r("lat_0")
        if abs(self.phi0) > M_HALFPI:
            raise ProjError("Invalid value for lat_0: |lat_0| should be <= 90")
        self.x0, self.y0, self.z0 = pl.d("x_0"), pl.d("y_0"), pl.d("z_0")
        self.k0 = pl.d("k_0") if pl.exists("k_0") else (pl.d("k") if pl.exists("k") else 1.0)
        if self.k0 <= 0:
            raise ProjError("Invalid value for k/k_0: it should be > 0")
        self.to_meter = self.fr_meter = 1.0
        s = self._units_string("units", "to_meter")
        if s is not None:
            self.to_meter = _parse_to_meter(s)
            self.fr_meter = 1 / self.to_meter
        self.vto_meter, self.vfr_meter = self.to_meter, self.fr_meter
        s = self._units_string("vunits", "vto_meter")
        if s is not None:
            self.vto_meter = _parse_to_meter(s)
            self.vfr_meter = 1. / self.vto_meter
        self.from_greenwich = 0.0
        pm = pl.s("pm")
        if pm is not None:
            if pm in PRIME_MERIDIANS:
                pm = PRIME_MERIDIANS[pm]
            self.from_greenwich = dmstor(pm)

    def _units_string(self, ukey: str, mkey: str) -> Optional[str]:
        name = self.params.s(ukey)
        if name is not None:
            for uid, to_m, _, _ in LINEAR_UNITS:
                if uid == name:
                    return to_m
            raise ProjError("Invalid value for " + ukey)
        return self.params.s(mkey)

    # ---- create.cpp:49-205 ------------------------------------------------
    def _cs2cs_emulation_setup(self) -> None:
        pl = self.params
        if pl.exists("break_cs2cs_recursion"):
            return
        if pl.exists("axis"):
            self.h_axisswap = _helper("proj=axisswap axis=" + self.axis)
        if pl.s("geoidgrids"):
            raise NotSupported("geoidgrids (vgridshift) not ported")
        nad = pl.s("nadgrids")
        if nad:
            self.h_hgridshift = _helper("proj=hgridshift grids=" + nad)
        do_cart = False
        if self.h_hgridshift is None and pl.exists("towgs84"):
            d = self.datum_params
            if all(v == 0 for v in d):
                if not (abs(self.a_orig - 6378137.0) < 1e-8 and abs(self.es_orig - 0.0066943799901413) < 1e-15):
                    do_cart = True
            else:
                h = _helper("proj=helmert exact towgs84=%s convention=position_vector" % pl.s("towgs84"))
                self.h_helmert = h
        if self.is_geocent or self.h_helmert is not None or do_cart:
            self.h_cart = _helper("proj=cart a=%r es=%r" % (self.a_orig, self.es_orig))
            if not self.is_geocent:
                self.h_cart_wgs84 = _helper("proj=cart ellps=WGS84")

    # ---- to be implemented -------------------------------------------------
    def setup(self) -> None:
        pass

    def fwd4d(self, c: Coord, ctx: _ExecContext) -> Coord:
        raise NotImplementedError

    def inv4d(self, c: Coord, ctx: _ExecContext) -> Coord:
        raise NotImplementedError

    # ---- pj_left / pj_right (internal.cpp:49-61) ---------------------------
    def pj_left(self) -> int:
        u = self.right if self.inverted else self.left
        return PROJECTED if u == CLASSIC else u

    def pj_right(self) -> int:
        u = self.left if self.inverted else self.right
        return PROJECTED if u == CLASSIC else u

    def has_inverse(self) -> bool:  # internal.cpp:130-136
        return (self.inverted and self.has_fwd) or self.has_inv


def _helper(defn: str) -> Operation:
    """A cs2cs helper (create.cpp:31-41: skip_prep_fin)."""
    op = _create_op(["break_cs2cs_recursion"] + proj_tokenize(defn))
    op.skip_fwd_prepare = op.skip_fwd_finalize = True
    op.skip_inv_prepare = op.skip_inv_finalize = True
    return op


def _parse_to_meter(s: str) -> float:  # init.cpp:693-713
    v, n = pj_strtod(s)
    if s[n:n + 1] == "/":
        den = pj_atof(s[n + 1:])
        if den == 0.0:
            raise ProjError("Invalid value for to_meter denominator")
        v /= den
    if not (v > 0.0):
        raise ProjError("Invalid value for to_meter")
    return v


def _geocentric_latitude(P: Operation, fwd: bool, phi: Tensor) -> Tensor:
    """geoc.cpp:37-64."""
    limit = M_HALFPI - 1e-9
    if P.es == 0:
        return phi
    keep = (phi > limit) | (phi < -limit)
    ps = torch.where(keep, 0.0, phi)
    k = P.ell.one_es if fwd else P.ell.rone_es
    return torch.where(keep, phi, torch.atan(k * torch.tan(ps)))


def proj_trans(P: Operation, fwd: bool, c: Coord, ctx: Optional[_ExecContext] = None) -> Coord:
    """trans.cpp:214 (no alternative operations): honours P.inverted."""
    ctx = ctx or _ExecContext()
    if P.inverted:
        fwd = not fwd
    return pj_fwd4d(P, c, ctx) if fwd else pj_inv4d(P, c, ctx)


def _fwd_prepare(P: Operation, c: Coord) -> Coord:
    """fwd.cpp:40-136."""
    x, y, z, t = c
    if P.left == RADIANS:
        over = y.abs() > M_HALFPI
        err = over & ((x == INF) | (y == INF))
        err = err | (over & (y > 0) & (y - M_HALFPI > PJ_EPS_LAT))
        err = err | (over & ~(y > 0) & (y - M_HALFPI < -PJ_EPS_LAT))  # sic: PROJ tests phi - pi/2
        y = torch.where(over, torch.where(y > 0, M_HALFPI, torch.full_like(y, -M_HALFPI)), y)
        err = err | ((x.abs() > M_PI) & (x.abs() > 10))
        if not P.over:
            x = adjlon(x)
        err = err | (z == INF)
        c = _err(err, (x, y, z, t))
        if P.geoc:
            c = (c[0], _geocentric_latitude(P, False, c[1]), c[2], c[3])
        if P.h_hgridshift is not None:
            c = proj_trans(P.h_hgridshift, False, c)
        elif P.h_helmert is not None or (P.h_cart_wgs84 is not None and P.h_cart is not None):
            c = proj_trans(P.h_cart_wgs84, True, c)
            if P.h_helmert is not None:
                c = proj_trans(P.h_helmert, False, c)
            c = proj_trans(P.h_cart, False, c)
        x, y, z, t = c
        if P.from_greenwich != 0.0 or P.lam0 != 0.0:
            x = (x - P.from_greenwich) - P.lam0
        if not P.over:
            x = adjlon(x)
        return (x, y, z, t)
    err = (x == INF) | (y == INF) | (z == INF)
    c = _err(err, c)
    if P.left == CARTESIAN and P.h_helmert is not None:
        c = proj_trans(P.h_helmert, False, c)
    return c


def _fwd_finalize(P: Operation, c: Coord) -> Coord:
    """fwd.cpp:138-189."""
    out = P.right
    if out == CARTESIAN:
        if P.is_geocent:
            c = proj_trans(P.h_cart, True, c)
        if P.fr_meter != 1.0:
            c = (c[0] * P.fr_meter, c[1] * P.fr_meter, c[2] * P.fr_meter, c[3])
    elif out in (CLASSIC, PROJECTED):
        x, y, z, t = c
        if out == CLASSIC:
            x, y = x * P.a, y * P.a
        x = P.fr_meter * (x + P.x0)
        y = P.fr_meter * (y + P.y0)
        z = P.vfr_meter * (z + P.z0)
        c = (x, y, z, t)
    elif out == RADIANS:
        x, y, z, t = c
        if P.vfr_meter != 1.0 or P.z0 != 0.0:
            z = P.vfr_meter * (z + P.z0)
        if P.is_long_wrap_set:
            x = P.long_wrap_center + adjlon(x - P.long_wrap_center)
        c = (x, y, z, t)
    if P.h_axisswap is not None:
        c = proj_trans(P.h_axisswap, True, c)
    return c


def _inv_prepare(P: Operation, c: Coord) -> Coord:
    """inv.cpp:39-100."""
    x, y, z, t = c
    c = _err((x == INF) | (y == INF) | (z == INF), c)
    if P.h_helmert is not None:
        c = (c[0], c[1], c[2], torch.where(c[3] == INF, 0.0, c[3]))
    if P.h_axisswap is not None:
        c = proj_trans(P.h_axisswap, False, c)
    x, y, z, t = c
    inp = P.right
    if inp == CARTESIAN:
        if P.to_meter != 1.0:
            x, y, z = x * P.to_meter, y * P.to_meter, z * P.to_meter
        c = (x, y, z, t)
        if P.is_geocent:
            c = proj_trans(P.h_cart, False, c)
    elif inp in (PROJECTED, CLASSIC):
        x = P.to_meter * x - P.x0
        y = P.to_meter * y - P.y0
        z = P.vto_meter * z - P.z0
        if inp == CLASSIC:
            x, y = x * P.ra, y * P.ra
        c = (x, y, z, t)
    elif inp == RADIANS:
        if P.vto_meter != 1.0 or P.z0 != 0.0:
            z = P.vto_meter * z - P.z0
        c = (x, y, z, t)
    return c


def _inv_finalize(P: Operation, c: Coord) -> Coord:
    """inv.cpp:102-142."""
    if P.left == RADIANS:
        x, y, z, t = c
        if P.from_greenwich != 0.0 or P.lam0 != 0.0:
            x = x + P.from_greenwich + P.lam0
        if not P.over:
            x = adjlon(x)
        c = (x, y, z, t)
        if P.h_hgridshift is not None:
            c = proj_trans(P.h_hgridshift, True, c)
        elif P.h_helmert is not None or (P.h_cart_wgs84 is not None and P.h_cart is not None):
            c = proj_trans(P.h_cart, True, c)
            if P.h_helmert is not None:
                c = proj_trans(P.h_helmert, True, c)
            c = proj_trans(P.h_cart_wgs84, False, c)
        if P.geoc:
            c = (c[0], _geocentric_latitude(P, True, c[1]), c[2], c[3])
    return c


def pj_fwd4d(P: Operation, c: Coord, ctx: _ExecContext) -> Coord:
    """fwd.cpp:270-311. Errors: NaN in x (mapped to a full NaN coordinate by the caller)."""
    if not P.skip_fwd_prepare:
        c = _fwd_prepare(P, c)
    if not P.has_fwd:  # PROJ_ERR_OTHER_NO_INVERSE_OP
        return _err(torch.ones_like(c[0], dtype=torch.bool), c)
    c = P.fwd4d(c, ctx)
    c = _err(torch.isnan(c[0]), c)  # HUGE_VAL == x -> proj_coord_error() (NaN constants: no grad leak)
    if not P.skip_fwd_finalize:
        c = _fwd_finalize(P, c)
    return c


def pj_inv4d(P: Operation, c: Coord, ctx: _ExecContext) -> Coord:
    """inv.cpp:223-262."""
    if not P.skip_inv_prepare:
        c = _inv_prepare(P, c)
    if not P.has_inv:
        return _err(torch.ones_like(c[0], dtype=torch.bool), c)
    c = P.inv4d(c, ctx)
    c = _err(torch.isnan(c[0]), c)
    if not P.skip_inv_finalize:
        c = _inv_finalize(P, c)
    return c


# =============================================================================
# Conversions
# =============================================================================
class NoOp(Operation):
    """conversions/noop.cpp."""

    def fwd4d(self, c, ctx):
        return c

    def inv4d(self, c, ctx):
        return c


class LatLong(Operation):
    """projections/latlong.cpp: identity, prepare/finalize do lon_0, over, lon_wrap, vunits."""

    def setup(self):
        self.is_latlong = True
        self.x0 = self.y0 = 0.0
        self.left = self.right = RADIANS

    def fwd4d(self, c, ctx):
        return c

    def inv4d(self, c, ctx):
        return c


class Geocent(LatLong):
    """conversions/geocent.cpp: the work is done by the cart helper in prepare/finalize."""

    def setup(self):
        self.is_geocent = True
        self.x0 = self.y0 = 0.0
        self.left, self.right = RADIANS, CARTESIAN


class Geoc(Operation):
    """conversions/geoc.cpp."""
    need_ellps = True

    def setup(self):
        self.is_latlong = True
        self.left = self.right = RADIANS

    def fwd4d(self, c, ctx):
        return (c[0], _geocentric_latitude(self, True, c[1]), c[2], c[3])

    def inv4d(self, c, ctx):
        return (c[0], _geocentric_latitude(self, False, c[1]), c[2], c[3])


# ---- unitconvert.cpp ---------------------------------------------------------
def _is_leap(y: Tensor) -> Tensor:  # unitconvert.cpp:97-100
    return ((torch.remainder(y, 4) == 0) & (torch.remainder(y, 100) != 0)) | (torch.remainder(y, 400) == 0)


def _days_in_year(y: Tensor) -> Tensor:
    return torch.where(_is_leap(y), 366.0, torch.full_like(y, 365.0))


def _n_leap(y: Tensor) -> Tensor:
    """number of leap years in [1, y] (proleptic Gregorian, floor arithmetic)."""
    return torch.floor(y / 4) - torch.floor(y / 100) + torch.floor(y / 400)


_N_LEAP_1858 = 1858 // 4 - 1858 // 100 + 1858 // 400


def _year_start_mjd(y: Tensor) -> Tensor:
    """mjd of Jan 1st of year y >= 1858 as accumulated by the loops in unitconvert.cpp:160-205."""
    return 45.0 + 365.0 * (y - 1859) + (_n_leap(y - 1) - _N_LEAP_1858)


def _decimalyear_to_mjd(dy: Tensor) -> Tensor:  # unitconvert.cpp:160-184
    ok = (dy >= -10000) & (dy <= 10000)
    d = torch.where(ok, dy, 0.0)
    year = torch.floor(d)
    mjd = (year - 1859) * 365 + 14 + 31
    mjd = mjd + (d - year) * _days_in_year(year)
    mjd = mjd + torch.clamp(_n_leap(year - 1) - _N_LEAP_1858, min=0)
    return torch.where(ok, mjd, 0.0)


def _mjd_to_decimalyear(mjd: Tensor) -> Tensor:  # unitconvert.cpp:187-205 (loop -> closed form)
    m = mjd.detach()
    y = torch.clamp(1859 + torch.floor((m - 45) / 365.2425), min=1858, max=1e7)
    for _ in range(2):
        y = torch.where((y > 1858) & (_year_start_mjd(y) > m), y - 1, y)
        y = torch.where(_year_start_mjd(y + 1) <= m, y + 1, y)
    return y + (mjd - _year_start_mjd(y)) / _days_in_year(y)


_TIME_UNITS: Dict[str, Tuple[Callable, Callable]] = {
    "mjd": (lambda t: t, lambda t: t),
    "decimalyear": (_decimalyear_to_mjd, _mjd_to_decimalyear),
    "gps_week": (lambda w: 44244.0 + w * 7.0, lambda m: (m - 44244.0) / 7.0),
}


class UnitConvert(Operation):
    """conversions/unitconvert.cpp:438-595."""

    def setup(self):
        pl = self.params
        self.skip_fwd_prepare = self.skip_inv_prepare = True
        self.xy_factor = self.z_factor = 1.0
        lin = {}
        for key in ("xy_in", "xy_out", "z_in", "z_out"):
            name = pl.s(key)
            if name is None:
                continue
            f, norm = 0.0, None
            for table, is_lin in ((LINEAR_UNITS, 1), (ANGULAR_UNITS, 0)):
                for uid, _, uname, factor in table:
                    if uid == name and f == 0.0:
                        f, norm, lin[key] = factor, uname, is_lin
            if f == 0.0:
                f = pj_atof(name)
                if f == 0.0 or 1.0 / f == 0.0:
                    raise ProjError("unknown %s unit" % key)
            if key == "xy_in":
                self.xy_factor = f
            elif key == "xy_out":
                self.xy_factor /= f
            elif key == "z_in":
                self.z_factor = f
            else:
                self.z_factor /= f
            if key in ("xy_in", "xy_out") and norm in ("Radian", "Degree"):
                u = RADIANS if norm == "Radian" else DEGREES
                if key == "xy_in":
                    self.left = u
                else:
                    self.right = u
        for a, b in (("xy_in", "xy_out"), ("z_in", "z_out")):
            if a in lin and b in lin and lin[a] != lin[b]:
                raise ProjError("inconsistent unit type between %s and %s" % (a, b))
        self.t_in = self.t_out = None
        for key in ("t_in", "t_out"):
            name = pl.s(key)
            if name is None:
                continue
            if name == "yyyymmdd":
                raise NotSupported("unitconvert yyyymmdd time unit not ported")
            if name not in _TIME_UNITS:
                raise ProjError("unknown %s unit" % key)
            setattr(self, key, name)

    def fwd4d(self, c, ctx):  # unitconvert.cpp:279-371
        x, y, z, t = c
        x, y, z = x * self.xy_factor, y * self.xy_factor, z * self.z_factor
        if self.t_in is not None:
            t = _TIME_UNITS[self.t_in][0](t)
        if self.t_out is not None:
            t = _TIME_UNITS[self.t_out][1](t)
        return (x, y, z, t)

    def inv4d(self, c, ctx):  # unitconvert.cpp:294-392
        x, y, z, t = c
        x, y, z = x / self.xy_factor, y / self.xy_factor, z / self.z_factor
        if self.t_out is not None:
            t = _TIME_UNITS[self.t_out][0](t)
        if self.t_in is not None:
            t = _TIME_UNITS[self.t_in][1](t)
        return (x, y, z, t)


class AxisSwap(Operation):
    """conversions/axisswap.cpp:158-309."""

    def setup(self):
        pl = self.params
        if pl.exists("order") == pl.exists("axis"):
            raise ProjError("must provide EITHER 'order' OR 'axis' parameter.", "invalid_op_mutually_exclusive_args")
        axis = [4, 5, 6, 7]
        sign = [1, 1, 1, 1]
        n = 0
        if pl.exists("order"):
            order = pl.s("order") or ""
            if any(ch not in "1234-," for ch in order):
                raise ProjError("unknown axis")
            for tok in order.split(",")[:4]:
                v = int(pj_atof(tok)) if tok not in ("", "-") else 0
                axis[n] = abs(v) - 1
                if axis[n] > 3 or axis[n] < 0:
                    raise ProjError("invalid axis")
                sign[n] = (v > 0) - (v < 0)
                n += 1
        if pl.exists("axis"):
            m = {"w": (-1, 0), "e": (1, 0), "s": (-1, 1), "n": (1, 1), "d": (-1, 2), "u": (1, 2)}
            for i in range(3):
                sign[i], axis[i] = m[self.axis[i]]
            n = 3
        for i in range(4):
            for j in range(4):
                if i != j and axis[i] == axis[j]:
                    raise ProjError("axisswap: duplicate axes specified")
        self.mode = None
        if n == 4:
            self.mode = "4d"
        if n == 3 and axis[0] < 3 and axis[1] < 3 and axis[2] < 3:
            self.mode = "3d"
        if n == 2:
            if axis[0] == 1 and sign[0] == 1 and axis[1] == 0 and sign[1] == 1:
                self.mode = "swap"
            elif axis[0] < 2 and axis[1] < 2:
                self.mode = "2d"
        if self.mode is None:
            raise ProjError("axisswap: bad axis order")
        self.axis_idx, self.sign = axis, sign
        self.left = self.right = RADIANS if pl.exists("angularunits") else WHATEVER
        self.skip_fwd_prepare = self.skip_fwd_finalize = True
        self.skip_inv_prepare = self.skip_inv_finalize = True

    def _n(self) -> int:
        return {"4d": 4, "3d": 3, "2d": 2}[self.mode]

    def fwd4d(self, c, ctx):  # axisswap.cpp:75-144
        if self.mode == "swap":
            return (c[1], c[0], c[2], c[3])
        out = list(c)
        for i in range(self._n()):
            v = c[self.axis_idx[i]]
            out[i] = v if self.sign[i] > 0 else -v
        return tuple(out)

    def inv4d(self, c, ctx):  # axisswap.cpp:85-155
        if self.mode == "swap":
            return (c[1], c[0], c[2], c[3])
        out = list(c)
        for i in range(self._n()):
            out[self.axis_idx[i]] = c[i] if self.sign[i] > 0 else -c[i]
        return tuple(out)


class Cart(Operation):
    """conversions/cart.cpp: geodetic <-> geocentric cartesian.
    Inverse = Bowring (1976) single step with Fukushima-style normalisation and a
    polar branch (cart.cpp:156-239), i.e. NOT iterative."""
    need_ellps = True

    def setup(self):
        self.left, self.right = RADIANS, CARTESIAN
        self.b_div_a = 1 - self.f

    def _N(self, sinphi: Tensor) -> Tensor:  # cart.cpp:106-112
        if self.es == 0:
            return torch.full_like(sinphi, self.a)
        return self.a / torch.sqrt(1 - self.es * sinphi * sinphi)

    def fwd4d(self, c, ctx):  # cart.cpp:139-153
        lam, phi, h, t = c
        cosphi, sinphi = torch.cos(phi), torch.sin(phi)
        N = self._N(sinphi)
        x = (N + h) * cosphi * torch.cos(lam)
        y = (N + h) * cosphi * torch.sin(lam)
        z = (N * (1 - self.es) + h) * sinphi
        return (x, y, z, t)

    def inv4d(self, c, ctx):  # cart.cpp:156-239
        X, Y, Z, t = c
        ra, b_div_a = self.ell.ra, self.b_div_a
        xa, ya, za = X * ra, Y * ra, Z * ra
        p2 = xa * xa + ya * ya
        p = _sqrt0(p2)
        pb = p * b_div_a
        n2 = za * za + pb * pb
        nz = n2 != 0
        inv_norm = 1.0 / torch.sqrt(torch.where(nz, n2, 1.0))
        cc = torch.where(nz, pb * inv_norm, 1.0)
        s = torch.where(nz, za * inv_norm, 0.0)
        y_phi = za + self.ell.e2s * b_div_a * s * s * s
        x_phi = p - self.es * cc * cc * cc
        nphi2 = y_phi * y_phi + x_phi * x_phi
        nzp = nphi2 != 0
        inv_np = 1.0 / torch.sqrt(torch.where(nzp, nphi2, 1.0))
        cosphi = torch.where(nzp, x_phi * inv_np, 1.0)
        sinphi = torch.where(nzp, y_phi * inv_np, 0.0)
        neg = x_phi <= 0
        zpos = Z >= 0
        phi = torch.where(neg, torch.where(zpos, M_HALFPI, torch.full_like(Z, -M_HALFPI)),
                          torch.atan(y_phi / torch.where(neg, 1.0, x_phi)))
        cosphi = torch.where(neg, 0.0, cosphi)
        sinphi = torch.where(neg, torch.where(zpos, 1.0, torch.full_like(Z, -1.0)), sinphi)
        pp = p2 > 0  # atan2(0, 0) has a NaN gradient: keep it out of the graph
        lam = torch.where(pp, torch.atan2(torch.where(pp, ya, 0.0), torch.where(pp, xa, 1.0)),
                          torch.atan2(ya.detach(), xa.detach()))
        polar = cosphi < 1e-6
        # geocentric_radius (cart.cpp:115-136)
        c2, s2 = cosphi * cosphi, sinphi * sinphi
        bb = b_div_a * b_div_a
        bbs = bb * s2
        r = self.a * torch.sqrt((c2 + bb * bbs) / (c2 + bbs))
        h_polar = Z.abs() - r
        h_norm = self.a * p / torch.where(polar, 1.0, cosphi) - self._N(sinphi)
        return (lam, phi, torch.where(polar, h_polar, h_norm), t)


class Push(Operation):
    """pipeline.cpp:641-718: +proj=push (+v_1 +v_2 +v_3 +v_4)."""
    _is_push = True

    def setup(self):
        self.v = [self.params.exists("v_%d" % (i + 1)) for i in range(4)]

    def _push(self, c, ctx):
        if ctx is None:
            return c
        for i in range(4):
            if self.v[i]:
                ctx.stack[i].append(c[i])
        return c

    def _pop(self, c, ctx):
        if ctx is None:
            return c
        out = list(c)
        for i in range(4):
            if self.v[i] and ctx.stack[i]:
                out[i] = ctx.stack[i].pop()
        return tuple(out)

    def fwd4d(self, c, ctx):
        return self._push(c, ctx) if self._is_push else self._pop(c, ctx)

    def inv4d(self, c, ctx):
        return self._pop(c, ctx) if self._is_push else self._push(c, ctx)


class Pop(Push):
    _is_push = False


class SetOp(Operation):
    """conversions/set.cpp."""

    def setup(self):
        self.vals = [(i, self.params.d("v_%d" % (i + 1))) for i in range(4)
                     if self.params.exists("v_%d" % (i + 1))]

    def fwd4d(self, c, ctx):
        out = list(c)
        for i, v in self.vals:
            out[i] = torch.full_like(c[i], v)
        return tuple(out)

    inv4d = fwd4d


# =============================================================================
# Transformations
# =============================================================================
class Affine(Operation):
    """transformations/affine.cpp:59-223 (also geogoffset, :228-250)."""

    def setup(self):
        pl = self.params
        self.off = [pl.d("xoff"), pl.d("yoff"), pl.d("zoff"), pl.d("toff")]
        S = {k: pl.d(k) for k in ("s12", "s13", "s21", "s23", "s31", "s32")}
        for k in ("s11", "s22", "s33", "tscale"):
            S[k] = pl.d(k) if pl.exists(k) else 1.0
        self.S = S
        a, b, c_, d, e, f, g, h, i = (S[k] for k in ("s11", "s12", "s13", "s21", "s22", "s23", "s31", "s32", "s33"))
        A, B, C = e * i - f * h, -(d * i - f * g), (d * h - e * g)
        D, E, F = -(b * i - c_ * h), (a * i - c_ * g), -(a * h - b * g)
        G, H, I = b * f - c_ * e, -(a * f - c_ * d), a * e - b * d
        det = a * A + b * B + c_ * C
        if det == 0.0 or S["tscale"] == 0.0:
            self.has_inv = False
            self.R = None
        else:
            self.R = [[A / det, D / det, G / det], [B / det, E / det, H / det],
                      [C / det, F / det, I / det], 1.0 / S["tscale"]]

    def fwd4d(self, c, ctx):
        x, y, z, t = c
        S, o = self.S, self.off
        return (o[0] + S["s11"] * x + S["s12"] * y + S["s13"] * z,
                o[1] + S["s21"] * x + S["s22"] * y + S["s23"] * z,
                o[2] + S["s31"] * x + S["s32"] * y + S["s33"] * z,
                o[3] + S["tscale"] * t)

    def inv4d(self, c, ctx):
        o, R = self.off, self.R
        x, y, z = c[0] - o[0], c[1] - o[1], c[2] - o[2]
        return (R[0][0] * x + R[0][1] * y + R[0][2] * z,
                R[1][0] * x + R[1][1] * y + R[1][2] * z,
                R[2][0] * x + R[2][1] * y + R[2][2] * z,
                R[3] * (c[3] - o[3]))


class GeogOffset(Affine):
    def setup(self):
        pl = self.params
        self.off = [pl.d("dlon") * ARCSEC_TO_RAD, pl.d("dlat") * ARCSEC_TO_RAD, pl.d("dh"), 0.0]
        self.S = {"s11": 1.0, "s12": 0.0, "s13": 0.0, "s21": 0.0, "s22": 1.0, "s23": 0.0,
                  "s31": 0.0, "s32": 0.0, "s33": 1.0, "tscale": 1.0}
        self.R = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], 1.0]
        self.left = self.right = RADIANS


def _rot_matrix(o, p, k, exact: bool, position_vector: bool):
    """helmert.cpp:161-321 build_rot_matrix; works on floats or tensors."""
    if exact:
        cos, sin = (torch.cos, torch.sin) if isinstance(o, Tensor) else (math.cos, math.sin)
        cf, sf, ct, st, cp, sp = cos(o), sin(o), cos(p), sin(p), cos(k), sin(k)
        R = [[ct * cp, cf * sp + sf * st * cp, sf * sp - cf * st * cp],
             [-ct * sp, cf * cp - sf * st * sp, sf * cp + cf * st * sp],
             [st, -sf * ct, cf * ct]]
    else:
        one = 1.0 if not isinstance(o, Tensor) else torch.ones_like(o)
        R = [[one, k, -p], [-k, one, o], [p, -o, one]]
    if position_vector:
        R = [[R[0][0], R[1][0], R[2][0]], [R[0][1], R[1][1], R[2][1]], [R[0][2], R[1][2], R[2][2]]]
    return R


class Helmert(Operation):
    """transformations/helmert.cpp:556-696 (3/7/15 parameter + 2D 4-parameter)."""
    molobadekas = False

    def setup(self):
        pl = self.params
        self.left = self.right = CARTESIAN
        g = lambda k: pl.d(k) if pl.exists(k) else 0.0  # noqa: E731
        self.xyz_0 = [g("x"), g("y"), g("z")]
        self.opk_0 = [g("rx") * ARCSEC_TO_RAD, g("ry") * ARCSEC_TO_RAD, g("rz") * ARCSEC_TO_RAD]
        self.exact = pl.b("exact")
        self.refp = [0.0, 0.0, 0.0]
        self.scale_0 = self.theta_0 = 0.0
        self.dxyz, self.dopk, self.dscale, self.dtheta, self.t_epoch = [0.0] * 3, [0.0] * 3, 0.0, 0.0, 0.0
        self.fourparam = False
        if self.molobadekas:  # helmert.cpp:699-760
            self.scale_0 = g("s")
            self.no_rotation = False
            self._read_convention()
            self.refp = [g("px"), g("py"), g("pz")]
            self.xyz_0 = [self.xyz_0[i] + self.refp[i] for i in range(3)]
        else:
            if pl.exists("theta"):
                self.left = self.right = PROJECTED
            if pl.exists("transpose"):
                raise ProjError("helmert: 'transpose' argument is no longer valid.")
            if pl.exists("towgs84"):
                dp = self.datum_params
                self.xyz_0 = list(dp[:3])
                self.opk_0 = list(dp[3:6])
                self.scale_0 = 0.0 if dp[6] == 0 else (dp[6] - 1) * 1e6
            if pl.exists("theta"):
                self.theta_0 = pl.d("theta") * ARCSEC_TO_RAD
                self.fourparam = True
                self.scale_0 = 1.0
            if pl.exists("s"):
                self.scale_0 = pl.d("s")
                if self.scale_0 <= -1.0e6 or (pl.exists("theta") and self.scale_0 == 0.0):
                    raise ProjError("helmert: invalid value for s.")
            self.dxyz = [g("dx"), g("dy"), g("dz")]
            self.dopk = [g("drx") * ARCSEC_TO_RAD, g("dry") * ARCSEC_TO_RAD, g("drz") * ARCSEC_TO_RAD]
            self.dtheta = g("dtheta") * ARCSEC_TO_RAD
            self.dscale = g("ds")
            self.t_epoch = g("t_epoch")
            self.no_rotation = all(v == 0 for v in self.opk_0 + self.dopk)
            self._read_convention()
        self.has_rates = any(v != 0 for v in self.dxyz + self.dopk + [self.dscale, self.dtheta])
        # parameters at t_obs == t_epoch (the only ones needed when there are no rates)
        self.R = _rot_matrix(*self.opk_0, self.exact, self.is_position_vector)

    def _read_convention(self):  # helmert.cpp:517-553
        self.is_position_vector = False
        if not self.no_rotation:
            conv = self.params.s("convention")
            if conv is None:
                raise ProjError("helmert: missing 'convention' argument", "invalid_op_missing_arg")
            if conv not in ("position_vector", "coordinate_frame"):
                raise ProjError("helmert: invalid value for 'convention' argument")
            self.is_position_vector = conv == "position_vector"
            if self.params.exists("towgs84") and not self.is_position_vector:
                raise ProjError("helmert: towgs84 should only be used with convention=position_vector")

    def _params_at(self, t: Tensor):
        """update_parameters + build_rot_matrix (helmert.cpp:102-158, 438-474), per point."""
        if not self.has_rates:
            return self.xyz_0, self.R, self.scale_0, self.theta_0
        dt = torch.where(t == INF, 0.0, t - self.t_epoch)  # t_obs = t_epoch when t is HUGE_VAL
        xyz = [self.xyz_0[i] + self.dxyz[i] * dt for i in range(3)]
        opk = [self.opk_0[i] + self.dopk[i] * dt for i in range(3)]
        R = _rot_matrix(*opk, self.exact, self.is_position_vector)
        return xyz, R, self.scale_0 + self.dscale * dt, self.theta_0 + self.dtheta * dt

    def fwd4d(self, c, ctx):  # helmert.cpp:324-399
        X, Y, Z, t = c
        xyz, R, s, th = self._params_at(t) if not self.molobadekas else (self.xyz_0, self.R, self.scale_0, 0.0)
        if self.fourparam:
            cos, sin = (torch.cos, torch.sin) if isinstance(th, Tensor) else (math.cos, math.sin)
            cr, sr = cos(th) * s, sin(th) * s
            x0 = self.xyz_0
            return (cr * X + sr * Y + x0[0], -sr * X + cr * Y + x0[1], Z, t)
        if self.no_rotation and not self.has_rates and s == 0:
            return (X + xyz[0], Y + xyz[1], Z + xyz[2], t)
        scale = 1 + s * 1e-6
        rp = self.refp
        if rp != [0.0, 0.0, 0.0]:
            X, Y, Z = X - rp[0], Y - rp[1], Z - rp[2]
        x = scale * (R[0][0] * X + R[0][1] * Y + R[0][2] * Z) + xyz[0]
        y = scale * (R[1][0] * X + R[1][1] * Y + R[1][2] * Z) + xyz[1]
        z = scale * (R[2][0] * X + R[2][1] * Y + R[2][2] * Z) + xyz[2]
        return (x, y, z, t)

    def inv4d(self, c, ctx):  # helmert.cpp:343-436
        X, Y, Z, t = c
        xyz, R, s, th = self._params_at(t) if not self.molobadekas else (self.xyz_0, self.R, self.scale_0, 0.0)
        if self.fourparam:
            cos, sin = (torch.cos, torch.sin) if isinstance(th, Tensor) else (math.cos, math.sin)
            cr, sr = cos(th) / s, sin(th) / s
            x, y = X - self.xyz_0[0], Y - self.xyz_0[1]
            return (x * cr - y * sr, x * sr + y * cr, Z, t)
        if self.no_rotation and not self.has_rates and s == 0:
            return (X - xyz[0], Y - xyz[1], Z - xyz[2], t)
        scale = 1 + s * 1e-6
        x, y, z = (X - xyz[0]) / scale, (Y - xyz[1]) / scale, (Z - xyz[2]) / scale
        rp = self.refp
        return ((R[0][0] * x + R[1][0] * y + R[2][0] * z) + rp[0],
                (R[0][1] * x + R[1][1] * y + R[2][1] * z) + rp[1],
                (R[0][2] * x + R[1][2] * y + R[2][2] * z) + rp[2], t)


class MoloBadekas(Helmert):
    molobadekas = True


class Molodensky(Operation):
    """transformations/molodensky.cpp (standard + abridged; the inverse evaluates the
    shifts at the *output* point and subtracts, exactly as PROJ does)."""
    need_ellps = True

    def setup(self):
        pl = self.params
        self.left = self.right = RADIANS
        for k in ("dx", "dy", "dz", "da", "df"):
            if not pl.exists(k):
                raise ProjError("missing " + k, "invalid_op_missing_arg")
            setattr(self, k, pl.d(k))
        self.abridged = pl.exists("abridged")

    def _RN(self, sphi):  # molodensky.cpp:68-80
        if self.es == 0:
            return torch.full_like(sphi, self.a)
        return self.a / torch.sqrt(1 - self.es * sphi * sphi)

    def _RM(self, phi, sphi):  # molodensky.cpp:83-108
        a, es = self.a, self.es
        if es == 0:
            return torch.full_like(phi, a)
        gen = (a * (1 - es)) / torch.pow(1 - es * sphi * sphi, 1.5)
        return torch.where(phi == 0, a * (1 - es), torch.where(phi.abs() == M_HALFPI, a / math.sqrt(1 - es), gen))

    def _shifts(self, lam, phi, h):
        slam, clam, sphi, cphi = torch.sin(lam), torch.cos(lam), torch.sin(phi), torch.cos(phi)
        dx, dy, dz, da, df, a, f, es = self.dx, self.dy, self.dz, self.da, self.df, self.a, self.f, self.es
        if self.abridged:  # molodensky.cpp:162-200
            adffda = a * df + f * da
            dphi = -dx * sphi * clam - dy * sphi * slam + dz * cphi + adffda * torch.sin(2 * phi)
            dphi = dphi / self._RM(phi, sphi)
            dlam, bad = _nonzero_div(-dx * slam + dy * clam, self._RN(sphi) * cphi)
            dh = dx * cphi * clam + dy * cphi * slam + dz * sphi - da + adffda * sphi * sphi
            return dlam, dphi, dh, bad
        rho, nu = self._RM(phi, sphi), self._RN(sphi)  # molodensky.cpp:111-159
        dphi = ((-dx * sphi * clam) - (dy * sphi * slam) + (dz * cphi) + ((nu * es * sphi * cphi * da) / a)
                + (sphi * cphi * (rho / (1 - f) + nu * (1 - f)) * df))
        dphi, bad1 = _nonzero_div(dphi, rho + h)
        dlam, bad2 = _nonzero_div(-dx * slam + dy * clam, (nu + h) * cphi)
        dh = dx * cphi * clam + dy * cphi * slam + dz * sphi - (a / nu) * da + nu * (1 - f) * sphi * sphi * df
        return dlam, dphi, dh, bad1 | bad2

    def fwd4d(self, c, ctx):
        lam, phi, h, t = c
        dl, dp, dh, bad = self._shifts(lam, phi, h)
        return _err(bad, (lam + dl, phi + dp, h + dh, t))

    def inv4d(self, c, ctx):
        lam, phi, h, t = c
        dl, dp, dh, bad = self._shifts(lam, phi, h)
        return _err(bad, (lam - dl, phi - dp, h - dh, t))


# ---- hgridshift + grids.cpp --------------------------------------------------
GRID_SEARCH_PATHS: List[str] = []  # extra directories searched for grid files


def _grid_dirs() -> List[Path]:
    dirs = [Path(p) for p in GRID_SEARCH_PATHS]
    for env in ("PROJ_DATA", "PROJ_LIB"):
        if os.environ.get(env):
            dirs += [Path(p) for p in os.environ[env].split(os.pathsep)]
    dirs += [Path(f"{_PROJ_ROOT}/data"), Path(f"{_PROJ_ROOT}/data/tests")]
    return dirs


def _find_grid_file(name: str) -> Optional[Path]:
    p = Path(name)
    if p.is_absolute():
        return p if p.exists() else None
    for d in _grid_dirs():
        if (d / name).exists():
            return d / name
    return None


class _GridNode:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def read_ntv2(path: Path) -> List[_GridNode]:
    """NTv2GridSet::open + NTv2Grid::valueAt (grids.cpp:2123-2344), numpy only.
    Returns the nodes in file order (with .parent index or -1)."""
    raw = Path(path).read_bytes()
    if len(raw) < 176 or raw[0:8] != b"NUM_OREC" or raw[48:55] != b"GS_TYPE":
        raise NotSupported("%s is not an NTv2 file (only NTv2 is ported)" % path)
    if raw[56:63] != b"SECONDS":
        raise ProjError("Only GS_TYPE=SECONDS is supported", "invalid_op_file_not_found_or_invalid")
    end = "<" if raw[8] == 11 else ">"
    nsub = struct.unpack(end + "I", raw[40:44])[0]
    off, nodes, by_name = 176, [], {}
    conv = (math.pi / 180.0) / 3600.0
    for _ in range(nsub):
        h = raw[off:off + 176]
        off += 176
        if h[0:8] != b"SUB_NAME":
            raise ProjError("bad NTv2 subfile header", "invalid_op_file_not_found_or_invalid")
        v = [struct.unpack(end + "d", h[72 + 16 * i:80 + 16 * i])[0] for i in range(6)]
        south, north = v[0] * DEG_TO_RAD / 3600.0, v[1] * DEG_TO_RAD / 3600.0
        east, west = -v[2] * DEG_TO_RAD / 3600.0, -v[3] * DEG_TO_RAD / 3600.0
        resY, resX = v[4] * DEG_TO_RAD / 3600.0, v[5] * DEG_TO_RAD / 3600.0
        if not (abs(west) <= 4 * M_PI and abs(east) <= 4 * M_PI and abs(north) <= M_PI + 1e-5
                and abs(south) <= M_PI + 1e-5 and west < east and south < north
                and resX > 1e-10 and resY > 1e-10):
            raise ProjError("Inconsistent georeferencing", "invalid_op_file_not_found_or_invalid")
        cols = int(abs((east - west) * (1.0 / resX) + 0.5) + 1)
        rows = int(abs((north - south) * (1.0 / resY) + 0.5) + 1)
        gs_count = struct.unpack(end + "I", h[168:172])[0]
        if gs_count // cols != rows:
            raise ProjError("GS_COUNT does not match expected cells", "invalid_op_file_not_found_or_invalid")
        data = np.frombuffer(raw, dtype=end + "f4", count=gs_count * 4, offset=off).reshape(rows, cols, 4)
        off += gs_count * 16
        data = data[:, ::-1, :]  # NTv2 rows are stored east to west
        # valueAt: float(float(v) * conv) in radians, longitude positive east
        lat = (data[..., 0].astype(np.float64) * conv).astype(np.float32).astype(np.float64)
        lon = -((data[..., 1].astype(np.float64) * conv).astype(np.float32).astype(np.float64))
        name, parent = h[8:16], h[24:32]
        node = _GridNode(name=name, west=west, east=east, south=south, north=north, resX=resX, resY=resY,
                         width=cols, height=rows, values=np.stack([lon, lat], -1).reshape(-1, 2),
                         parent=by_name.get(parent, -1), is_null=False, idx=len(nodes))
        by_name[name] = node.idx
        nodes.append(node)
    return nodes


class HGridShift(Operation):
    """transformations/hgridshift.cpp + grids.cpp:3252-3540 (pj_hgrid_apply):
    vectorised subgrid search, bilinear interpolation and the iterative inverse."""
    MAX_ITERATIONS = 10          # grids.cpp:3399
    TOL = 1e-12                  # grids.cpp:3400
    REL_TOLERANCE = 1e-5         # grids.cpp:2760 REL_TOLERANCE_HGRIDSHIFT

    def setup(self):
        pl = self.params
        self.left = self.right = RADIANS
        grids = pl.s("grids")
        if grids is None:
            raise ProjError("+grids parameter missing.", "invalid_op_missing_arg")
        tf = pl.s("t_final")
        self.t_final = 0.0
        if tf is not None:  # param.cpp:221 pj_parse_t_final
            if tf == "now":
                import datetime
                now = datetime.datetime.now(datetime.timezone.utc)
                self.t_final = now.year + (now.timetuple().tm_yday - 1) / 365.0
            else:
                self.t_final = pj_atof(tf)
        self.t_epoch = pl.d("t_epoch") if pl.exists("t_epoch") else 0.0
        nodes: List[_GridNode] = []
        for gname in grids.split(","):  # grids.cpp:3267-3292
            can_fail = gname.startswith("@")
            gname = gname[1:] if can_fail else gname
            if gname == "null":
                nodes.append(_GridNode(name=b"null", west=-M_PI, east=M_PI, south=-M_HALFPI, north=M_HALFPI,
                                       resX=M_PI, resY=M_HALFPI, width=3, height=3,
                                       values=np.zeros((9, 2)), parent=-1, is_null=True, idx=0))
                continue
            path = _find_grid_file(gname)
            if path is None:
                if can_fail:
                    continue
                raise ProjError("could not find required grid(s).", "invalid_op_file_not_found_or_invalid")
            base = len(nodes)
            for nd in read_ntv2(path):
                nd.parent = nd.parent + base if nd.parent >= 0 else -1
                nodes.append(nd)
        for i, nd in enumerate(nodes):
            nd.idx = i
            nd.eps = (nd.resX + nd.resY) * self.REL_TOLERANCE
            nd.full_world = nd.east - nd.west + nd.resX >= 2 * M_PI - 1e-10  # grids.cpp:91-94
            nd.depth = 0 if nd.parent < 0 else nodes[nd.parent].depth + 1
            nd.offset = sum(n.width * n.height for n in nodes[:i])
        self.nodes = nodes
        self.levels: List[List[_GridNode]] = []
        for nd in nodes:
            while len(self.levels) <= nd.depth:
                self.levels.append([])
            self.levels[nd.depth].append(nd)
        if not nodes:
            self.register_buffer("values", torch.zeros(0, 2, dtype=torch.float64))
            return
        self.register_buffer("values", torch.from_numpy(np.concatenate([n.values for n in nodes])).to(torch.float64))
        f64 = lambda k: torch.tensor([getattr(n, k) for n in nodes], dtype=torch.float64)  # noqa: E731
        i64 = lambda k: torch.tensor([getattr(n, k) for n in nodes], dtype=torch.int64)  # noqa: E731
        for k in ("west", "south", "resX", "resY", "eps"):
            self.register_buffer("n_" + k, f64(k))
        self.register_buffer("n_ew", torch.tensor([n.east - n.west for n in nodes], dtype=torch.float64))
        for k in ("width", "height", "offset"):
            self.register_buffer("n_" + k, i64(k))
        self.register_buffer("n_null", torch.tensor([n.is_null for n in nodes], dtype=torch.bool))

    # -- grids.cpp:1690-1705 isPointInExtent, 2762-2792 gridAt, 3252-3263 findGrid
    @staticmethod
    def _contains(nd: _GridNode, x: Tensor, y: Tensor) -> Tensor:
        if nd.is_null:
            return torch.ones_like(x, dtype=torch.bool)
        eps = nd.eps
        ok = (y + eps >= nd.south) & (y - eps <= nd.north)
        if nd.full_world:
            return ok
        x = torch.where(x + eps < nd.west, x + 2 * M_PI, torch.where(x - eps > nd.east, x - 2 * M_PI, x))
        return ok & (x + eps >= nd.west) & (x - eps <= nd.east)

    def _find_grid(self, lam: Tensor, phi: Tensor) -> Tensor:
        sel = torch.full_like(lam, -1, dtype=torch.int64)
        for nd in self.levels[0] if self.levels else []:
            sel = torch.where((sel < 0) & self._contains(nd, lam, phi), nd.idx, sel)
        for level in self.levels[1:]:
            claimed = torch.zeros_like(lam, dtype=torch.bool)
            new = sel
            for nd in level:
                m = (sel == nd.parent) & ~claimed & self._contains(nd, lam, phi)
                new = torch.where(m, nd.idx, new)
                claimed = claimed | m
            sel = new
        return sel

    def _normalize(self, lam, phi, g):  # grids.cpp:3415-3422
        dt = lam.dtype
        tl = lam - self.n_west[g].to(dt)
        eps, ew = self.n_eps[g].to(dt), self.n_ew[g].to(dt)
        tl = torch.where(tl + eps < 0, tl + 2 * M_PI, torch.where(tl - eps > ew, tl - 2 * M_PI, tl))
        return tl, phi - self.n_south[g].to(dt)

    def _interpolate(self, tl: Tensor, tp: Tensor, g: Tensor):
        """grids.cpp:3326-3396 pj_hgrid_interpolate -> (dlam, dphi, invalid). Invalid
        (HUGE_VAL in PROJ) points get a finite 0 so that no NaN enters the graph."""
        dt = tl.dtype
        W, H, off = self.n_width[g], self.n_height[g], self.n_offset[g]
        ul, up = tl / self.n_resX[g].to(dt), tp / self.n_resY[g].to(dt)
        il = torch.floor(torch.where(torch.isnan(ul), 0.0, ul))
        ip = torch.floor(torch.where(torch.isnan(up), 0.0, up))
        fl, fp = ul - il, up - ip
        Wf, Hf = W.to(dt), H.to(dt)
        bad = torch.zeros_like(ul, dtype=torch.bool)
        res = []
        for i, f, N in ((il, fl, Wf), (ip, fp, Hf)):
            lo = i < 0
            lo_fix = lo & (i == -1) & (f > 1 - 10 * self.REL_TOLERANCE)
            hi = ~lo & (i + 1 >= N)
            hi_fix = hi & (i + 1 == N) & (f < 10 * self.REL_TOLERANCE)
            bad = bad | (lo & ~lo_fix) | (hi & ~hi_fix)
            i = torch.where(lo_fix, i + 1, torch.where(hi_fix, i - 1, i))
            f = torch.where(lo_fix, 0.0, torch.where(hi_fix, 1.0, f))
            res.append((i, f))
        (il, fl), (ip, fp) = res
        ix = torch.minimum(torch.clamp(il, min=0.0), Wf - 2).to(torch.int64)
        iy = torch.minimum(torch.clamp(ip, min=0.0), Hf - 2).to(torch.int64)
        base = off + iy * W + ix
        vals = self.values.to(dt)
        f00, f10, f01, f11 = vals[base], vals[base + 1], vals[base + W], vals[base + W + 1]
        m10 = fl
        m11 = m10
        m01 = 1. - fl
        m00 = m01
        m11 = m11 * fp
        m01 = m01 * fp
        fp = 1. - fp
        m00 = m00 * fp
        m10 = m10 * fp
        vl = m00 * f00[..., 0] + m10 * f10[..., 0] + m01 * f01[..., 0] + m11 * f11[..., 0]
        vp = m00 * f00[..., 1] + m10 * f10[..., 1] + m01 * f01[..., 1] + m11 * f11[..., 1]
        return torch.where(bad, 0.0, vl), torch.where(bad, 0.0, vp), bad

    def _apply(self, lam: Tensor, phi: Tensor, fwd: bool):
        """grids.cpp:3402-3540 pj_hgrid_apply(_internal)."""
        if not self.nodes:  # no grid loaded: pass through (hgridshift.cpp:42-46)
            return lam, phi
        g = self._find_grid(lam, phi)
        none = g < 0
        g = torch.clamp(g, min=0)
        is_null = self.n_null[g] & ~none
        tbl, tbp = self._normalize(lam, phi, g)
        t0l, t0p, bad0 = self._interpolate(tbl, tbp, g)
        fail = none | bad0
        if fwd:
            ol, op = lam + t0l, phi + t0p
        else:
            tl, tp = tbl - t0l, tbp - t0p
            done = fail | is_null
            toltol = self.TOL * self.TOL
            for k in range(1, self.MAX_ITERATIONS + 1):
                active = ~done
                dl, dp, dinv = self._interpolate(tl, tp, g)
                # grid switch when the iterate leaves the current grid (grids.cpp:3455-3480)
                west, south = self.n_west[g].to(lam.dtype), self.n_south[g].to(lam.dtype)
                lpl, lpp = tl + west, tp + south
                ng = self._find_grid(lpl, lpp)
                ngc = torch.clamp(ng, min=0)
                cant = (ng < 0) | (ng == g) | self.n_null[ngc]
                brk = active & dinv & cant
                swt = active & dinv & ~cant
                nrm = active & ~dinv
                difl, difp = tl + dl - tbl, tp + dp - tbp
                ntbl, ntbp = self._normalize(lam, phi, ngc)
                tl = torch.where(nrm, tl - difl, torch.where(swt, lpl - self.n_west[ngc].to(lam.dtype), tl))
                tp = torch.where(nrm, tp - difp, torch.where(swt, lpp - self.n_south[ngc].to(lam.dtype), tp))
                tbl, tbp = torch.where(swt, ntbl, tbl), torch.where(swt, ntbp, tbp)
                g = torch.where(swt, ngc, g)
                if k == self.MAX_ITERATIONS:  # --i == 0: not converged
                    fail = fail | (active & ~brk)
                else:
                    done = done | brk | (nrm & ~(difl * difl + difp * difp > toltol))
            ol = adjlon(tl + self.n_west[g].to(lam.dtype))
            op = tp + self.n_south[g].to(lam.dtype)
        ol = torch.where(is_null, lam, ol)
        op = torch.where(is_null, phi, op)
        return torch.where(fail, NAN, ol), torch.where(fail, NAN, op)

    def _shift4d(self, c, fwd):  # hgridshift.cpp:72-124
        lam, phi, z, t = c
        ol, op = self._apply(lam, phi, fwd)
        if self.t_final != 0 and self.t_epoch != 0:
            use = (t < self.t_epoch) & (self.t_final > self.t_epoch)
            ol, op = torch.where(use, ol, lam), torch.where(use, op, phi)
        return (ol, op, z, t)

    def fwd4d(self, c, ctx):
        return self._shift4d(c, True)

    def inv4d(self, c, ctx):
        return self._shift4d(c, False)


# =============================================================================
# Projections from other modules of the spike
# =============================================================================
EXTERNAL_PROJECTIONS: Dict[str, Tuple[str, str]] = {
    # proj name -> (module, factory taking a core.parse_proj_string-style dict)
    "tmerc": ("tmerc", "make"), "etmerc": ("tmerc", "make"), "utm": ("tmerc", "make"),
    "lcc": ("lcc_stere", "make"), "stere": ("lcc_stere", "make"), "ups": ("lcc_stere", "make"),
    "laea": ("laea_healpix_moll", "make"), "healpix": ("laea_healpix_moll", "make"),
    "rhealpix": ("laea_healpix_moll", "make"), "moll": ("laea_healpix_moll", "make"),
    "wag4": ("laea_healpix_moll", "make"), "wag5": ("laea_healpix_moll", "make"),
    "geos": ("geos_obtran", "make"), "eqc": ("geos_obtran", "make"), "ob_tran": ("geos_obtran", "make"),
}


def _load_external(name: str) -> Callable:
    modname, attr = EXTERNAL_PROJECTIONS[name]
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        mod = importlib.import_module(modname)
        return getattr(mod, attr)
    except Exception as exc:  # module missing or broken: treat as not ported
        raise NotSupported("projection %r: module %s.py unavailable (%s)" % (name, modname, exc))


class ProjectionStep(Operation):
    """A classic projection implemented by another module (core.Projection API:
    forward/inverse on [..., D] radians <-> metres, doing fwd.cpp/inv.cpp itself).
    The cs2cs datum shift helpers (+towgs84/+nadgrids) are applied here around it."""

    factory: Optional[Callable] = None

    def setup(self):
        self.left, self.right = RADIANS, CLASSIC
        self.skip_fwd_prepare = self.skip_fwd_finalize = True
        self.skip_inv_prepare = self.skip_inv_finalize = True
        try:
            self.proj = self.factory(self.params.as_dict())
        except NotImplementedError as exc:
            raise NotSupported(str(exc))
        except ValueError as exc:  # setup errors of the other modules (e.g. ProjSetupError)
            raise ProjError(str(exc))
        except Exception as exc:
            raise NotSupported("%s: %r" % (self.name, exc))
        self.has_inv = hasattr(self.proj, "inverse")

    def _datum_in(self, c):  # fwd.cpp:94-109
        if self.h_hgridshift is not None:
            return proj_trans(self.h_hgridshift, False, c)
        if self.h_helmert is not None or (self.h_cart_wgs84 is not None and self.h_cart is not None):
            c = proj_trans(self.h_cart_wgs84, True, c)
            if self.h_helmert is not None:
                c = proj_trans(self.h_helmert, False, c)
            c = proj_trans(self.h_cart, False, c)
        return c

    def _datum_out(self, c):  # inv.cpp:124-134
        if self.h_hgridshift is not None:
            return proj_trans(self.h_hgridshift, True, c)
        if self.h_helmert is not None or (self.h_cart_wgs84 is not None and self.h_cart is not None):
            c = proj_trans(self.h_cart, True, c)
            if self.h_helmert is not None:
                c = proj_trans(self.h_helmert, True, c)
            c = proj_trans(self.h_cart_wgs84, False, c)
        return c

    def fwd4d(self, c, ctx):
        c = self._datum_in(c)
        out = self.proj.forward(torch.stack(c, -1))
        return (out[..., 0], out[..., 1], c[2], c[3])

    def inv4d(self, c, ctx):
        out = self.proj.inverse(torch.stack(c, -1))
        return self._datum_out((out[..., 0], out[..., 1], c[2], c[3]))


# =============================================================================
# Registry and factory (create.cpp / init.cpp / pipeline.cpp)
# =============================================================================
REGISTRY: Dict[str, type] = {
    "noop": NoOp, "unitconvert": UnitConvert, "axisswap": AxisSwap, "cart": Cart,
    "geocent": Geocent, "geoc": Geoc, "latlong": LatLong, "longlat": LatLong, "latlon": LatLong,
    "lonlat": LatLong, "helmert": Helmert, "molobadekas": MoloBadekas, "molodensky": Molodensky,
    "hgridshift": HGridShift, "push": Push, "pop": Pop, "set": SetOp, "affine": Affine,
    "geogoffset": GeogOffset,
}


def register_operation(name: str, factory: Callable) -> None:
    """Plug in a projection: factory(dict_params) -> object with forward/inverse
    on [..., D] tensors (core.Projection API)."""
    REGISTRY[name] = type("Projection_" + name, (ProjectionStep,), {"factory": staticmethod(factory)})


def operation_class(name: str) -> type:
    if name in REGISTRY:
        return REGISTRY[name]
    if name in EXTERNAL_PROJECTIONS:
        fac = _load_external(name)
        return type("Projection_" + name, (ProjectionStep,), {"factory": staticmethod(fac)})
    raise NotSupported("operation %r not ported" % name)


def _create_op(argv: List[str]) -> Operation:
    """pj_create_argv_internal for a non-pipeline operation (init.cpp:330-790)."""
    pl = ParamList(list(argv))
    if pl.exists("init"):
        raise NotSupported("+init= is not supported")
    name = pl.s("proj")
    if name is None:
        raise ProjError("missing proj", "invalid_op_missing_arg")
    if name == "pipeline":
        raise ProjError("Pipeline: nested pipelines not allowed", "invalid_op_wrong_syntax")
    cls = operation_class(name)
    # init.cpp:322-365 append_default_ellipsoid_to_paralist
    if not any(pl.exists(k) for k in ("datum", "ellps", "a", "b", "rf", "f", "e", "es")):
        pl.append("ellps=GRS80")
    return cls(pl)


class Step(torch.nn.Module):
    def __init__(self, op: Operation, omit_fwd: bool, omit_inv: bool):
        super().__init__()
        self.op, self.omit_fwd, self.omit_inv = op, omit_fwd, omit_inv


class Pipeline(torch.nn.Module):
    """A PROJ operation created from a PROJ string (pipeline.cpp:361-639 for
    +proj=pipeline, a single step otherwise). forward()/inverse() take [..., D]."""

    def __init__(self, definition: str):
        super().__init__()
        self.definition = definition
        argv = proj_tokenize(definition)
        if not argv:
            raise ProjError("empty definition", "invalid_op_missing_arg")
        self.is_pipeline = "proj=pipeline" in argv
        if not self.is_pipeline:
            op = _create_op(argv)
            self.steps = torch.nn.ModuleList([Step(op, False, False)])
            self.left, self.right = op.pj_left(), op.pj_right()
            self._has_inverse = op.has_inv
            self._has_forward = op.has_fwd
            return
        argv = argv + ["step"]  # sentinel (pipeline.cpp:292-305)
        i_pipeline, i_first, nsteps = -1, -1, 0
        for i, a in enumerate(argv):  # pipeline.cpp:416-455
            if a == "step":
                if i_pipeline == -1:
                    raise ProjError("Pipeline: +step before +proj=pipeline", "invalid_op_wrong_syntax")
                if nsteps == 0:
                    i_first = i
                nsteps += 1
                continue
            if a == "proj=pipeline":
                if i_pipeline != -1:
                    raise ProjError("Pipeline: Nesting only allowed when child pipelines are wrapped in '+init's",
                                    "invalid_op_wrong_syntax")
                i_pipeline = i
            elif nsteps == 0 and a.startswith("proj="):
                raise ProjError("Pipeline: proj= operator before first step not allowed", "invalid_op_wrong_syntax")
            elif nsteps == 0 and a.startswith("o_proj="):
                raise ProjError("Pipeline: o_proj= operator before first step not allowed", "invalid_op_wrong_syntax")
        nsteps -= 1
        if nsteps <= 0:
            raise ProjError("Pipeline: no steps", "invalid_op_wrong_syntax")
        glob = []
        j = i_pipeline + 1
        while argv[j] != "step":
            glob.append(argv[j])
            j += 1
        steps, cur = [], i_first
        for _ in range(nsteps):  # pipeline.cpp:469-531
            j, local = cur + 1, []
            while argv[j] != "step":
                local.append(argv[j])
                j += 1
            cur = j
            args = local + glob
            try:
                op = _create_op(args)
            except ProjError as exc:
                raise ProjError("Pipeline: Bad step definition: %s" % exc, exc.errno)
            for a in args:
                if a == "inv":
                    op.inverted = not op.inverted
            steps.append(Step(op, op.params.b("omit_fwd"), op.params.b("omit_inv")))
        for st in steps:  # forward path required (pipeline.cpp:533-556)
            if st.omit_fwd:
                continue
            if st.op.inverted and not st.op.has_inv:
                raise ProjError("Pipeline: Inverse operation for %s is not available" % st.op.name,
                                "other_no_inverse_op")
            if not st.op.inverted and not st.op.has_fwd:
                raise ProjError("Pipeline: Forward operation for %s is not available" % st.op.name,
                                "invalid_op_wrong_syntax")
        self._has_forward = True
        self._has_inverse = all(st.omit_inv or st.op.has_inverse() for st in steps)
        ops = [st.op for st in steps]
        for i in range(nsteps - 2, -1, -1):  # pipeline.cpp:581-609 (WHATEVER propagation)
            pj = ops[i]
            if pj.pj_left() == WHATEVER and pj.pj_right() == WHATEVER:
                rl, rr = ops[i + 1].pj_left(), ops[i + 1].pj_right()
                if rl != rr or rl != WHATEVER:
                    pj.left = pj.right = rl
        for i in range(1, nsteps):
            pj = ops[i]
            if pj.pj_left() == WHATEVER and pj.pj_right() == WHATEVER:
                ll, lr = ops[i - 1].pj_left(), ops[i - 1].pj_right()
                if ll != lr or lr != WHATEVER:
                    pj.left = pj.right = lr
        for i in range(nsteps - 1):  # pipeline.cpp:613-627
            a, b = ops[i].pj_right(), ops[i + 1].pj_left()
            if a != WHATEVER and b != WHATEVER and a != b:
                raise ProjError("Pipeline: Mismatched units between step %d and %d" % (i + 1, i + 2),
                                "invalid_op_wrong_syntax")
        self.left, self.right = ops[0].pj_left(), ops[-1].pj_right()
        self.steps = torch.nn.ModuleList(steps)

    # ---- coordinates.cpp:53-94 ---------------------------------------------
    def angular_input(self, forward: bool = True) -> bool:
        return (self.left if forward else self.right) == RADIANS

    def angular_output(self, forward: bool = True) -> bool:
        return self.angular_input(not forward)

    def degree_input(self, forward: bool = True) -> bool:
        return (self.left if forward else self.right) == DEGREES

    def degree_output(self, forward: bool = True) -> bool:
        return self.degree_input(not forward)

    @property
    def has_inverse(self) -> bool:
        return self._has_inverse

    # ---- execution ----------------------------------------------------------
    def _run(self, coo: Tensor, fwd: bool) -> Tensor:
        D = coo.shape[-1]
        if not 2 <= D <= 4:
            raise ValueError("coordinates must have 2..4 components")
        x, y = coo[..., 0], coo[..., 1]
        z = coo[..., 2] if D > 2 else torch.zeros_like(x)
        t = coo[..., 3] if D > 3 else torch.full_like(x, INF)
        c: Coord = (x, y, z, t)
        ctx = _ExecContext() if self.is_pipeline else None
        if (fwd and not self._has_forward) or (not fwd and not self._has_inverse):
            failed = torch.ones_like(x, dtype=torch.bool)
        else:
            failed = torch.zeros_like(x, dtype=torch.bool)
            steps = list(self.steps) if fwd else list(self.steps)[::-1]
            for st in steps:  # pipeline.cpp:163-193
                if (st.omit_fwd if fwd else st.omit_inv):
                    continue
                c = pj_fwd4d(st.op, c, ctx) if (fwd != st.op.inverted) else pj_inv4d(st.op, c, ctx)
                failed = failed | torch.isnan(c[0])  # PROJ breaks out at the first failure
        out = [torch.where(failed, NAN, v) for v in c[:D]]
        return torch.stack(out, -1)

    def forward(self, coo: Tensor) -> Tensor:
        return self._run(coo, True)

    def inverse(self, coo: Tensor) -> Tensor:
        return self._run(coo, False)

    def transform(self, coo: Tensor, direction: str = "fwd") -> Tensor:
        return self._run(coo, direction.lower().startswith("f"))


def create(definition: str) -> Pipeline:
    """proj_create() equivalent for PROJ strings (pipelines or single operations)."""
    return Pipeline(definition)
