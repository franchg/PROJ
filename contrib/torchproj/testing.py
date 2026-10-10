"""Test helpers: gie vector extraction and pyproj reference values."""
import math
import re
from pathlib import Path

import numpy as np

_PROJ_ROOT = Path(__file__).resolve().parents[2]  # root of the PROJ source tree

GIE_DIR = Path(f"{_PROJ_ROOT}/test/gie")
COMMANDS = {"operation", "crs_src", "crs_dst", "use_proj4_init_rules", "accept",
            "expect", "roundtrip", "banner", "verbose", "direction", "tolerance",
            "ignore", "require_grid", "echo", "skip"}
GRS80_DEG = 111319.4908


def _scaled(args, default_scale=1.0):
    m = re.match(r"\s*([+-]?[0-9.]+(?:[eE][+-]?\d+)?)\s*([a-z]*)", args)
    if not m:
        return 0.0005
    s = float(m.group(1))
    unit = m.group(2)
    scale = {"km": 1e3, "m": 1, "dm": 0.1, "cm": 0.01, "mm": 1e-3, "um": 1e-6, "nm": 1e-9}
    if unit in scale:
        return s * scale[unit]
    if unit == "rad":
        return GRS80_DEG * math.degrees(s)
    if unit == "deg":
        return GRS80_DEG * s
    return s * default_scale


def _logical_lines(path):
    """Yield (command, args) pairs from a .gie file, following gie's rules closely
    enough for simple operation blocks."""
    text = Path(path).read_text()
    blocks = re.findall(r"<gie(-strict)?>(.*?)</gie(?:-strict)?>", text, re.S)
    for strict, body in blocks:
        cur_cmd, cur_args = None, []
        lines = body.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].split("#", 1)[0].rstrip()
            i += 1
            s = line.strip()
            if not s or set(s) <= set("-=") :
                continue
            first = s.split()[0]
            if first in COMMANDS and not line[:1].isspace():
                if cur_cmd:
                    yield cur_cmd, " ".join(cur_args)
                cur_cmd, cur_args = first, [s[len(first):].strip()]
                if strict:
                    while cur_args[-1].endswith("\\") and i < len(lines):
                        cur_args[-1] = cur_args[-1][:-1]
                        cur_args.append(lines[i].split("#", 1)[0].strip())
                        i += 1
            elif cur_cmd and not strict:
                cur_args.append(s)
        if cur_cmd:
            yield cur_cmd, " ".join(cur_args)


_DMS = re.compile(r"^([+-]?)(\d+(?:\.\d*)?)(?:d(\d+(?:\.\d*)?)?(?:'(\d+(?:\.\d*)?)?)?(?:\")?)?([NSEWnsew]?)$")


def parse_num(tok):
    """Float or PROJ DMS (e.g. 83d10'W, 41d30'27.5\"N)."""
    try:
        return float(tok)
    except ValueError:
        pass
    m = _DMS.match(tok)
    if not m:
        raise ValueError(tok)
    sign, d, mi, se, hemi = m.groups()
    v = float(d) + float(mi or 0) / 60 + float(se or 0) / 3600
    if sign == "-" or hemi in ("S", "s", "W", "w"):
        v = -v
    return v


def _clean_operation(args):
    keep = []
    for tok in args.split():
        if "=" in tok or tok.startswith("+") or tok in {"step", "inv", "omit_fwd", "omit_inv"}:
            keep.append(tok)
    return " ".join(keep)


def gie_cases(proj_name, files=None):
    """Return list of operation blocks whose operation is a single +proj=<proj_name>
    (no pipeline, no init, no grids). Each block:
      {'operation': str, 'file': str, 'cases': [ {'direction': 'fwd'|'inv',
        'accept': [floats], 'expect': [floats] | 'failure', 'tolerance_m': float} ],
       'roundtrips': [ {'accept': [...], 'n': int, 'tolerance_m': float} ] }
    Geographic values are in DEGREES as written in the gie file. For angular
    output gie measures the error as a geodesic distance in metres."""
    files = files or sorted(GIE_DIR.glob("*.gie"))
    out = []
    for f in files:
        blk = None
        direction, tol, accept = "fwd", 0.0005, None
        for cmd, args in _logical_lines(f):
            if cmd == "operation":
                op = _clean_operation(args)
                toks = op.replace("+", " ").split()
                ok = (f"proj={proj_name}" in toks and "proj=pipeline" not in toks
                      and not any(t.startswith(("init=", "nadgrids", "geoidgrids", "grids")) for t in toks))
                blk = {"operation": op, "file": f.name, "cases": [], "roundtrips": []} if ok else None
                if blk:
                    out.append(blk)
                direction, tol, accept = "fwd", 0.0005, None
            elif blk is None:
                continue
            elif cmd == "direction":
                direction = "inv" if args.strip()[:1].lower() in "ir" else "fwd"
            elif cmd == "tolerance":
                tol = _scaled(args)
            elif cmd == "accept":
                try:
                    accept = [parse_num(v) for v in args.split()[:4]]
                except ValueError:
                    accept = None
            elif cmd == "expect":
                if args.strip().startswith("failure"):
                    exp = "failure"
                else:
                    try:
                        exp = [parse_num(v) for v in args.split()[:4]]
                    except ValueError:
                        continue
                if accept is not None:
                    blk["cases"].append({"direction": direction, "accept": accept,
                                         "expect": exp, "tolerance_m": tol})
            elif cmd == "roundtrip":
                toks = args.split()
                n = int(float(toks[0])) if toks else 100
                rt_tol = _scaled(" ".join(toks[1:])) if len(toks) > 1 else tol
                if accept is not None:
                    blk["roundtrips"].append({"accept": accept, "n": n, "tolerance_m": rt_tol})
    return out


def angular_error_m(lon1_deg, lat1_deg, lon2_deg, lat2_deg, a=6378137.0):
    """Small-difference approximation of the geodesic distance (metres)."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1_deg, lat1_deg, lon2_deg, lat2_deg))
    dlon = np.remainder(lon2 - lon1 + np.pi, 2 * np.pi) - np.pi
    return a * np.hypot(lat2 - lat1, np.cos(0.5 * (lat1 + lat2)) * dlon)


def pyproj_ref(definition, x, y, inverse=False):
    """Reference values from PROJ itself (via pyproj). For a +proj=<projection>
    definition, forward input is lon/lat in DEGREES and output metres; inverse
    the other way round."""
    from pyproj import Transformer
    from pyproj.enums import TransformDirection
    defn = definition if "proj=pipeline" in definition else (
        "+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step " + definition)
    t = Transformer.from_pipeline(defn)
    d = TransformDirection.INVERSE if inverse else TransformDirection.FORWARD
    xo, yo = t.transform(np.asarray(x, dtype=float), np.asarray(y, dtype=float), direction=d)
    return np.asarray(xo), np.asarray(yo)
