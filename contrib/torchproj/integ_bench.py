"""CPU benchmark: torch pipeline executor (eager fp64/fp32, torch.compile) vs pyproj.

Run: cd contrib/torchproj && python integ_bench.py [--sizes 1e6,1e7] [--threads 4] [--no-compile]
       [--cases utm,lcc,obtran,geos,cart,hgrid,hgrid_inv] [--opcount-only]
All timings are wall clock, best of `--reps` runs, in points/second. pyproj = Transformer
.transform on float64 numpy arrays (PROJ 9.8.1, single-threaded C). Also prints per-point
operation counts (from a TorchDispatchMode trace) to reason about GPU throughput.
"""
import argparse
import gc
import math
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pipeline as PL  # noqa: E402
import integ_registry as IR  # noqa: E402

import pyproj  # noqa: E402
from pyproj import Transformer  # noqa: E402

_PROJ_ROOT = Path(__file__).resolve().parents[2]  # root of the PROJ source tree

pyproj.network.set_network_enabled(False)
pyproj.datadir.append_data_dir(f"{_PROJ_ROOT}/data/tests")

U = "+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step "
CASES = {
    # name: (definition, lon range, lat range, D, direction)
    "utm": (U + "+proj=utm +zone=32 +ellps=WGS84", (3, 15), (0, 84), 2, "fwd"),
    "lcc": (U + "+proj=lcc +lat_0=45 +lon_0=10 +lat_1=30 +lat_2=60 +x_0=0 +y_0=0 +R=6370000", (-40, 60), (15, 80), 2, "fwd"),
    "obtran": (U + "+proj=ob_tran +o_proj=longlat +o_lon_p=-170 +o_lat_p=43 +lon_0=10 +R=6371229 "
               "+step +proj=unitconvert +xy_in=rad +xy_out=deg", (-30, 50), (25, 75), 2, "fwd"),
    "geos": (U + "+proj=geos +lon_0=0 +h=35785831 +x_0=0 +y_0=0 +a=6378169 +b=6356583.8", (-80, 80), (-80, 80), 2, "fwd"),
    "cart": (U + "+proj=cart +ellps=WGS84", (-180, 180), (-90, 90), 3, "fwd"),
    "hgrid": (U + "+proj=hgridshift +grids=ntf_r93.gsb +step +proj=unitconvert +xy_in=rad +xy_out=deg",
              (-5.4, 9.9), (41.1, 51.9), 2, "fwd"),
    "hgrid_inv": (U + "+proj=hgridshift +grids=ntf_r93.gsb +step +proj=unitconvert +xy_in=rad +xy_out=deg",
                  (-5.4, 9.9), (41.1, 51.9), 2, "inv"),
}

TRANS = {"sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh", "asinh", "acosh",
         "atanh", "exp", "log", "log1p", "expm1", "pow", "hypot", "exp2", "log2", "erf"}
SQRTDIV = {"sqrt", "rsqrt", "div", "reciprocal", "remainder", "fmod", "floor_divide"}
ARITH = {"add", "sub", "mul", "neg", "abs", "rsub", "copysign", "sign", "floor", "ceil", "trunc", "round",
         "clamp", "clamp_min", "clamp_max", "minimum", "maximum", "square", "addcmul", "lerp", "sgn", "frac"}
SELECT = {"where", "lt", "le", "gt", "ge", "eq", "ne", "logical_and", "logical_or", "logical_not",
          "bitwise_and", "bitwise_or", "bitwise_not", "isnan", "isinf", "isfinite", "masked_fill", "bitwise_xor"}
MEMORY = {"index", "gather", "index_select", "cat", "stack", "clone", "copy", "_to_copy", "full_like",
          "zeros_like", "ones_like", "empty_like", "fill", "select", "slice", "unsqueeze", "view", "expand",
          "alias", "detach", "squeeze", "_unsafe_view", "lift_fresh", "t", "transpose", "unbind", "split",
          "scalar_tensor", "full", "zeros", "ones", "empty", "arange", "unsqueeze_copy", "lift_fresh_copy"}


def opcount(T, x, fwd=True):
    """Per-point counts of elementwise aten ops (outputs with >= N elements) and eager bytes/point."""
    from torch.utils._python_dispatch import TorchDispatchMode
    N = x.shape[0]
    cnt = Counter()
    nbytes = [0]

    class M(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            name = func.overloadpacket.__name__.rstrip("_")
            outs = out if isinstance(out, (tuple, list)) else [out]
            big = [o for o in outs if isinstance(o, torch.Tensor) and o.numel() >= N]
            if big:
                cnt[name] += 1
                if name not in MEMORY or name in ("index", "gather", "cat", "stack", "_to_copy", "copy", "clone"):
                    ins = [a for a in list(args) + list((kwargs or {}).values()) if isinstance(a, torch.Tensor)]
                    ins += [b for a in args if isinstance(a, (list, tuple)) for b in a if isinstance(b, torch.Tensor)]
                    nbytes[0] += sum(t.numel() * t.element_size() for t in ins + big if t.numel() >= N)
            return out
    with torch.no_grad(), M():
        (T.forward if fwd else T.inverse)(x)
    groups = Counter()
    for k, v in cnt.items():
        g = ("transcendental" if k in TRANS else "sqrt/div" if k in SQRTDIV else "arith" if k in ARITH
             else "select/compare" if k in SELECT else "memory/view" if k in MEMORY else "other:" + k)
        groups[g] += v
    trans = {k: v for k, v in cnt.items() if k in TRANS or k in SQRTDIV}
    return groups, trans, nbytes[0] / N


def timeit(fn, reps):
    best = math.inf
    for _ in range(reps):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def make_input(case, n, seed=0):
    defn, lon, lat, D, d = CASES[case]
    rng = np.random.default_rng(seed)
    a = [rng.uniform(lon[0], lon[1], n), rng.uniform(lat[0], lat[1], n)]
    if D == 3:
        a.append(rng.uniform(0, 1000, n))
    return np.stack(a, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1e6,1e7")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--opcount-only", action="store_true")
    ap.add_argument("--compile-chunk", type=float, default=1e6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    IR.install("generic")
    sizes = [int(float(s)) for s in a.sizes.split(",")]
    cases = a.cases.split(",")
    print("torch %s, threads=%d, pyproj %s / PROJ %s, CPU: %s" % (
        torch.__version__, torch.get_num_threads(), pyproj.__version__, pyproj.proj_version_str,
        open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": ")))

    print("== per-point operation counts (TorchDispatchMode, fp64, eager; ops on full-size tensors) ==")
    for case in cases:
        defn, lon, lat, D, d = CASES[case]
        T = PL.create(defn)
        x = torch.tensor(make_input(case, 4096))
        groups, trans, bpp = opcount(T, x, d == "fwd")
        print("  %-9s kernels/pt %3d | %s | transc.: %s | eager bytes moved/pt %.0f" % (
            case, sum(groups.values()), ", ".join("%s %d" % kv for kv in sorted(groups.items())),
            " ".join("%s:%d" % kv for kv in sorted(trans.items())), bpp))
    if a.opcount_only:
        return

    print("== throughput (points/s; best of %d) ==" % a.reps)
    hdr = "  %-9s %9s | %12s | %12s %12s | %12s %12s | %s" % (
        "case", "N", "pyproj f64", "torch f64", "torch f32", "compiled f64", "compiled f32", "notes")
    print(hdr)
    for case in cases:
        defn, lon, lat, D, d = CASES[case]
        T = PL.create(defn)
        tp = Transformer.from_pipeline(defn)
        fn = T.forward if d == "fwd" else T.inverse
        comp = {}
        notes = []
        if not a.no_compile:
            chunk = int(a.compile_chunk)
            for dt in (torch.float64, torch.float32):
                try:
                    torch._dynamo.reset()
                    from torch._inductor import metrics
                    metrics.reset()
                    cf = torch.compile(fn, dynamic=False)
                    xin = torch.tensor(make_input(case, chunk, 1), dtype=dt)
                    t0 = time.perf_counter()
                    cf(xin)
                    ct = time.perf_counter() - t0
                    ref = fn(xin)
                    got = cf(xin)
                    diff = (got - ref).abs().nan_to_num(0).max().item()
                    comp[dt] = cf
                    notes.append("compile %s %.0fs, %d kernels, max|compiled-eager| %.1e" % (
                        str(dt)[-7:], ct, metrics.generated_kernel_count, diff))
                except Exception as exc:
                    notes.append("compile %s failed: %s" % (str(dt)[-7:], str(exc)[:80]))
        for n in sizes:
            x64 = make_input(case, n)
            cols = [x64[:, k].copy() for k in range(x64.shape[1])]
            if d == "fwd":
                tpy = timeit(lambda: tp.transform(*cols), a.reps)
            else:
                tpy = timeit(lambda: tp.transform(*cols, direction="INVERSE"), a.reps)
            res = []
            for dt in (torch.float64, torch.float32):
                xt = torch.tensor(x64, dtype=dt)
                with torch.no_grad():
                    res.append(timeit(lambda: fn(xt), a.reps))
                del xt
            cres = []
            for dt in (torch.float64, torch.float32):
                if dt not in comp:
                    cres.append(float("nan"))
                    continue
                chunk = int(a.compile_chunk)
                xt = torch.tensor(x64, dtype=dt)
                parts = [xt[i:i + chunk] for i in range(0, n, chunk)]
                if any(p.shape[0] != chunk for p in parts):
                    cres.append(float("nan"))
                    continue
                with torch.no_grad():
                    cres.append(timeit(lambda: [comp[dt](p) for p in parts], a.reps))
                del xt, parts
            gc.collect()
            print("  %-9s %9d | %12.3g | %12.3g %12.3g | %12.3g %12.3g | peakRSS %.1f GB" % (
                case, n, n / tpy, n / res[0], n / res[1], n / cres[0], n / cres[1],
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))
        for nt in notes:
            print("            ", nt)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
