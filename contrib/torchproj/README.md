# torchproj: PROJ operations in PyTorch (feasibility spike)

This directory holds a prototype that ports a subset of PROJ's per-point
operations to PyTorch. It was written to answer one question: could PROJ be
ported to torch for weather and climate ML? It is a spike, not a library. It
has known defects (listed below), no packaging and no stable API.

The design keeps PROJ as the planner and moves only the executor to torch.
pyproj resolves a CRS pair into a PROJ pipeline string, and `pipeline.py`
parses and runs that string on tensors. Everything runs in float64 by default,
works on any device, and is differentiable with autograd.

## Modules

| File | Operations | Test |
| --- | --- | --- |
| `core.py`, `testing.py` | shared conventions, gie vector reader, pyproj reference | |
| `lcc_stere.py` | lcc, stere, ups | `lcc_stere_test.py` |
| `tmerc.py` | tmerc (exact, approx, auto, sphere), utm, etmerc | `tmerc_test.py` |
| `geos_obtran.py` | geos, ob_tran, eqc, longlat | `geos_obtran_test.py` |
| `laea_healpix_moll.py` | laea, healpix, rhealpix, moll, wag4, wag5 | `laea_healpix_moll_test.py` |
| `pipeline.py` | pipeline executor, unitconvert, axisswap, cart, helmert, molodensky, NTv2 hgridshift, push/pop | `pipeline_test.py` |
| `integ_registry.py` | merc, webmerc, and the adapter that plugs every module into `pipeline.py` | `integ_test.py`, `integ_bench.py` |

Each module also has `<module>_adv_test.py`, written by an independent
verifier to break it. Those scripts exit non-zero by design: they report the
known defects.

## Running

Tested with Python 3.13, torch 2.14.1, pyproj 3.8.0 (bundling PROJ 9.8.1) and
numpy 2.

```
pip install -r requirements.txt
cd contrib/torchproj
python tmerc_test.py          # likewise for the other *_test.py
python integ_test.py          # 10 CRS pairs end to end + scale factors vs PROJ
python integ_bench.py         # CPU throughput against pyproj
```

The tests read PROJ's gie vectors from `../../test/gie` and grids from
`../../data/tests`, so run them from inside this source tree.

Using the executor:

```python
import torch, pyproj
import integ_registry, pipeline

integ_registry.install()      # register all modules, generic PROJ layer on
defn = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True).definition
T = pipeline.create(defn)
xy = T.forward(torch.tensor([[12.5, 41.9]], dtype=torch.float64))
```

## Results

- PROJ's own gie test cases for the ported operations: 758 pass, 1 skipped.
- Ten real CRS pairs (UTM, LAEA Europe, polar stereographic, geocentric,
  British National Grid, web Mercator, WRF LCC, COSMO rotated pole, MSG
  geostationary, LAEA inverse) agree with PROJ 9.8.1 to 2e-9 to 2e-8 m in
  float64, with identical failure masks. float32 errors are 1.6 to 9.5 m.
- Scale factors from autograd Jacobians match PROJ's `proj_factors` to about
  1e-10 relative, the precision of PROJ's finite differences.
- On 4 CPU threads, `torch.compile` runs 1.6 to 7 times faster than pyproj on
  one thread; eager torch is no faster. No GPU was measured.

## Known defects

- The modules handle PROJ's generic parameters (+pm, +axis, +units, booleans,
  ellipsoid key priority, 0..360 longitudes) incompletely on their own. Use
  them through `integ_registry.install("generic")`, which runs the generic
  layer from `pipeline.py` around every kernel.
- float32 is unsafe near the poles: the oblique stereographic inverse can return
  latitudes outside [-90, 90], Mollweide flips sign, approximate tmerc returns inf.
- Gradients are zero or NaN within about 0.1 to 1 m of poles, pole images and
  antipodes, and near the apex of small-n LCC cones.
- No per-point choice among candidate operations (PROJ's area-of-use
  selection), no GeoTIFF grid reader, no vertical grid shift.
- A single non-pipeline operation with `+inv` runs forward; `geocent` with a
  non-zero `towgs84` raises.
- The implicit-gradient Mollweide solver (optional) returns zero second
  derivatives.

## Note on `src/fwd.cpp`

`src/fwd.cpp:61` tests `coo.lp.phi - M_HALFPI < -PJ_EPS_LAT` for southern
latitudes. Any latitude even 1e-15 rad below -90 degrees therefore fails,
while the northern test tolerates 1e-12 rad and clamps. PROJ 9.8.1 clamps
both. `tools/south_pole_check.c` reproduces the difference against a build of
this tree. `pipeline.py` follows this tree's behaviour.
