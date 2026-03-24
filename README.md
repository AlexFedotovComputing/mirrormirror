# COMSOL-like ray tracing on Python

This is a compact primitive-based ray-tracing framework intended as a practical replacement for a COMSOL Geometrical Optics model when you want:

- explicit OOP assembly of an optical scheme,
- simple scene compilation,
- Gaussian-beam launch with a COMSOL-like hexapolar grid,
- mirrors, refractive interfaces, 50/50 beam splitters, windows, detectors and beam dumps,
- one code path for CPU (`numpy`) and optional GPU (`cupy`).

## What is implemented

### Source
- `GaussianBeamSource`
  - waist position,
  - beam axis,
  - waist radius,
  - peak intensity,
  - wavelength,
  - polarization reference direction,
  - hexapolar grid with `1 + 6 * sum(k)` points,
  - per-ray intensity and power weights.

### Geometry primitives
- `PlaneSurface`
  - rectangle,
  - disk,
  - infinite plane.
- `SphericalCapSurface`
- `CylinderSurface`

### Optical elements
- `PlaneMirror`
- `BeamSplitter`
- `Window`
- `SphericalLens`
- `Detector`
- `BeamDump`

### Physics
- specular reflection,
- Snell refraction,
- optional Fresnel power coefficients,
- user-defined 50/50 splitting,
- transmit-only or absorb-only boundaries,
- ray branching.

## Installation

Minimum:

```bash
pip install numpy
```

Optional:

```bash
pip install matplotlib plotly
# and a matching CuPy build for your CUDA installation
```

## Quick start

```python
from comsol_like_raytrace import GaussianBeamSource, RayTracer, build_demo_scene

source = GaussianBeamSource(
    waist_position=(-0.34448, -0.80321, 3.3199),
    axis=(0.0, 0.0, -1.0),
    waist_radius=0.01,
    peak_intensity=8.49e10,
    wavelength_m=266e-9,
    polarization_reference=(0.0, 1.0, 0.0),
    radial_positions=15,
    cutoff_ratio=1.0,
    n_medium=1.0,
    backend="numpy",
)

scene = build_demo_scene(n_glass=1.50)  # placeholder value
tracer = RayTracer(scene=scene, backend="numpy", max_interactions=8)
result = tracer.trace(source.emit())

print(result.detector_power_summary())
```

## Demo script

```bash
python example_comsol_like.py --backend numpy --plot
```

This writes:

- `segments.csv`
- `detector_hits.csv`
- optionally `trajectories.html`

## Notes

- The scene in `build_demo_scene()` is only a clean primitive-based demonstration.
- It is **not** the imported SAT geometry.
- The report does not contain enough information to reconstruct every surface automatically with exact labels and material tables, so the intended workflow is to build the optical scheme directly in code from primitives.
- For the uploaded COMSOL report, the source settings already match the exported Gaussian beam: waist position `(-0.34448, -0.80321, 3.3199) m`, axis `(0, 0, -1)`, `w0 = 10 mm`, 15 radial positions and 721 total rays, peak intensity `8.49e10 W/m^2`, polarization reference `(0, 1, 0)`.

## Limitations of this first version

- no imported CAD/mesh intersections,
- no BVH or triangle-mesh accelerator yet,
- no full Stokes/Jones phase tracking,
- no wave optics,
- no exact COMSOL parametric-sweep recreation,
- material dispersion must be supplied by the user.

## Recommended next steps

1. Replace placeholder refractive indices by your measured or tabulated values.
2. Translate the main optical bench into primitives directly in Python.
3. Validate each arm against COMSOL one subsystem at a time.
4. Only after profiling, move hot kernels to a native extension if needed.
