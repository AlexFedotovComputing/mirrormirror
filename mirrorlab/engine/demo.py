from __future__ import annotations


import numpy as np

from .backend import normalize


from .scene import Scene


from .elements import PlaneMirror, SemiTransparentMirror, BeamDump, Detector, Window


from .materials import AIR


def build_demo_scene(n_glass: float = 1.50) -> Scene:
    """
    Small demonstration scene with a 50/50 semi-transparent mirror, one mirror and a detector.
    The geometry is *not* the imported SAT geometry; it is a clean primitive-based
    optical bench showing the intended architecture.
    """

    scene = Scene()
    scene.add(
        SemiTransparentMirror(
            name="BS_50_50",
            center=(-0.34448, -0.80321, 3.05),
            normal=normalize(np.asarray((0.0, 1.0, 1.0), dtype=float)),
            shape="rectangle",
            width=0.05,
            height=0.05,
            reflectance=0.5,
            transmittance=0.5,
            n_minus=AIR,
            n_plus=AIR,
            in_plane_reference=(1.0, 0.0, 0.0),
        ),
        PlaneMirror(
            name="Mirror_arm",
            center=(-0.34448, -0.76, 2.96),
            normal=(0.0, -1.0, 0.0),
            shape="rectangle",
            width=0.08,
            height=0.05,
            in_plane_reference=(1.0, 0.0, 0.0),
        ),
        Window(
            name="FS_window",
            center=(-0.34448, -0.80321, 2.86),
            normal=(0.0, 0.0, -1.0),
            thickness=0.008,
            n_glass=n_glass,
            n_outside=AIR,
            shape="disk",
            radius=0.02,
            use_fresnel=True,
        ),
        Detector(
            name="Detector_main",
            center=(-0.34448, -0.80321, 2.60),
            normal=(0.0, 0.0, 1.0),
            shape="rectangle",
            width=0.08,
            height=0.08,
            in_plane_reference=(1.0, 0.0, 0.0),
        ),
        BeamDump(
            name="Dump_side",
            center=(-0.34448, -0.65, 2.95),
            normal=(0.0, -1.0, 0.0),
            shape="rectangle",
            width=0.10,
            height=0.10,
            in_plane_reference=(1.0, 0.0, 0.0),
        ),
    )
    return scene
