"""Preserved NumPy/CuPy geometric optics engine, separated by responsibility."""
from .backend import get_array_module, get_backend, to_numpy, scalar_to_python, asarray1, dot, norm, normalize, cross, orthonormal_basis, safe_where, project_to_transverse, count_nonzero, ArrayLike, Number, IndexModel, LIGHT_SPEED_M_S
from .materials import Material, InteractionMode, SurfaceOptics, AIR
from .rays import RayBundle, Intersection, NearestHit
from .physics import fresnel_amplitudes
from .surfaces import Surface, PlaneSurface, SphericalCapSurface, CylinderSurface
from .elements import OpticalElement, PlaneMirror, MirrorArrayBundle, BlockMirror, SemiTransparentMirror, BeamDump, Detector, CylindricalScreen, Window, SphericalLens, TriangularPrism
from .scene import CompiledScene, Scene
from .source import GaussianBeamSource
from .tracer import TraceResult, RayTracer
from .demo import build_demo_scene
