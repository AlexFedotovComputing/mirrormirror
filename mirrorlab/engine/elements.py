from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np


from .backend import normalize, orthonormal_basis, IndexModel


from .materials import InteractionMode, SurfaceOptics, AIR


from .surfaces import Surface, PlaneSurface, SphericalCapSurface, CylinderSurface


class OpticalElement:
    def build_surfaces(self) -> List[Surface]:
        raise NotImplementedError

@dataclass
class PlaneMirror(OpticalElement):
    name: str
    center: Sequence[float]
    normal: Sequence[float]
    shape: str = "rectangle"
    width: Optional[float] = None
    height: Optional[float] = None
    radius: Optional[float] = None
    in_plane_reference: Optional[Sequence[float]] = None
    reflectance: float = 1.0
    reflect_from_minus_side: bool = True
    reflect_from_plus_side: bool = True

    def build_surfaces(self) -> List[Surface]:
        optics = SurfaceOptics(
            mode=InteractionMode.MIRROR,
            label=self.name,
            reflectance=self.reflectance,
            transmittance=0.0,
            release_reflected=True,
            release_transmitted=False,
            use_fresnel=False,
            reflect_from_minus_side=self.reflect_from_minus_side,
            reflect_from_plus_side=self.reflect_from_plus_side,
        )
        return [
            PlaneSurface(
                name=self.name,
                optics=optics,
                center=self.center,
                normal=self.normal,
                shape=self.shape,
                width=self.width,
                height=self.height,
                radius=self.radius,
                in_plane_reference=self.in_plane_reference,
            )
        ]

@dataclass
class MirrorArrayBundle(OpticalElement):
    configs: Sequence[Any]
    name: str = "MirrorArrayBundle"
    tilt_deg: float = 45.0
    reflectance: float = 0.5
    transmittance: float = 0.5
    reflect_from_minus_side: bool = True
    reflect_from_plus_side: bool = False

    def _parse_config(
        self,
        config: Any,
        idx: int,
    ) -> tuple[
        str,
        tuple[float, float, float],
        float,
        float,
        Optional[tuple[float, float, float]],
        Optional[tuple[float, float, float]],
    ]:
        if isinstance(config, dict):
            center = tuple(float(v) for v in config["center"])
            phi_deg = float(config.get("phi_deg", config.get("phi")))
            radius = float(config["radius"])
            name = str(config.get("name", f"{self.name} Mirror {idx}"))
            normal = config.get("normal")
            in_plane_reference = config.get("in_plane_reference")
            return (
                name,
                center,
                phi_deg,
                radius,
                None if normal is None else tuple(float(v) for v in normal),
                None if in_plane_reference is None else tuple(float(v) for v in in_plane_reference),
            )

        if len(config) == 3:
            center, phi_deg, radius = config
            name = f"{self.name} Mirror {idx}"
            return name, tuple(float(v) for v in center), float(phi_deg), float(radius), None, None

        if len(config) == 4:
            name, center, phi_deg, radius = config
            return str(name), tuple(float(v) for v in center), float(phi_deg), float(radius), None, None

        raise ValueError(f"Unsupported mirror bundle config at index {idx}: {config!r}")

    def build_surfaces(self) -> List[OpticalElement]:
        tilt_rad = math.radians(float(self.tilt_deg))
        sin_tilt = math.sin(tilt_rad)
        cos_tilt = math.cos(tilt_rad)
        mirrors: List[OpticalElement] = []

        for idx, config in enumerate(self.configs, start=1):
            name, center, phi_deg, radius, normal_override, in_plane_reference_override = self._parse_config(config, idx)
            phi_rad = math.radians(phi_deg)
            if normal_override is None:
                normal = normalize(
                    np.array(
                        [
                            sin_tilt * math.sin(phi_rad),
                            -sin_tilt * math.cos(phi_rad),
                            cos_tilt,
                        ],
                        dtype=float,
                    )
                )
            else:
                normal = normalize(np.asarray(normal_override, dtype=float))

            if in_plane_reference_override is None:
                in_plane_reference = normalize(
                    np.array(
                        [
                            math.cos(phi_rad),
                            math.sin(phi_rad),
                            0.0,
                        ],
                        dtype=float,
                    )
                )
            else:
                in_plane_reference = normalize(np.asarray(in_plane_reference_override, dtype=float))
            mirrors.append(
                SemiTransparentMirror(
                    name=name,
                    center=tuple(float(v) for v in center),
                    normal=tuple(float(v) for v in normal),
                    shape="disk",
                    radius=float(radius),
                    in_plane_reference=tuple(float(v) for v in in_plane_reference),
                    reflectance=self.reflectance,
                    transmittance=self.transmittance,
                    reflect_from_minus_side=self.reflect_from_minus_side,
                    reflect_from_plus_side=self.reflect_from_plus_side,
                )
            )

        return mirrors

@dataclass
class BlockMirror(OpticalElement):
    name: str
    center: Sequence[float]
    normal: Sequence[float]
    width: float
    height: float
    thickness: float
    in_plane_reference: Optional[Sequence[float]] = None
    reflectance: float = 1.0

    def build_surfaces(self) -> List[Surface]:
        c = np.asarray(self.center, dtype=float)
        w_ax = normalize(np.asarray(self.normal, dtype=float))
        u_ax, v_ax, _ = orthonormal_basis(w_ax, self.in_plane_reference, xp=np)

        optics = SurfaceOptics(
            mode=InteractionMode.MIRROR,
            label=self.name,
            reflectance=self.reflectance,
            transmittance=0.0,
            release_reflected=True,
            release_transmitted=False,
            use_fresnel=False,
        )

        half_t = 0.5 * self.thickness
        half_w = 0.5 * self.width
        half_h = 0.5 * self.height

        surfaces = [
            PlaneSurface(
                name=f"{self.name}:front",
                optics=optics,
                center=c - half_t * w_ax,
                normal=-w_ax,
                shape="rectangle",
                width=self.width,
                height=self.height,
                in_plane_reference=u_ax,
            ),
            PlaneSurface(
                name=f"{self.name}:back",
                optics=optics,
                center=c + half_t * w_ax,
                normal=w_ax,
                shape="rectangle",
                width=self.width,
                height=self.height,
                in_plane_reference=u_ax,
            ),
            PlaneSurface(
                name=f"{self.name}:side_u_minus",
                optics=optics,
                center=c - half_w * u_ax,
                normal=-u_ax,
                shape="rectangle",
                width=self.thickness,
                height=self.height,
                in_plane_reference=w_ax,
            ),
            PlaneSurface(
                name=f"{self.name}:side_u_plus",
                optics=optics,
                center=c + half_w * u_ax,
                normal=u_ax,
                shape="rectangle",
                width=self.thickness,
                height=self.height,
                in_plane_reference=w_ax,
            ),
            PlaneSurface(
                name=f"{self.name}:side_v_minus",
                optics=optics,
                center=c - half_h * v_ax,
                normal=-v_ax,
                shape="rectangle",
                width=self.thickness,
                height=self.width,
                in_plane_reference=w_ax,
            ),
            PlaneSurface(
                name=f"{self.name}:side_v_plus",
                optics=optics,
                center=c + half_h * v_ax,
                normal=v_ax,
                shape="rectangle",
                width=self.thickness,
                height=self.width,
                in_plane_reference=w_ax,
            ),
        ]
        return surfaces

@dataclass
class SemiTransparentMirror(OpticalElement):
    name: str
    center: Sequence[float]
    normal: Sequence[float]
    n_minus: IndexModel = AIR
    n_plus: IndexModel = AIR
    reflectance: float = 0.5
    transmittance: float = 0.5
    shape: str = "rectangle"
    width: Optional[float] = None
    height: Optional[float] = None
    radius: Optional[float] = None
    in_plane_reference: Optional[Sequence[float]] = None
    # Convenience alias for the interface side pointed to by the surface normal.
    refractive_index: Optional[IndexModel] = None
    thickness: Optional[float] = None
    n_glass: Optional[IndexModel] = None
    n_outside: IndexModel = AIR
    front_reflectance: Optional[float] = None
    front_transmittance: Optional[float] = None
    back_reflectance: Optional[float] = None
    back_transmittance: Optional[float] = None
    reflect_from_minus_side: bool = True
    reflect_from_plus_side: bool = True

    def build_surfaces(self) -> List[Surface]:
        if self.thickness is not None:
            center = np.asarray(self.center, dtype=float)
            normal = normalize(np.asarray(self.normal, dtype=float))
            axis_u, axis_v, axis_w = orthonormal_basis(normal, self.in_plane_reference, xp=np)
            front = center - 0.5 * self.thickness * normal
            back = center + 0.5 * self.thickness * normal
            n_glass = self.n_glass if self.n_glass is not None else self.refractive_index
            if n_glass is None:
                n_glass = self.n_plus
            front_reflectance = self.reflectance if self.front_reflectance is None else self.front_reflectance
            front_transmittance = self.transmittance if self.front_transmittance is None else self.front_transmittance
            back_reflectance = self.reflectance if self.back_reflectance is None else self.back_reflectance
            back_transmittance = self.transmittance if self.back_transmittance is None else self.back_transmittance
            side_reflectance = self.reflectance
            side_transmittance = self.transmittance
            surfaces: List[Surface] = [
                PlaneSurface(
                    name=f"{self.name}:front",
                    optics=SurfaceOptics(
                        mode=InteractionMode.BEAMSPLITTER,
                        label=f"{self.name}:front",
                        n_minus=self.n_outside,
                        n_plus=n_glass,
                        reflectance=front_reflectance,
                        transmittance=front_transmittance,
                        release_reflected=True,
                        release_transmitted=True,
                        use_fresnel=False,
                        reflect_from_minus_side=front_reflectance > 0.0,
                        reflect_from_plus_side=False,
                    ),
                    center=front,
                    normal=normal,
                    shape=self.shape,
                    width=self.width,
                    height=self.height,
                    radius=self.radius,
                    in_plane_reference=self.in_plane_reference,
                ),
                PlaneSurface(
                    name=f"{self.name}:back",
                    optics=SurfaceOptics(
                        mode=InteractionMode.BEAMSPLITTER,
                        label=f"{self.name}:back",
                        n_minus=n_glass,
                        n_plus=self.n_outside,
                        reflectance=back_reflectance,
                        transmittance=back_transmittance,
                        release_reflected=True,
                        release_transmitted=True,
                        use_fresnel=False,
                        reflect_from_minus_side=False,
                        reflect_from_plus_side=back_reflectance > 0.0,
                    ),
                    center=back,
                    normal=normal,
                    shape=self.shape,
                    width=self.width,
                    height=self.height,
                    radius=self.radius,
                    in_plane_reference=self.in_plane_reference,
                ),
            ]
            if self.shape == "rectangle" and self.width is not None and self.height is not None:
                side_optics_out_to_in = SurfaceOptics(
                    mode=InteractionMode.BEAMSPLITTER,
                    label=f"{self.name}:side",
                    n_minus=self.n_outside,
                    n_plus=n_glass,
                    reflectance=side_reflectance,
                    transmittance=side_transmittance,
                    release_reflected=True,
                    release_transmitted=True,
                    use_fresnel=False,
                    reflect_from_minus_side=True,
                    reflect_from_plus_side=False,
                )
                side_optics_in_to_out = SurfaceOptics(
                    mode=InteractionMode.BEAMSPLITTER,
                    label=f"{self.name}:side",
                    n_minus=n_glass,
                    n_plus=self.n_outside,
                    reflectance=side_reflectance,
                    transmittance=side_transmittance,
                    release_reflected=True,
                    release_transmitted=True,
                    use_fresnel=False,
                    reflect_from_minus_side=False,
                    reflect_from_plus_side=True,
                )
                surfaces.extend(
                    [
                        PlaneSurface(
                            name=f"{self.name}:side_u_minus",
                            optics=side_optics_out_to_in,
                            center=center - 0.5 * self.width * axis_u,
                            normal=axis_u,
                            shape="rectangle",
                            width=self.thickness,
                            height=self.height,
                            in_plane_reference=axis_w,
                        ),
                        PlaneSurface(
                            name=f"{self.name}:side_u_plus",
                            optics=side_optics_in_to_out,
                            center=center + 0.5 * self.width * axis_u,
                            normal=axis_u,
                            shape="rectangle",
                            width=self.thickness,
                            height=self.height,
                            in_plane_reference=axis_w,
                        ),
                        PlaneSurface(
                            name=f"{self.name}:side_v_minus",
                            optics=side_optics_out_to_in,
                            center=center - 0.5 * self.height * axis_v,
                            normal=axis_v,
                            shape="rectangle",
                            width=self.thickness,
                            height=self.width,
                            in_plane_reference=axis_w,
                        ),
                        PlaneSurface(
                            name=f"{self.name}:side_v_plus",
                            optics=side_optics_in_to_out,
                            center=center + 0.5 * self.height * axis_v,
                            normal=axis_v,
                            shape="rectangle",
                            width=self.thickness,
                            height=self.width,
                            in_plane_reference=axis_w,
                        ),
                    ]
                )
            return surfaces

        n_minus = self.n_minus
        n_plus = self.n_plus
        if self.refractive_index is not None:
            n_plus = self.refractive_index
        optics = SurfaceOptics(
            mode=InteractionMode.BEAMSPLITTER,
            label=self.name,
            n_minus=n_minus,
            n_plus=n_plus,
            reflectance=self.reflectance,
            transmittance=self.transmittance,
            release_reflected=True,
            release_transmitted=True,
            use_fresnel=False,
            reflect_from_minus_side=self.reflect_from_minus_side,
            reflect_from_plus_side=self.reflect_from_plus_side,
        )
        return [
            PlaneSurface(
                name=self.name,
                optics=optics,
                center=self.center,
                normal=self.normal,
                shape=self.shape,
                width=self.width,
                height=self.height,
                radius=self.radius,
                in_plane_reference=self.in_plane_reference,
            )
        ]

@dataclass
class BeamDump(OpticalElement):
    name: str
    center: Sequence[float]
    normal: Sequence[float]
    shape: str = "rectangle"
    width: Optional[float] = None
    height: Optional[float] = None
    radius: Optional[float] = None
    in_plane_reference: Optional[Sequence[float]] = None
    detector: bool = False

    def build_surfaces(self) -> List[Surface]:
        optics = SurfaceOptics(
            mode=InteractionMode.ABSORB,
            label=self.name,
            detector=self.detector,
            release_reflected=False,
            release_transmitted=False,
        )
        return [
            PlaneSurface(
                name=self.name,
                optics=optics,
                center=self.center,
                normal=self.normal,
                shape=self.shape,
                width=self.width,
                height=self.height,
                radius=self.radius,
                in_plane_reference=self.in_plane_reference,
            )
        ]

@dataclass
class Detector(BeamDump):
    detector: bool = True

@dataclass
class CylindricalScreen(OpticalElement):
    name: str
    center: Sequence[float]
    axis: Sequence[float]
    radius: float
    length: float
    detector: bool = True

    def build_surfaces(self) -> List[Surface]:
        optics = SurfaceOptics(
            mode=InteractionMode.ABSORB,
            label=self.name,
            detector=self.detector,
            release_reflected=False,
            release_transmitted=False,
        )
        return [
            CylinderSurface(
                name=self.name,
                optics=optics,
                center=self.center,
                axis=self.axis,
                radius=self.radius,
                length=self.length,
            )
        ]

@dataclass
class Window(OpticalElement):
    name: str
    center: Sequence[float]
    normal: Sequence[float]
    thickness: float
    n_glass: IndexModel
    n_outside: IndexModel = AIR
    shape: str = "disk"
    width: Optional[float] = None
    height: Optional[float] = None
    radius: Optional[float] = None
    in_plane_reference: Optional[Sequence[float]] = None
    use_fresnel: bool = True

    def build_surfaces(self) -> List[Surface]:
        center = np.asarray(self.center, dtype=float)
        normal = normalize(np.asarray(self.normal, dtype=float))
        front = center - 0.5 * self.thickness * normal
        back = center + 0.5 * self.thickness * normal

        optics_front = SurfaceOptics(
            mode=InteractionMode.REFRACT,
            label=f"{self.name}:front",
            n_minus=self.n_outside,
            n_plus=self.n_glass,
            release_reflected=False,
            release_transmitted=True,
            use_fresnel=self.use_fresnel,
        )
        optics_back = SurfaceOptics(
            mode=InteractionMode.REFRACT,
            label=f"{self.name}:back",
            n_minus=self.n_glass,
            n_plus=self.n_outside,
            release_reflected=False,
            release_transmitted=True,
            use_fresnel=self.use_fresnel,
        )
        return [
            PlaneSurface(
                name=f"{self.name}:front",
                optics=optics_front,
                center=front,
                normal=normal,
                shape=self.shape,
                width=self.width,
                height=self.height,
                radius=self.radius,
                in_plane_reference=self.in_plane_reference,
            ),
            PlaneSurface(
                name=f"{self.name}:back",
                optics=optics_back,
                center=back,
                normal=normal,
                shape=self.shape,
                width=self.width,
                height=self.height,
                radius=self.radius,
                in_plane_reference=self.in_plane_reference,
            ),
        ]

@dataclass
class SphericalLens(OpticalElement):
    """
    A simple thick lens assembled from two spherical caps.

    Sign convention:
    - front vertex = center - 0.5 * thickness * axis
    - back vertex  = center + 0.5 * thickness * axis
    - the geometric normal at both vertices points along +axis
    - therefore a typical biconvex lens is: radius_front > 0, radius_back < 0
    """

    name: str
    center: Sequence[float]
    axis: Sequence[float]
    thickness: float
    aperture_radius: float
    radius_front: float
    radius_back: float
    n_lens: IndexModel
    n_outside: IndexModel = AIR
    use_fresnel: bool = True

    def build_surfaces(self) -> List[Surface]:
        c = np.asarray(self.center, dtype=float)
        a = normalize(np.asarray(self.axis, dtype=float))
        v_front = c - 0.5 * self.thickness * a
        v_back = c + 0.5 * self.thickness * a
        front = SphericalCapSurface(
            name=f"{self.name}:front",
            optics=SurfaceOptics(
                mode=InteractionMode.REFRACT,
                label=f"{self.name}:front",
                n_minus=self.n_outside,
                n_plus=self.n_lens,
                release_reflected=False,
                release_transmitted=True,
                use_fresnel=self.use_fresnel,
            ),
            vertex=v_front,
            axis=a,
            curvature_radius=self.radius_front,
            aperture_radius=self.aperture_radius,
        )
        back = SphericalCapSurface(
            name=f"{self.name}:back",
            optics=SurfaceOptics(
                mode=InteractionMode.REFRACT,
                label=f"{self.name}:back",
                n_minus=self.n_lens,
                n_plus=self.n_outside,
                release_reflected=False,
                release_transmitted=True,
                use_fresnel=self.use_fresnel,
            ),
            vertex=v_back,
            axis=a,
            curvature_radius=self.radius_back,
            aperture_radius=self.aperture_radius,
        )
        return [front, back]

@dataclass
class TriangularPrism(OpticalElement):
    name: str
    center: Sequence[float]
    normal: Sequence[float]
    in_plane_reference: Sequence[float]
    vertices_2d: Sequence[Sequence[float]]
    thickness: float
    n_glass: IndexModel
    n_outside: IndexModel = AIR
    reflectance: float = 0.5
    transmittance: float = 0.5
    bottom_reflectance: Optional[float] = None
    bottom_transmittance: Optional[float] = None
    top_reflectance: Optional[float] = None
    top_transmittance: Optional[float] = None
    side_reflectances: Optional[Sequence[Optional[float]]] = None
    side_transmittances: Optional[Sequence[Optional[float]]] = None

    def build_surfaces(self) -> List[Surface]:
        c = np.asarray(self.center, dtype=float)
        n = normalize(np.asarray(self.normal, dtype=float))
        u_ax, v_ax, w_ax = orthonormal_basis(n, self.in_plane_reference, xp=np)
        half_t = 0.5 * self.thickness

        def _surface_optics(
            reflectance: Optional[float],
            transmittance: Optional[float],
        ) -> SurfaceOptics:
            r = self.reflectance if reflectance is None else float(reflectance)
            t = self.transmittance if transmittance is None else float(transmittance)
            return SurfaceOptics(
                mode=InteractionMode.BEAMSPLITTER,
                label=self.name,
                n_minus=self.n_outside,
                n_plus=self.n_glass,
                reflectance=r,
                transmittance=t,
                release_reflected=True,
                release_transmitted=True,
                use_fresnel=False,
                reflect_from_minus_side=r > 0.0,
                reflect_from_plus_side=False,
            )

        surfaces: List[Surface] = []
        surfaces.append(
            PlaneSurface(
                name=f"{self.name}:bottom",
                optics=_surface_optics(self.bottom_reflectance, self.bottom_transmittance),
                center=c - half_t * n,
                normal=n,
                shape="triangle",
                triangle_vertices=self.vertices_2d,
                in_plane_reference=u_ax,
            )
        )
        surfaces.append(
            PlaneSurface(
                name=f"{self.name}:top",
                optics=_surface_optics(self.top_reflectance, self.top_transmittance),
                center=c + half_t * n,
                normal=-n,
                shape="triangle",
                # Reversing the normal also reverses local v. Compensate so
                # both caps have the same footprint in world coordinates.
                triangle_vertices=[(x, -y) for x, y in self.vertices_2d],
                in_plane_reference=u_ax,
            )
        )

        for i in range(3):
            p1, p2 = self.vertices_2d[i], self.vertices_2d[(i + 1) % 3]
            edge_c_2d = (0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1]))
            edge_vec = (p2[0] - p1[0], p2[1] - p1[1])
            edge_len = math.hypot(*edge_vec)
            inward_n_2d = (-edge_vec[1] / edge_len, edge_vec[0] / edge_len)
            side_reflectance = None if self.side_reflectances is None else self.side_reflectances[i]
            side_transmittance = None if self.side_transmittances is None else self.side_transmittances[i]

            side_c = c + edge_c_2d[0] * u_ax + edge_c_2d[1] * v_ax
            side_n = inward_n_2d[0] * u_ax + inward_n_2d[1] * v_ax
            surfaces.append(
                PlaneSurface(
                    name=f"{self.name}:side_{i}",
                    optics=_surface_optics(side_reflectance, side_transmittance),
                    center=side_c,
                    normal=side_n,
                    shape="rectangle",
                    # With ``n`` as the in-plane reference, PlaneSurface's
                    # local u-axis runs through the prism thickness and its
                    # local v-axis runs along this triangular edge.
                    width=self.thickness,
                    height=edge_len,
                    in_plane_reference=n,
                )
            )

        return surfaces
