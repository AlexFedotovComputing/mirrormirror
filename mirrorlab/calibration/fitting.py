"""Orthogonal line fitting and rigid, scale-preserving point-to-line registration."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .models import RayFit


def fit_line(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("Для прямой нужны минимум две полные конечные точки XYZ")
    center = points.mean(axis=0)
    _, singular, axes = np.linalg.svd(points - center, full_matrices=False)
    if singular[0] < 1e-12:
        raise ValueError("Совпадающие точки не определяют направление луча")
    direction = axes[0]
    if np.dot(direction, points[1] - points[0]) < 0:
        direction = -direction
    residual = points - center
    residual -= np.outer(residual @ direction, direction)
    # For exactly two points keep the historic p0=P1 convention.
    origin = points[0] if len(points) == 2 else center
    return RayFit(origin, direction, float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))), len(points))


def fit_rigid_to_lines(points, origins, directions, *, starts=12):
    """Minimize Σ||(I-ddᵀ)(R p+t-o)||², using fixed point/ray correspondences.

    Translation is solved linearly for each trial rotation. Deterministic starts
    cover both signs of the local Z axis. No scale, reflection or permutation is
    fitted. This independent reconstruction is not attributed to Optimis_code.
    """
    points, origins, directions = [np.asarray(x, dtype=float) for x in (points, origins, directions)]
    if (points.shape != origins.shape or points.shape != directions.shape
            or points.ndim != 2 or points.shape[1] != 3 or len(points) < 4
            or not all(np.isfinite(x).all() for x in (points, origins, directions))):
        raise ValueError("Для регистрации нужны минимум четыре соответствия XYZ / прямая")
    lengths = np.linalg.norm(directions, axis=1)
    if np.any(lengths < 1e-12):
        raise ValueError("Нулевое направление луча")
    directions = directions / lengths[:, None]
    projections = np.eye(3) - np.einsum("ni,nj->nij", directions, directions)
    system = projections.sum(axis=0)
    if np.linalg.matrix_rank(system, tol=1e-10) < 3:
        raise ValueError("Параллельные прямые не определяют полный перенос")
    if np.linalg.matrix_rank(points - points.mean(axis=0), tol=1e-10) < 2:
        raise ValueError("Коллинеарные точки не определяют пространственный поворот")

    def transformed(rotvec):
        rotated = Rotation.from_rotvec(rotvec).apply(points)
        translation = np.linalg.solve(system, np.einsum("nij,nj->i", projections, origins - rotated))
        return rotated + translation, translation

    def residual(rotvec):
        moved, _ = transformed(rotvec)
        return np.einsum("nij,nj->ni", projections, moved - origins).ravel()

    fits = []
    for flip in (0.0, np.pi):
        for angle in np.linspace(-np.pi, np.pi, starts, endpoint=False):
            initial = Rotation.from_euler("zx", [angle, flip]).as_rotvec()
            fits.append(least_squares(residual, initial, method="trf", max_nfev=400,
                                      ftol=1e-11, xtol=1e-11, gtol=1e-11))
    best = min(fits, key=lambda result: result.cost)
    moved, translation = transformed(best.x)
    distances = np.linalg.norm(residual(best.x).reshape(-1, 3), axis=1)
    rank = int(np.linalg.matrix_rank(best.jac, tol=1e-8))
    return {"rotation": Rotation.from_rotvec(best.x).as_matrix().tolist(),
            "translation_m": translation.tolist(), "points_m": moved.tolist(),
            "rms_mm": float(np.sqrt(np.mean(distances**2)) * 1000),
            "max_mm": float(distances.max() * 1000), "residuals_mm": (distances * 1000).tolist(),
            "success": bool(best.success and rank == 3), "rotation_rank": rank,
            "starts": len(fits), "evaluations": sum(x.nfev for x in fits),
            "message": str(best.message)}
