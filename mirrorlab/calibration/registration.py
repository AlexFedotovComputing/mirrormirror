"""Recover the measured coordinate registration from existing paired tables."""
import numpy as np

from .io import read_equations


def fit_rigid_points(source, target):
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if (source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3
            or len(source) < 3 or not np.isfinite(source).all() or not np.isfinite(target).all()):
        raise ValueError("Для привязки нужны минимум три пары конечных координат XYZ")
    x, y = source - source.mean(axis=0), target - target.mean(axis=0)
    if min(np.linalg.matrix_rank(x), np.linalg.matrix_rank(y)) < 2:
        raise ValueError("Коллинеарные опорные точки не определяют привязку")
    u, _, vt = np.linalg.svd(x.T @ y)
    correction = np.diag([1, 1, np.linalg.det(vt.T @ u.T)])
    rotation = vt.T @ correction @ u.T
    translation = target.mean(axis=0) - rotation @ source.mean(axis=0)
    residuals = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
    return {"rotation": rotation.tolist(), "translation_m": translation.tolist(),
            "rms_mm": float(np.sqrt(np.mean(residuals**2)) * 1000),
            "max_mm": float(residuals.max() * 1000), "point_count": len(source)}


def register_west_to_east(rays, data_dir, *, scene_z_offset=-6.63974, references=None):
    source_path = data_dir / "из-к+_уравнения_лучей.txt"
    target_path = data_dir / "уравнения_лучей_в_СК_идеальной_геометрии.txt"
    source, target = read_equations(source_path), read_equations(target_path)
    keys = sorted(key for key in source.keys() & target.keys()
                  if all(x is not None for x in source[key][:3] + target[key][:3]))
    source_xyz = np.array([source[key][:3] for key in keys]) / 1000
    target_xyz = np.array([target[key][:3] for key in keys]) / 1000
    target_xyz[:, 2] += scene_z_offset
    transform = fit_rigid_points(source_xyz, target_xyz)
    transform.update(method="Kabsch_from_paired_original_tables", coordinate_frame="scene",
                     sources=[source_path.name, target_path.name], scene_z_offset_m=scene_z_offset)
    # Files have millimetre coordinates rounded to 0.001 mm. A substantially
    # larger mismatch means a single rigid mapping no longer explains them.
    transform["accepted"] = transform["max_mm"] <= .01
    registered, distances = [], []
    if transform["accepted"]:
        rotation, translation = np.array(transform["rotation"]), np.array(transform["translation_m"])
        for ray in rays:
            if ray["direction"] is None:
                continue
            point = rotation @ ray["point_m"] + translation
            direction = rotation @ ray["direction"]
            # This sector convention is explicit in the archived diagnostic
            # _analyze_hex_bundle_fit.py, but missing from the old main script.
            if ray["rod"] in (1, 3):
                point[:2] *= -1
                direction[:2] *= -1
            if references and ray["id"] in references:
                center = np.array(references[ray["id"]]["point_m"])
                parameter = np.dot(center - point, direction)
                distance = float(np.linalg.norm(center - point - parameter * direction))
                distances.append({"ray_id": ray["id"], "line_distance_mm": distance * 1000})
                # Survey direction P1->P2 may point back toward the mirror.
                # Orient it from the mirror toward the measured screen point.
                if parameter > 0:
                    direction *= -1
            registered.append({"ray_id": ray["id"], "point_m": point.tolist(),
                               "direction": direction.tolist()})
    compatible = (bool(registered) and len(distances) == len(registered)
                  and max(item["line_distance_mm"] for item in distances) <= 2.0)
    warnings = [] if compatible else [
        "Привязка двух таблиц воспроизведена, но положения лучей не согласованы с текущими зеркалами в пределах 2 мм. Автоматическое изменение зеркал отключено."]
    return {"transform": transform, "accepted": compatible,
            "rays": registered if compatible else [], "candidate_rays": registered,
            "sector_convention": "Поворот XY на 180° для стержней 1 и 3 из legacy/_analyze_hex_bundle_fit.py",
            "compatibility": {"max_line_distance_mm": max((x["line_distance_mm"] for x in distances), default=None),
                              "limit_mm": 2.0, "distances": distances}, "warnings": warnings}
