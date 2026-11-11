"""Register both archived optical scenes to the original survey XYZ frame.

Optics is traced in its native frame. One distance-preserving map then converts
every displayed point, including detectors and the source. Survey points never
pass through that map. The two models share a gas axis and a centre plane z=0.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .calibration.io import FILES, read_survey
from .configurations import DATA_DIRECTORY, configuration_ray_references, load_configuration

ROOT = Path(__file__).resolve().parents[1]


def intersect_ray_group(points, directions):
    """Estimate a microassembly centre from its diverging measured ray lines."""
    points, directions = np.asarray(points, float), np.asarray(directions, float)
    lengths = np.linalg.norm(directions, axis=1)
    if len(points) < 2 or np.any(lengths < 1e-12) or not np.isfinite([points, directions]).all():
        raise ValueError("Недостаточно полных лучей для привязки микросборки")
    directions = directions / lengths[:, None]
    projectors = np.eye(3) - directions[:, :, None] * directions[:, None, :]
    matrix = projectors.reshape(-1, 3)
    if np.linalg.cond(matrix) > 1e6:
        raise ValueError("Параллельные лучи не определяют положение микросборки")
    rhs = np.einsum("nij,nj->ni", projectors, points).ravel()
    initial = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    fit = least_squares(lambda p: matrix @ p - rhs, initial, loss="soft_l1", f_scale=.002)
    residuals = np.linalg.norm((matrix @ fit.x - rhs).reshape(-1, 3), axis=1)
    return fit.x, float(np.sqrt(np.mean(residuals**2)))


def _xy_matrix(angle, parity):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -parity * s], [s, parity * c]])


@dataclass(frozen=True)
class SurveyFrame:
    rotation: np.ndarray
    translation: np.ndarray
    report: dict
    measurements: tuple

    def points(self, values):
        return np.asarray(values) @ self.rotation.T + self.translation

    def vectors(self, values):
        return np.asarray(values) @ self.rotation.T

    def transform_traces(self, traces):
        for trace in traces:
            if not all(axis in trace for axis in "xyz"):
                continue
            xyz = np.array([trace[axis] for axis in "xyz"], dtype=float).T
            valid = np.isfinite(xyz).all(axis=1)
            xyz[valid] = self.points(xyz[valid])
            for i, axis in enumerate("xyz"):
                trace[axis] = [float(p[i]) if keep else None for p, keep in zip(xyz, valid)]

    def metadata(self):
        return dict(coordinate_frame="survey", calculation_frame="legacy",
                    rotation=self.rotation.tolist(), translation_m=self.translation.tolist(),
                    **self.report)


def _side_data(direction):
    config = load_configuration(direction, "ideal")
    references = configuration_ray_references(config)
    rows, _ = read_survey(ROOT / FILES[direction])
    groups = {}
    for row in rows:
        if len(row.points) == 2:
            groups.setdefault((row.rod, row.window), []).append(row)
    anchors, keys, errors, counts = [], [], [], []
    for key, group in sorted(groups.items()):
        point, error = intersect_ray_group([r.points[0] for r in group],
                                          [r.points[1] - r.points[0] for r in group])
        keys.append(key)
        anchors.append(point)
        errors.append(error)
        counts.append(len(group))
    if len(keys) < 8 or len({rod for rod, _ in keys}) != 4:
        raise ValueError("Для общей привязки нужны измерения четырёх стержней минимум на двух уровнях")
    target = np.array(anchors)
    midpoint = (config.tracer_options["bundle_clip_z_min_m"] + config.tracer_options["bundle_clip_z_max_m"]) / 2
    # Test both axial directions and both window orders. The known centre of
    # the gas volume fixes z=0, eliminating a spurious solution with both the
    # laser and the injection mirrors on the same half of the apparatus.
    candidates = []
    for reverse in (False, True):
        source = np.array([np.mean([references[f"{rod}_{5-window if reverse else window}_{ray}"]["point_m"]
                                   for ray in range(1, 8)], axis=0) for rod, window in keys])
        for sign in (-1, 1):
            dz = sign * (source[:, 2] - midpoint) - target[:, 2]
            candidates.append((float(np.mean(dz**2)), source, sign, reverse))
    score, source, z_sign, reverse = min(candidates, key=lambda item: item[0])
    if np.sqrt(score) > .05:
        raise ValueError("Уровни микросборок не согласуются с геодезией; проверьте нумерацию окон и начало z")
    u, _, vt = np.linalg.svd((source[:, :2] - source[:, :2].mean(0)).T @
                           (target[:, :2] - target[:, :2].mean(0)))
    xy = vt.T @ u.T
    return dict(config=config, rows=tuple(rows), keys=keys, source=source, target=target,
                errors=np.array(errors), counts=counts, z_sign=z_sign, z_shift=-z_sign * midpoint,
                reverse=reverse, angle=np.arctan2(xy[1, 0], xy[0, 0]), parity=round(np.linalg.det(xy)))


def survey_frames():
    paths = [ROOT / FILES[d] for d in FILES] + [DATA_DIRECTORY / f"{d}_ideal.json" for d in FILES]
    signature = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths)
    return _fit_frames(signature)


@lru_cache(maxsize=4)
def _fit_frames(signature):
    sides = {direction: _side_data(direction) for direction in FILES}
    values = list(sides.values())

    def residual(parameters):
        chunks = []
        for angle, side in zip(parameters[:2], values):
            moved = side["source"][:, :2] @ _xy_matrix(angle, side["parity"]).T + parameters[2:]
            weight = np.minimum(1., .002 / np.maximum(side["errors"], 1e-12))
            chunks.append(((moved - side["target"][:, :2]) * np.sqrt(weight[:, None])).ravel())
        return np.concatenate(chunks)

    # Both rings belong to the same cylindrical apparatus: fit a shared XY axis
    # offset rather than independently translating two copies of the volume.
    fit = least_squares(residual, [s["angle"] for s in values] + [0., 0.],
                        loss="soft_l1", f_scale=.005)
    frames = {}
    for angle, (direction, side) in zip(fit.x[:2], sides.items()):
        rotation = np.zeros((3, 3))
        rotation[:2, :2] = _xy_matrix(angle, side["parity"])
        rotation[2, 2] = side["z_sign"]
        translation = np.r_[fit.x[2:], side["z_shift"]]
        predicted = side["source"] @ rotation.T + translation
        distances = np.linalg.norm(predicted - side["target"], axis=1) * 1000
        caps = [s.center[2] for s in side["config"].scene.build_surfaces()
                if getattr(s, "shape", None) == "triangle"]
        ring = rotation @ [0., 0., float(np.mean(caps))] + translation
        report = dict(status="estimated", method="measured_ray_intersections_shared_gas_axis",
                      source=FILES[direction], source_signature=[list(x) for x in signature],
                      gas_axis_point_m=[*fit.x[2:].tolist(), 0.], ring_center_m=ring.tolist(),
                      rms_mm=float(np.sqrt(np.mean(distances**2))), max_mm=float(distances.max()),
                      survey_window_to_legacy_window={str(w): 5-w if side["reverse"] else w for w in range(1, 5)},
                      groups=[dict(rod=rod, survey_window=window, ray_count=count,
                                   survey_point_m=point.tolist(), model_point_m=model.tolist(),
                                   ray_intersection_rms_mm=float(error * 1000), residual_mm=float(distance))
                              for (rod, window), count, point, model, error, distance in zip(
                                  side["keys"], side["counts"], side["target"], predicted, side["errors"], distances)],
                      assumptions=["Центр газового объёма: z=0; его ось параллельна геодезической z",
                                   "Номера стержней сохранены; порядок окон выбран по измеренным уровням",
                                   "Пересечение семи лучей оценивает центр микросборки; отдельные зеркала не переюстируются"])
        frames[direction] = SurveyFrame(rotation, translation, report, side["rows"])
    return frames
