"""Reproducible static calibration, with provenance and explicit missing data."""
from pathlib import Path

import numpy as np

from .fitting import fit_line, fit_rigid_to_lines
from .io import FILES, read_equations, read_lavsan, read_survey
from .registration import register_west_to_east

ROOT = Path(__file__).resolve().parents[2]


def _ray_record(measurement):
    record = measurement.to_dict()
    points = measurement.points
    record["status"] = ("complete" if len(points) == 2 else
                        "single_screen" if points else "missing")
    record["coefficients_mm"] = ([float(v * 1000) for v in points[0]] if points else [None] * 3)
    record["coefficients_mm"] += ([float(v * 1000) for v in points[1] - points[0]]
                                  if len(points) == 2 else [None] * 3)
    if len(points) == 2:
        record.update(fit_line(points).to_dict())
    else:
        record.update(point_m=points[0].tolist() if points else None,
                      direction=None, rms_mm=None, point_count=len(points))
    return record


def _compare_saved(rays, path):
    saved = read_equations(path)
    matched, differences = 0, []
    for record in rays:
        key = record["rod"], record["window"], record["ray"]
        expected = saved.get(key)
        if expected is None:
            differences.append({"id": record["id"], "reason": "В сохранённом файле нет строки"})
            continue
        actual = record["coefficients_mm"]
        matches = [a is None and b is None or a is not None and b is not None and abs(a - b) <= .005001
                   for a, b in zip(actual, expected)]
        if all(matches):
            matched += 1
        else:
            differences.append({"id": record["id"], "survey_mm": actual, "saved_mm": expected,
                                "differing_columns": [i for i, value in enumerate(matches) if not value]})
    return {"source": path.name, "compared": len(rays), "matched": matched,
            "tolerance_mm": .005001, "differences": differences,
            "interpretation": "Проверка p0=P1, a=P2−P1; совпадение не доказывает оптимизацию по лавсану."}


def _register_lavsan(rays, path):
    points = read_lavsan(path)
    by_key = {(r["rod"], r["window"], r["ray"]): r for r in rays}
    rods, refined, warnings = [], [], []
    for rod in range(1, 5):
        keys = sorted(key for key in points if key[0] == rod and key in by_key
                      and by_key[key]["direction"] is not None)
        if len(keys) < 4:
            rods.append({"rod": rod, "success": False, "point_count": len(keys),
                         "message": "Недостаточно полных соответствий"})
            continue
        fit = fit_rigid_to_lines([points[key] for key in keys],
                                 [by_key[key]["point_m"] for key in keys],
                                 [by_key[key]["direction"] for key in keys])
        fit.update(rod=rod, point_count=len(keys), ids=[by_key[key]["id"] for key in keys])
        # Windows represent different longitudinal levels. Coincident levels
        # in the supplied table are retained, but cannot silently drive optics.
        mean_z = {window: np.mean([p[2] for k, p in points.items() if k[:2] == (rod, window)])
                  for window in range(1, 5)}
        fit["accepted"] = fit["success"]
        for first in range(1, 5):
            for second in range(first + 1, 5):
                if abs(mean_z[first] - mean_z[second]) < .01:
                    fit["accepted"] = False
                    warnings.append(f"Стержень {rod}: окна {first} и {second} имеют практически одинаковый z лавсана; результат не принят для уточнения лучей.")
        rods.append(fit)
        if not fit["accepted"]:
            continue
        rotation, translation = np.array(fit["rotation"]), np.array(fit["translation_m"])
        for key in sorted(points):
            if key[0] != rod or key not in by_key:
                continue
            ray = by_key[key]
            geodetic = [p for p in ray["screens_m"] if p and all(v is not None for v in p)]
            moved = rotation @ points[key] + translation
            if not geodetic:
                continue
            result = fit_line(geodetic + [moved])
            refined.append({"id": ray["id"], **result.to_dict(), "lavsan_point_m": moved.tolist(),
                            "status": "refined" if len(geodetic) == 2 else "recovered_with_lavsan"})
    return {"status": "fitted" if all(r.get("accepted", False) for r in rods) else "partial",
            "method": "rigid_point_to_line_least_squares",
            "assumption": "Соответствие стержень/окно/точка = стержень/микросборка/луч; отдельный жёсткий перенос и поворот каждого стержня, масштаб 1.",
            "provenance": "Независимо восстановленный алгоритм; предоставленный архив Optimis_code содержит оптимизацию быстродействия трассировки, но не привязку лавсана.",
            "source": path.name, "available_points": len(points), "rods": rods,
            "refined_rays": refined, "warnings": warnings,
            "limitations": "Невязка рассчитана по обучающим точкам и не является независимой оценкой точности. Соответствия индексов и принадлежность к половине требуют подтверждения."}


def analyze_measurements(direction="west_to_east", *, fit_lavsan=False, data_dir=None):
    if direction not in FILES:
        raise ValueError(f"Неизвестное направление: {direction}")
    data_dir = Path(data_dir) if data_dir else ROOT
    path = data_dir / FILES[direction]
    measurements, warnings = read_survey(path)
    rays = [_ray_record(measurement) for measurement in measurements]
    warnings.append("Координаты сохранены в исходной геодезической системе: знак z в таблицах противоположен названию половины. Привязка к сцене должна задаваться отдельно.")
    counts = {name: sum(r["status"] == name for r in rays) for name in ("complete", "single_screen", "missing")}
    counts["total"] = len(rays)
    result = {"direction": direction, "coordinate_frame": "survey", "units": "m",
              "method": "two_screen_lines", "counts": counts, "rays": rays,
              "warnings": warnings, "sources": [path.name],
              "saved_equations_comparison": None, "lavsan": {"status": "not_requested"},
              "scene_registration": None}
    saved = data_dir / "из-к+_уравнения_лучей.txt"
    if direction == "west_to_east" and saved.is_file():
        result["saved_equations_comparison"] = _compare_saved(rays, saved)
        result["sources"].append(saved.name)
        if (data_dir / "уравнения_лучей_в_СК_идеальной_геометрии.txt").is_file():
            from ..configurations import configuration_ray_references, load_configuration
            config = load_configuration("west_to_east", "real")
            result["scene_registration"] = register_west_to_east(
                rays, data_dir, references=configuration_ray_references(config),
                scene_z_offset=config.metadata["provenance"]["z_offset_m"])
            result["warnings"].extend(result["scene_registration"]["warnings"])
            result["sources"].append("уравнения_лучей_в_СК_идеальной_геометрии.txt")
    if fit_lavsan:
        if direction != "east_to_west":
            raise ValueError("Файл лавсан_из+к-.txt относится к направлению EAST → WEST; для WEST → EAST данные лавсана отсутствуют.")
        lavsan = data_dir / "лавсан_из+к-.txt"
        result["lavsan"] = _register_lavsan(rays, lavsan)
        result["sources"].append(lavsan.name)
        result["warnings"].extend(result["lavsan"]["warnings"])
    return result
