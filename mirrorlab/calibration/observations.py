"""Immutable, component-aware observation ledger for reconstruction.

Only original survey workbooks contribute to the likelihood. Derived equations
and historical fits are inventoried, never silently counted a second time.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from .io import FILES, read_equations, read_lavsan, read_survey


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_lasertest_template(path):
    """Read the documented test_coords format; zero is otherwise a real number."""
    path = Path(path)
    result, window = [], None
    digest = file_digest(path)
    for line_number, text in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        match = re.match(r"Окно\s+(\d+)", text.strip(), re.I)
        if match:
            window = int(match[1])
            continue
        values = text.split()
        if window is None or len(values) != 4 or not values[0].isdigit():
            continue
        point = int(values[0])
        coordinates = [float(x.replace(",", ".")) / 1000 for x in values[1:]]
        placeholder = window in (2, 4) and point == 7 and coordinates == [0., 0., 0.]
        result.append(dict(id=f"lasertest:{window}:{point}", window=window, point=point,
                           source_file=path.name, source_sha256=digest, line=line_number,
                           raw_coordinates_m=coordinates,
                           coordinates_m=[None] * 3 if placeholder else coordinates,
                           missing_mask=[placeholder] * 3, source_type="computational_template",
                           transformations=["known_window_2_4_point_7_zero_placeholder"] if placeholder else [],
                           excluded_reason="Not an independent measurement of physical mirror centres"))
    return result


def observation_registry(root, direction):
    root = Path(root)
    path = root / FILES[direction]
    rows, warnings = read_survey(path)
    digest = file_digest(path)
    records = []
    for row in rows:
        identifier = f"{direction}:survey:{row.id}"
        match = re.search(r" / (.+)!A(\d+):H\d+$", row.source)
        records.append(dict(
            id=identifier, old_id=row.id, rod=row.rod, window=row.window, ray=row.ray,
            independent_source_id=identifier, direction=direction,
            source_file=path.name, source_sha256=digest,
            sheet=match[1] if match else None, row=int(match[2]) if match else None,
            cells=[[f"{c}{match[2]}" for c in cols] for cols in ("BCD", "FGH")] if match else None,
            units="m", original_units="mm", coordinate_frame="survey",
            screens_m=row.screens, missing_mask=[[x is None for x in p] for p in row.screens],
            source_type="primary", transformations=["numeric_XML_mm_to_m"],
            complete_points=len(row.points), exclusion_reason=None,
            metrology_sigma_m=None,
        ))
    counts = dict(total=len(records), complete=sum(r["complete_points"] == 2 for r in records),
                  single_screen=sum(r["complete_points"] == 1 for r in records),
                  missing=sum(r["complete_points"] == 0 for r in records))
    for record in records:
        for warning in warnings:
            match = re.search(r"!([A-Z]+)(\d+):", warning)
            if match and int(match[2]) == record["row"]:
                record["transformations"].append(dict(operation="read_raw_numeric_XML_despite_date_format",
                                                       cell=match[1] + match[2], diagnostic=warning))
    conflicts, inventory = [], [dict(file=path.name, sha256=digest, type="primary", used=True)]
    if direction == "west_to_east":
        equation_path = root / "из-к+_уравнения_лучей.txt"
        if equation_path.is_file():
            equations = read_equations(equation_path)
            inventory.append(dict(file=equation_path.name, sha256=file_digest(equation_path),
                                  type="derived", used=False, reason="correlated_copy_of_survey"))
            for row in records:
                equation = equations.get(tuple(int(x) for x in row["old_id"].split("_")))
                if row["complete_points"] < 2 and equation and all(v is not None for v in equation):
                    conflicts.append(dict(observation_id=row["id"], old_id=row["old_id"],
                                          type="derived_direction_without_complete_primary_points",
                                          derived_source=equation_path.name, coefficients_mm=equation,
                                          decision="primary_only", alternative="Explicit TXT trust scenario required; not used in likelihood"))
    lavsan = dict(status="not_available", points=[], rods=[])
    lavsan_path = root / "лавсан_из+к-.txt"
    if direction == "east_to_west" and lavsan_path.is_file():
        points = read_lavsan(lavsan_path)
        lavsan = dict(status="excluded_pending_geometric_correspondence", points=[], rods=[])
        for key, point in points.items():
            lavsan["points"].append(dict(old_id="_".join(map(str, key)), point_m=point,
                                         source_sha256=file_digest(lavsan_path), coordinate_frame="lavsan"))
        for rod in range(1, 5):
            levels = {w: float(np.mean([p[2] for k, p in points.items() if k[:2] == (rod, w)])) for w in range(1, 5)}
            repeats = [[a, b] for a in levels for b in levels if a < b and abs(levels[a] - levels[b]) < .01]
            lavsan["rods"].append(dict(rod=rod, levels_m=levels, repeated_levels=repeats,
                                      status="excluded_inconsistent_levels" if repeats else "not_used_unverified_correspondence"))
        inventory.append(dict(file=lavsan_path.name, sha256=file_digest(lavsan_path), type="primary_other_frame", used=False,
                              reason="correspondence_and_registration_not_independently_validated"))
    historical = ("Восстановленные_микросборки.txt", "Текущие_положения_микросборок.txt", "уравнения_лучей_в_СК_идеальной_геометрии.txt")
    for name in historical:
        if (root / name).is_file():
            inventory.append(dict(file=name, sha256=file_digest(root / name), type="derived", used=False,
                                  reason="historical_fit_not_manufacturing_measurement"))
    laser_path = root.parent / "lasertest" / "test_coords"
    laser = read_lasertest_template(laser_path) if laser_path.is_file() else []
    return dict(direction=direction, units="m", coordinate_frame="survey", records=records,
                counts=counts, warnings=warnings, conflicts=conflicts, sources=inventory,
                lavsan=lavsan, lasertest=laser,
                uncertainty_note="Two-point lines have zero algebraic residual, not zero measurement uncertainty")


def exclude_sources(registry, independent_ids):
    """A fold filter is applied to source identities, including derived copies."""
    excluded = set(independent_ids)
    return [r for r in registry["records"] if r["independent_source_id"] not in excluded
            and not (set(r.get("derived_from", [])) & excluded)]
