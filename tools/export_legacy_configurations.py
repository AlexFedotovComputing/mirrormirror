"""Export the four historical scene builders as portable geometry data.

Run from the repository root to reproduce snapshots with documented corrections.
The historical scripts read data from the root; they are never modified here.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from enum import Enum
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "legacy")]


def encode(value):
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value):
        return {"type": type(value).__name__, "parameters": {
            field.name: encode(getattr(value, field.name))
            for field in dataclasses.fields(value) if field.init
        }}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    return value


def export(direction, kind, filename):
    path = ROOT / filename
    if not path.exists():
        path = ROOT / "legacy" / filename
    source = path.read_text(encoding="utf-8-sig")
    namespace = {"__name__": "legacy_configuration_export", "__file__": str(ROOT / path.name)}
    exec(compile(source, str(path), "exec"), namespace)
    scene = namespace["build_initial_scene"]()
    gaussian = namespace["build_initial_source"]()
    z_offset = -2 * namespace["REFLECTION_PLANE_Z_M"] if direction == "west_to_east" else 0.0
    elements = []
    coating_surfaces, prism_caps = [], []
    for element in scene.elements:
        surfaces = []
        for surface in element.build_surfaces():
            record = encode(surface)
            # This legacy builder faces the coating away from the +z incident
            # beam after reconstructing and reflecting the bundle normals.
            if direction == "west_to_east" and kind == "ideal" and surface.name.startswith("BUNDLE_"):
                optics = record["parameters"]["optics"]["parameters"]
                optics.update(reflect_from_minus_side=True, reflect_from_plus_side=False)
                coating_surfaces.append(surface.name)
            # The archived prism builder reverses the top normal without
            # compensating its local v coordinates. Preserve the archive,
            # but export a closed prism, matching the corrected engine.
            if type(element).__name__ == "TriangularPrism" and surface.name.endswith(":top"):
                record["parameters"]["triangle_vertices"] = [[x, -y] for x, y in element.vertices_2d]
                prism_caps.append(surface.name)
            for key in ("center", "vertex"):
                if key in record["parameters"]:
                    record["parameters"][key][2] += z_offset
            surfaces.append(record)
        elements.append({"name": getattr(element, "name", type(element).__name__), "surfaces": surfaces})
    gaussian.waist_position = list(gaussian.waist_position)
    gaussian.waist_position[2] += z_offset
    options = {}
    for name in ("inner_radius", "outer_radius", "z_min", "z_max"):
        symbol = f"BUNDLE_RAY_{name.upper()}_M"
        if name.endswith("radius"):
            symbol = f"BUNDLE_RAY_{name.upper().replace('_RADIUS', '_CYLINDER_RADIUS')}_M"
        value = namespace.get(symbol)
        if value is not None:
            options[f"bundle_clip_{name}_m"] = value + (z_offset if name.startswith("z_") else 0)
    record = {
        "direction": direction, "kind": kind, "source": encode(gaussian),
        "elements": elements, "integration_time_s": namespace["INTEGRATION_TIME_S"],
        "max_interactions": namespace["MAX_SECONDARY_RAY_GENERATIONS"],
        "tracer_options": options,
        "provenance": {
            "legacy_file": filename, "legacy_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "z_offset_m": z_offset,
            "inputs_sha256": {
                value.name: hashlib.sha256(value.read_bytes()).hexdigest()
                for name, value in namespace.items()
                if name.endswith("_PATH") and isinstance(value, Path) and value.is_file()
            },
            "description": "Built legacy geometry snapshot; survey fits are applied separately.",
        },
    }
    corrections = []
    if coating_surfaces:
        corrections.append({
            "description": "Face the ideal bundle coating toward the incident +z beam: reflect from the normal's minus side.",
            "surfaces": coating_surfaces,
        })
    if prism_caps and kind == "ideal":
        corrections.append({
            "description": "Negate the prism top cap's local v coordinates to compensate its reversed normal and close the solid.",
            "surfaces": prism_caps,
        })
    if corrections:
        record["provenance"]["corrections"] = corrections
    if kind == "real":
        # Only the microassemblies differ. Everything else, including source,
        # screens, coatings and trace limits, comes from the ideal snapshot.
        record = {"direction": direction, "kind": kind,
                  "base_configuration": f"{direction}_ideal.json",
                  "elements": [e for e in elements if e["name"].startswith("BUNDLE_")],
                  "provenance": record["provenance"]}
        record["provenance"]["description"] = "Real microassemblies; all common optics inherited from the ideal configuration."
    directory = ROOT / "data" / "static"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{direction}_{kind}.json"
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{target.name}: {len(record['elements'])} stored elements, base={record.get('base_configuration', 'none')}")


if __name__ == "__main__":
    for direction, prefix in (("west_to_east", "из-к+"), ("east_to_west", "из+к-")):
        for kind, label in (("ideal", "идеальная"), ("real", "реальная")):
            filename = ("legacy/variants/из+к-_идеальная.txt" if (direction, kind) == ("east_to_west", "ideal")
                        else f"{prefix}_{label}.py")
            export(direction, kind, filename)
