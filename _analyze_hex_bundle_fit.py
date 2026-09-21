import csv
import runpy

import numpy as np


m = runpy.run_path("из-к+_реальная.py")
equations = {}
sector = None
with open("уравнения_лучей_в_СК_идеальной_геометрии.txt", encoding="utf-8-sig", newline="") as handle:
    for row in list(csv.reader(handle, delimiter="\t"))[1:]:
        if row[0].strip():
            sector = int(row[0])
        layer, mirror = map(int, row[1].split("_"))
        if all(cell.strip() != "-" for cell in row[2:8]):
            point = np.array([float(cell.replace(",", ".")) for cell in row[2:5]]) / 1000.0
            direction = np.array([float(cell.replace(",", ".")) for cell in row[5:8]])
            if sector in (1, 3):
                point[:2] *= -1.0
                direction[:2] *= -1.0
            direction /= np.linalg.norm(direction)
            equations[(sector, layer, mirror)] = (point, direction)

for sector in range(1, 5):
    for layer in range(1, 5):
        data = m[f"BUNDLE_{sector}_{layer}_DATA"]
        centers = []
        distances = []
        for mirror, item in enumerate(data, start=1):
            applied = np.asarray(item.get("ideal_ray_applied_shift", (0.0, 0.0, 0.0)))
            center = np.asarray(item["center"]) - applied
            centers.append(center)
            if (sector, layer, mirror) in equations:
                point, direction = equations[(sector, layer, mirror)]
                delta = center - point
                distances.append(np.linalg.norm(delta - direction * np.dot(delta, direction)))
        pair_distances = [
            np.linalg.norm(centers[i] - centers[j])
            for i in range(7)
            for j in range(i + 1, 7)
        ]
        print(
            f"BUNDLE_{sector}_{layer}",
            "lines", len(distances),
            "line_mm_mean/max", round(float(np.mean(distances)) * 1000, 3), round(float(np.max(distances)) * 1000, 3),
            "pairs_mm_min", round(float(np.min(pair_distances)) * 1000, 3),
            "outer_r_mm", [round(float(np.linalg.norm(centers[i] - centers[6])) * 1000, 3) for i in range(6)],
        )
