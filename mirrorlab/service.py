"""Application operations, independent of Electron or a future network server."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .configurations import DIRECTIONS, load_configuration
from .coordinates import survey_frames
from .engine import RayTracer, to_numpy
from .plotting import PlotlyRenderer, measurement_figure

ROOT = Path(__file__).resolve().parents[1]


def selected_directions(direction):
    return ("east_to_west", "west_to_east") if direction == "both" else (direction,)


def additional_screens(direction, report=None):
    source = "лавсан_из+к-.txt" if direction == "east_to_west" else None
    available = bool(source and (ROOT / source).is_file())
    status = "available" if available else "not_available"
    used, rejected = [], []
    count = 0
    if report is not None and available:
        lavsan = report["lavsan"]
        used = [r["rod"] for r in lavsan.get("rods", []) if r.get("accepted")]
        rejected = [r["rod"] for r in lavsan.get("rods", []) if not r.get("accepted")]
        count = len(lavsan.get("refined_rays", []))
        status = ("used_partial" if rejected else "used") if count else (
            "not_requested" if lavsan["status"] == "not_requested" else "rejected")
    return dict(direction=direction, injection_side=DIRECTIONS[direction]["injection_side"],
                available=available, source=source, status=status, refined_ray_count=count,
                accepted_rods=used, rejected_rods=rejected, applied_to_mirror_geometry=False)


@dataclass(frozen=True)
class TraceRequest:
    direction: str = "both"
    kind: str = "ideal"
    collision_mode: str = "expected"
    ray_count: int = 217
    max_interactions: int = 48

    def __post_init__(self):
        if self.direction not in {*DIRECTIONS, "both"}:
            raise ValueError("Неизвестное направление лучей.")
        if self.kind not in {"ideal", "real"}:
            raise ValueError("Выберите идеальную или реальную конфигурацию.")
        if self.collision_mode not in {"exhaustive", "expected"}:
            raise ValueError("Неизвестный режим поиска пересечений.")
        for name, maximum in (("ray_count", 50000), ("max_interactions", 100)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name}: требуется целое число от 1 до {maximum}.")


def catalog():
    return dict(
        directions=[dict(id="both", label="Обе стороны · WEST и EAST")] + [dict(id=key, label=f"{value['source_side']} → {value['injection_side']}", **value)
                    for key, value in DIRECTIONS.items()],
        additional_screens=[additional_screens(key) for key in DIRECTIONS],
        configurations=["ideal", "real"], collision_modes=["exhaustive", "expected"],
        defaults=asdict(TraceRequest()), units="m", separator_z_m=0, coordinate_frame="survey",
        modules={"static": "ready", "distributions": "stub", "dynamic": "stub"},
    )


def _trace_one(request):
    started = time.perf_counter()
    configuration = load_configuration(request.direction, request.kind)
    rays = configuration.emit(request.ray_count)
    tracer = RayTracer(scene=configuration.scene, collision_mode=request.collision_mode,
                       max_interactions=request.max_interactions, max_time_s=configuration.integration_time_s,
                       **configuration.tracer_options)
    result = tracer.trace(rays)
    frame = survey_frames()[request.direction]
    figure, displayed = PlotlyRenderer().render(configuration, result, frame=frame)
    compiled = tracer.compiled_scene
    segment_count = sum(len(block["x0"]) for block in result.segments)
    warnings = [f"{configuration.metadata['injection_side']}: привязка к геодезии по лучам — RMS {frame.report['rms_mm']:.1f} мм, максимум {frame.report['max_mm']:.1f} мм по центрам микросборок. Это оценка положения системы, без переюстировки зеркал.",
                "Исходные геодезические точки можно включить в легенде сцены."]
    if displayed < segment_count:
        warnings.append(f"На графике показано {displayed} из {segment_count} сегментов; расчёт выполнен для всех лучей.")
    if request.ray_count < 721:
        warnings.append("Предварительная сетка: малые зеркала могут оказаться между лучами. Увеличьте число лучей для проверки сходимости.")
    if request.kind == "real":
        warnings.append("Геометрия зеркал загружена из сохранённой реальной конфигурации. Измерения обработаны автоматически; их отчёт не подменяет положения зеркал без привязки систем координат.")
    unfinished = result.diagnostics.get("unfinished_count", 0)
    if unfinished:
        warnings.append(f"Лимит взаимодействий достигнут для {unfinished} лучей. Увеличьте лимит.")
    return dict(
        figure=figure, request=asdict(request), metadata={**configuration.metadata, "coordinate_frame": "survey", "separator_z_m": 0,
                                                        "frame_registration": frame.metadata()}, warnings=warnings,
        summary=dict(ray_count=rays.n_rays, surface_count=len(compiled.surfaces), segment_count=segment_count,
                     displayed_segments=displayed, detector_hits=sum(len(b["position"]) for b in result.detector_hits),
                     source_power_w=float(np.sum(to_numpy(rays.power))),
                     detector_power_w=result.detector_power_summary(),
                     integration_time_ns=configuration.integration_time_s * 1e9,
                     elapsed_s=round(time.perf_counter() - started, 3),
                     intersection_tests=compiled.intersection_tests, possible_tests=compiled.possible_tests,
                     unfinished_count=unfinished),
    )


def trace(payload):
    request = TraceRequest(**payload)
    started = time.perf_counter()
    results = [_trace_one(TraceRequest(**{**asdict(request), "direction": direction}))
               for direction in selected_directions(request.direction)]
    if len(results) == 1:
        result = results[0]
    else:
        data = []
        for index, item in enumerate(results):
            side = item["metadata"]["injection_side"]
            for layer in item["figure"]["data"]:
                if layer.get("name", "").startswith("Граница"):
                    if index == 0:
                        data.append(layer)
                    continue
                layer["name"] = (f"Лазер {item['metadata']['source_side']} → ввод {side}"
                                 if layer.get("name") == "Лазер" else f"{side} · {layer.get('name', 'Интенсивность')}")
                layer["legendgroup"] = f"{side}:survey" if layer.get("meta", {}).get("kind") == "survey" else side
                if layer.get("marker", {}).get("showscale"):
                    layer["marker"]["colorbar"].update(len=.45, y=.75 - index * .5)
                    layer["marker"]["colorbar"]["title"] = dict(text=f"{side}: log₁₀ I, Вт/м²")
                data.append(layer)
        summary = {key: sum(item["summary"][key] for item in results) for key in (
            "ray_count", "surface_count", "segment_count", "displayed_segments", "detector_hits",
            "source_power_w", "intersection_tests", "possible_tests", "unfinished_count")}
        summary["detector_power_w"] = {item["metadata"]["injection_side"]: item["summary"]["detector_power_w"]
                                       for item in results}
        result = dict(figure=dict(data=data, layout=results[0]["figure"]["layout"]),
                      request=asdict(request), summary=summary,
                      metadata=dict(direction="both", separator_z_m=0, units="m",
                                    coordinate_frame="survey", trace_strategy="independent_sources"),
                      warnings=list(dict.fromkeys(warning for item in results for warning in item["warnings"])))
    result["sides"] = [dict(direction=item["request"]["direction"], metadata=item["metadata"],
                            summary=item["summary"]) for item in results]
    if request.kind == "real":
        result["measurements"] = calibrate(dict(direction=request.direction))
    result["summary"]["elapsed_s"] = round(time.perf_counter() - started, 3)
    return result


def calibrate(payload):
    from .calibration import analyze_measurements

    if set(payload) - {"direction", "use_lavsan"}:
        raise ValueError("Неизвестные параметры расчёта измерений.")
    direction = payload.get("direction", "both")
    use_lavsan = payload.get("use_lavsan")
    if direction not in {*DIRECTIONS, "both"} or (use_lavsan is not None and type(use_lavsan) is not bool):
        raise ValueError("Некорректные параметры расчёта измерений.")
    reports = []
    for key in selected_directions(direction):
        available = additional_screens(key)["available"]
        fit_lavsan = available if use_lavsan is None else use_lavsan
        if direction == "both":
            fit_lavsan = fit_lavsan and available
        report = analyze_measurements(key, fit_lavsan=fit_lavsan)
        report["static_frame_registration"] = survey_frames()[key].metadata()
        report["additional_screens"] = additional_screens(key, report)
        report["figure"] = measurement_figure(report)
        reports.append(report)
    return dict(direction="both", reports=reports) if direction == "both" else reports[0]


def dispatch(method, payload):
    if not isinstance(payload, dict):
        raise ValueError("Параметры должны быть JSON-объектом.")
    if method == "catalog":
        return catalog()
    if method == "trace":
        return trace(payload)
    if method == "calibrate":
        return calibrate(payload)
    raise ValueError(f"Неизвестная операция: {method}")
