"""Application operations, independent of Electron or a future network server."""
from __future__ import annotations

import time
import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .configurations import DIRECTIONS, load_configuration
from .directions import ROUTES, SIDE_DESIGNATIONS, ENABLED_DIRECTIONS, DISABLED_DIRECTION_MESSAGE
from .coordinates import survey_frames
from .engine import RayTracer, to_numpy
from .plotting import PlotlyRenderer, measurement_figure

ROOT = Path(__file__).resolve().parents[1]


def _ring_note(metadata):
    report = metadata.get('ring_alignment')
    if not report:
        return None
    shifts = report.get('footprint_recentering', [])
    detail = (f" Пятна в стержнях сдвинуты настройкой MS/MC не более чем на {max(s['transverse_shift_m'] for s in shifts)*1000:.3f} мм."
              if shifts else '')
    return ('Кольцо WEST отражено; перископ направлен вниз. Положения микрозеркал и направления падающих лучей сохранены.'
            + detail + ' Распределение мощности по микрозеркалам может измениться.')


def selected_directions(direction):
    if direction == "both":
        return ENABLED_DIRECTIONS
    if direction not in DIRECTIONS:
        raise ValueError("Неизвестное направление лучей.")
    if direction not in ENABLED_DIRECTIONS:
        raise ValueError(DISABLED_DIRECTION_MESSAGE)
    return (direction,)


def additional_screens(direction, report=None):
    source = ROUTES[direction].additional_screens_file
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
    direction: str = ENABLED_DIRECTIONS[0] if len(ENABLED_DIRECTIONS) == 1 else "both"
    kind: str = "ideal"
    collision_mode: str = "expected"
    ray_count: int = 1000
    cpu_workers: int = 6
    max_interactions: int = 48

    def __post_init__(self):
        selected_directions(self.direction)
        if self.kind not in {"ideal", "real"}:
            raise ValueError("Выберите идеальную или реальную конфигурацию.")
        if self.collision_mode not in {"exhaustive", "expected"}:
            raise ValueError("Неизвестный режим поиска пересечений.")
        for name in ('ray_count', 'cpu_workers'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name}: требуется положительное целое число.')
        for name, maximum in (("max_interactions", 100),):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name}: требуется целое число от 1 до {maximum}.")


def catalog():
    return dict(
        directions=([dict(id="both", label="Обе стороны · WEST и EAST")] if len(ENABLED_DIRECTIONS) > 1 else []) +
            [dict(id=key, label=f"{DIRECTIONS[key]['source_side']} → {DIRECTIONS[key]['injection_side']}", **DIRECTIONS[key])
             for key in ENABLED_DIRECTIONS],
        disabled_directions=[dict(id=key, reason=DISABLED_DIRECTION_MESSAGE) for key in DIRECTIONS if key not in ENABLED_DIRECTIONS],
        additional_screens=[additional_screens(key) for key in ENABLED_DIRECTIONS],
        side_designations=SIDE_DESIGNATIONS,
        configurations=["ideal", "real"], collision_modes=["exhaustive", "expected"],
        defaults=asdict(TraceRequest()), units="m", separator_z_m=0, coordinate_frame="survey",
        modules={"static": "ready", "distributions": "ready", "dynamic": "stub"},
    )


def _dataset_frame(dataset):
    """A loaded document never re-fits its coordinates against local files."""
    from .coordinates import SurveyFrame
    from .calibration.models import Measurement
    frame = dataset.get("display_frame")
    if frame is None:
        return None
    registry = dataset.get("observations", [])
    rows = registry.get("records", []) if isinstance(registry, dict) else registry
    measurements = []
    for row in rows:
        if row.get("screens_m") is not None:
            parts = [int(x) for x in row.get("old_id", "1_1_1").split("_")]
            measurements.append(Measurement(*parts, row["screens_m"], row.get("source_file", "embedded")))
    return SurveyFrame(np.asarray(frame["rotation"]), np.asarray(frame["translation_m"]),
                       {k: v for k, v in frame.items() if k not in {
                           "rotation", "translation_m", "coordinate_frame", "calculation_frame"}}, tuple(measurements))


def _trace_one(request, dataset=None):
    started = time.perf_counter()
    if dataset is None:
        configuration = load_configuration(request.direction, request.kind)
    else:
        from .state import configuration_from_dataset
        configuration = configuration_from_dataset(dataset)
    rays = configuration.emit(request.ray_count)
    from .engine.parallel import trace_cpu
    result = trace_cpu(configuration.scene, rays, cpu_workers=request.cpu_workers, collision_mode=request.collision_mode,
                       max_interactions=request.max_interactions, max_time_s=configuration.integration_time_s,
                       **configuration.tracer_options)
    frame = survey_frames()[request.direction] if dataset is None else _dataset_frame(dataset)
    figure, displayed = PlotlyRenderer().render(configuration, result, frame=frame)
    from .visualization import build_visualization
    visualization = build_visualization(configuration, rays, result, frame)
    segment_count = sum(len(block["x0"]) for block in result.segments)
    warnings = ["Исходные геодезические точки можно включить в легенде сцены."]
    note = _ring_note(configuration.metadata)
    if note:
        warnings.append(note)
    if dataset is None:
        warnings.insert(0, f"{configuration.metadata['injection_side']}: привязка к геодезии по лучам — RMS {frame.report['rms_mm']:.1f} мм, максимум {frame.report['max_mm']:.1f} мм по центрам микросборок. Это оценка положения системы, без переюстировки зеркал.")
    else:
        warnings.insert(0, f"Трассируется выбранная физическая конфигурация: {dataset['status']}. Оптика и тела взяты из её сохранённого состояния.")
    if displayed < segment_count:
        warnings.append(f"На графике показано {displayed} из {segment_count} сегментов; расчёт выполнен для всех лучей.")
    if request.ray_count < 721:
        warnings.append("Предварительная сетка: малые зеркала могут оказаться между лучами. Увеличьте число лучей для проверки сходимости.")
    if request.kind == "real" and dataset is None:
        warnings.append("Общая оптика взята из идеальной конфигурации; реальные положения сохранены только у микросборок. Измерения обработаны автоматически; их отчёт не подменяет положения зеркал без привязки систем координат.")
    unfinished = result.diagnostics.get("unfinished_count", 0)
    if unfinished:
        warnings.append(f"Лимит взаимодействий достигнут для {unfinished} лучей. Увеличьте лимит.")
    return dict(
        figure=figure, visualization=visualization, request=asdict(request), metadata={**configuration.metadata, "coordinate_frame": "survey" if frame else "legacy", "separator_z_m": 0,
                                                        "frame_registration": frame.metadata() if frame else None}, warnings=warnings,
        summary=dict(ray_count=rays.n_rays, surface_count=len(result.surface_names), segment_count=segment_count,
                     cpu_workers=result.diagnostics['cpu_workers'],
                     displayed_segments=displayed, detector_hits=sum(len(b["position"]) for b in result.detector_hits),
                     source_power_w=float(np.sum(to_numpy(rays.power))),
                     detector_power_w=result.detector_power_summary(),
                     integration_time_ns=configuration.integration_time_s * 1e9,
                     elapsed_s=round(time.perf_counter() - started, 3),
                     intersection_tests=result.diagnostics['intersection_tests'], possible_tests=result.diagnostics['possible_tests'],
                     unfinished_count=unfinished),
    )


def static_manual_document(direction):
    """Load the reviewed manual mirrors with their supplied optics, without fitting."""
    from .manual_schema import BUNDLE, import_manual, manual_starting_geometry
    from .state import new_document
    datasets = {}
    for d in selected_directions(direction):
        filename = 'west_manual_corrected_survey.json' if d == 'east_to_west' else 'east_recovered_centers.json'
        imported = import_manual(BUNDLE / filename)['datasets'][d]
        datasets[d] = manual_starting_geometry(imported)
    from .ring_migration import upgrade_ring_document
    return upgrade_ring_document(new_document(datasets))


def trace_static(payload):
    """Trace the selected built-in preset independently of any accepted candidate."""
    clean = {k: v for k, v in payload.items() if k not in {
        'active_configuration', 'candidate_configuration', 'configuration_version'}}
    request = TraceRequest(**clean)
    if request.kind == 'real':
        clean['active_configuration'] = static_manual_document(request.direction)
    result = trace(clean)
    source = 'manual' if request.kind == 'real' else 'ideal'
    result['metadata']['static_configuration'] = source
    for side in result['sides']:
        side['metadata']['static_configuration'] = source
    return result


def trace(payload):
    payload = dict(payload)
    document = payload.pop("active_configuration", None)
    payload.pop("configuration_version", None)
    if document is not None:
        from .state import validate_document
        validate_document(document)
    request = TraceRequest(**payload)
    started = time.perf_counter()
    directions = selected_directions(request.direction)
    if document is not None and any(d not in document["datasets"] for d in directions):
        raise ValueError("В активной конфигурации нет выбранной стороны. Выберите сохранённую сторону или явно вернитесь к исторической конфигурации.")
    if document is not None and len(directions) > 1 and any(not document["datasets"][d].get("display_frame") for d in directions):
        raise ValueError("Для общей сцены обеих сторон нужны сохранённые преобразования в общую геодезическую систему. Трассируйте стороны отдельно.")
    results = [_trace_one(TraceRequest(**{**asdict(request), "direction": direction}),
                          document["datasets"][direction] if document else None) for direction in directions]
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
                layer["legendgroup"] = (f"{side}:{layer['legendgroup']}" if layer.get("meta", {}).get("kind") == "reconstructed_ray"
                                        else f"{side}:survey" if layer.get("meta", {}).get("kind") == "survey" else side)
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
    result["visualization"] = dict(sides=[item['visualization'] for item in results])
    result["sides"] = [dict(direction=item["request"]["direction"], metadata=item["metadata"],
                            summary=item["summary"]) for item in results]
    if request.kind == "real" and document is None:
        result["measurements"] = calibrate(dict(direction=request.direction))
    from .calibration.trajectory_matching import dataset_comparison_report
    result["comparison_reports"] = [dataset_comparison_report(document["datasets"][d]) if document else
        dict(direction=d, status="historical", diagnostics=dict(
            geometric_fit=dict(status="not_evaluated", reason="reconstruction_required"),
            trajectory_matching=dict(status="not_evaluated", reason="reconstruction_required")))
        for d in directions]
    result["summary"]["elapsed_s"] = round(time.perf_counter() - started, 3)
    result["summary"]["cpu_workers"] = max(item['summary']['cpu_workers'] for item in results)
    return result


def calibrate(payload):
    from .calibration import analyze_measurements

    payload = {k: v for k, v in payload.items() if k not in {"active_configuration", "configuration_version"}}
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


def _historical_document(payload):
    from .legacy_import import historical_dataset
    from .state import new_document
    direction = payload.get("direction", "both")
    if direction not in {*DIRECTIONS, "both"}:
        raise ValueError("Неизвестная сторона")
    datasets = {}
    for d in selected_directions(direction):
        config = load_configuration(d, payload.get("kind", "ideal"))
        dataset = historical_dataset(config)
        frame = survey_frames()[d]
        dataset["display_frame"] = {**frame.metadata(), "handedness_change": bool(np.linalg.det(frame.rotation) < 0)}
        datasets[d] = dataset
    return new_document(datasets, payload.get("configuration_version", 0))


def document_result(document):
    """Preview uses embedded state only; no Monte Carlo sampling required."""
    from .state import configuration_from_dataset, validate_document
    from .engine.tracer import TraceResult
    from .engine.bodies import MirrorBody
    from .plotting import _surface_traces
    from .calibration.trajectory_matching import dataset_comparison_report
    validate_document(document)
    from .ring_migration import upgrade_ring_document
    document = upgrade_ring_document(document)
    # Retain disabled datasets for a lossless save, but never build their scene,
    # diagnostics or rays. Their historical status must not block the active side.
    available = {d: value for d, value in document["datasets"].items() if d in ENABLED_DIRECTIONS}
    if not available:
        raise ValueError(DISABLED_DIRECTION_MESSAGE)
    figures, reports = [], []
    can_activate = True
    for direction, dataset in available.items():
        dataset.setdefault("diagnostics", {}).update(dataset_comparison_report(dataset)["diagnostics"])
        if dataset.get("optics_snapshot") is not None:
            config = configuration_from_dataset(dataset)
            empty = TraceResult(config.emit(1), [], [], [])
            figure, _ = PlotlyRenderer().render(config, empty, frame=_dataset_frame(dataset))
        else:
            can_activate = False
            data = []
            for m in dataset.get("mirrors", []):
                body = MirrorBody.from_dict(m)
                data.extend(_surface_traces(body.build_surfaces()[0]))
            figure = dict(data=data, layout=dict(template="plotly_dark", scene=dict(aspectmode="data"),
                title=dict(text="Частичный импорт: общая оптика отсутствует")))
        if dataset.get("manual_references"):
            figure["data"].extend(_manual_comparison_layers(dataset))
            from .calibration.manual_rays import current_analysis, scene_traces
            if dataset.get('manual_rays') and not current_analysis(dataset):
                dataset.pop('manual_rays')
            figure['data'].extend(scene_traces(dataset))
        for layer in figure["data"]:
            layer["name"] = f"{ROUTES[direction].injection_side} · {layer.get('name', '')}"
            meta = layer.setdefault("meta", {})
            meta.setdefault("layer", "optics")
            if meta["layer"] != "boundary":
                meta.update(direction=direction, side=ROUTES[direction].injection_side)
            if layer.get("meta", {}).get("mirror_id"):
                layer["meta"]["direction"] = direction
                for item in layer.get("customdata", []):
                    item["direction"] = direction
        figures.append(figure)
        reports.append(dataset)
    shared_frame = len(figures) == 1 or all(d.get("display_frame") for d in available.values())
    combined = (dict(data=[x for f in figures for x in f["data"]], layout=figures[0]["layout"])
                if shared_frame else figures[0])
    warnings = [f"{d}: {dataset['status']}" for d, dataset in available.items()]
    for dataset in available.values():
        note = _ring_note((dataset.get('optics_snapshot') or {}).get('metadata', {}))
        if note:
            warnings.append(note)
    if len(available) != len(document["datasets"]):
        warnings.append(DISABLED_DIRECTION_MESSAGE + " Данные отключённой стороны сохранены в документе.")
    if not shared_frame:
        warnings.append("Системы координат сторон не совмещены: показана только первая сторона. Для трассировки выберите одну сторону.")
    statuses = {d["status"] for d in available.values()}
    status = next(iter(statuses)) if len(statuses) == 1 else "partial"
    return dict(document=document, status=status, figure=combined, reports=reports,
                enabled_directions=list(available),
                scene_direction="both" if shared_frame and len(figures) > 1 else next(iter(available)),
                coordinate_frame=("survey" if all(d.get("display_frame") for d in available.values())
                                  else next(iter(available.values())).get("coordinate_frame", "legacy")),
                can_activate=can_activate and all(d["status"] != "infeasible" and
                    (not d.get("manual_references") or d["status"] == "validated")
                    for d in available.values()),
                warnings=warnings)


def _manual_comparison_layers(dataset):
    """Immutable manual references and primary points in their survey frame."""
    traces = []
    records = dataset.get("observations", {}).get("records", [])
    observations = {r["id"]: r for r in records}
    for mirror in dataset["mirrors"]:
        reference = mirror.get("manual_reference_survey", {})
        face = reference.get("survey_edge_points_as_transformed_m")
        points = face or reference.get("source_manual_points_survey_m")
        meta = dict(mirror_id=mirror["canonical_id"], layer="reconstructed_rays")
        if points:
            points = list(points) + ([points[0]] if face else [])
            traces.append(dict(type="scatter3d", mode="lines+markers",
                x=[p[0] for p in points], y=[p[1] for p in points], z=[p[2] for p in points],
                name=f"Ручной источник {mirror['source_manual_id']}", legendgroup="manual_references",
                line=dict(color="#d8a65d", width=3, dash="dash"), marker=dict(size=2),
                meta={**meta, "kind": "manual_reference"}, showlegend=False))
        row = observations.get(mirror.get("survey_observation_id"), {})
        points = [p for p in row.get("screens_m", []) if all(v is not None for v in p)]
        if points:
            traces.append(dict(type="scatter3d", mode="markers",
                x=[p[0] for p in points], y=[p[1] for p in points], z=[p[2] for p in points],
                name=f"Первичные точки {row.get('old_id', '')}", legendgroup="manual_primary",
                marker=dict(size=3, color="#78c8f0"),
                meta={**meta, "layer": "measurement_points", "kind": "manual_primary"}, showlegend=False))
    return traces


def optimize(payload, progress=None):
    from .state import new_document, dataset_from_config, validate_document, configuration_from_dataset
    from .calibration.reconstruction import reconstruct
    document = payload.get("candidate_configuration") or payload.get("active_configuration")
    selected = selected_directions(payload.get("direction", "both"))
    if document and any(d.get("manual_references") for key, d in document.get("datasets", {}).items() if key in selected):
        return refine_manual_configuration(payload, progress)
    if document is None and payload.get("kind", "real") == "ideal":
        raise ValueError("В исторической идеальной конфигурации все лучи заданы. "
                         "Для восстановления недостающих лучей выберите «Историческая реальная».")
    if document is not None:
        validate_document(document)
    direction = payload.get("direction", "both")
    if direction not in {*DIRECTIONS, "both"}:
        raise ValueError("Неизвестная сторона")
    datasets = {}
    for d in selected_directions(direction):
        if document is not None:
            if d not in document["datasets"]:
                raise ValueError("Выбранная сторона отсутствует в активной конфигурации")
            config = configuration_from_dataset(document["datasets"][d])
        else:
            config = load_configuration(d, payload.get("kind", "real"))
        def report(*args, **kwargs):
            if progress:
                value = args[0] if args else kwargs
                progress({"direction": d, **(value if isinstance(value, dict) else {"stage": str(value)})})
        report({"stage": "Проверка данных"})
        try:
            result = reconstruct(config, ROOT, options=payload.get("options"), progress=report)
            datasets[d] = dataset_from_config(config, result, result.get("display_frame"))
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            # Failure on one side is recorded without discarding another side.
            # Keep 112 explicit diagnostic poses; these are never accepted.
            from .legacy_import import historical_dataset
            result = historical_dataset(config)
            result.update(status="infeasible", model_kind="failed_reconstruction")
            for m in result["mirrors"]:
                m.update(status="infeasible", violations=["reconstruction_failed"])
            result["counts"] = dict(total=len(result["mirrors"]), measured=0, partial=0,
                                     predicted=len(result["mirrors"]), ambiguous=0, failed=len(result["mirrors"]))
            result["diagnostics"]["failure"] = dict(type=type(error).__name__, message=str(error))
            datasets[d] = result
            report({"stage": "Реконструкция стороны завершилась ошибкой", "error": str(error)})
    # Preserve an existing opposite side without blending its IDs or metrics.
    if document is not None:
        datasets = {**document["datasets"], **datasets}
    candidate = new_document(datasets, payload.get("configuration_version", 0))
    return document_result(candidate)


def refine_manual_configuration(payload, progress=None):
    """Refine a diagnostic document without changing the accepted state."""
    from .state import new_document, validate_document
    from .calibration.manual_refinement import refine_manual
    document = (payload.get("candidate_configuration") or payload.get("manual_configuration")
                or payload.get("document") or payload.get("active_configuration"))
    validate_document(document)
    direction = payload.get("direction", "both")
    if direction not in {*DIRECTIONS, "both"}:
        raise ValueError("Неизвестная сторона")
    selected = selected_directions(direction)
    chosen = [d for d in selected if d in document["datasets"]] if direction == "both" else selected
    if not chosen:
        raise ValueError(DISABLED_DIRECTION_MESSAGE)
    datasets = copy.deepcopy(document["datasets"])
    for d in chosen:
        if d not in datasets or not datasets[d].get("manual_references"):
            raise ValueError("Для выбранной стороны отсутствует ручная реконструкция")
        def report(value, direction=d):
            if progress:
                progress({"direction": direction, **value})
        datasets[d] = refine_manual(datasets[d], options=payload.get("options"), progress=report)
    return document_result(new_document(datasets, payload.get("configuration_version", 0)))


def add_manual_analysis_rays(payload, progress=None):
    from .state import new_document, validate_document
    from .manual_schema import import_manual, BUNDLE
    from .calibration.manual_rays import add_manual_rays
    document = payload.get('candidate_configuration') or payload.get('active_configuration')
    if document:
        validate_document(document)
    datasets = copy.deepcopy(document['datasets']) if document else {}
    direction = payload.get('direction', 'both')
    if direction not in {*DIRECTIONS, 'both'}:
        raise ValueError('Неизвестная сторона')
    for d in selected_directions(direction):
        existing = datasets.get(d, {})
        if not existing.get('manual_references') or existing.get('manual_geometry_base'):
            filename = 'west_manual_corrected_survey.json' if d == 'east_to_west' else 'east_recovered_centers.json'
            manual = import_manual(BUNDLE / filename)['datasets'][d]
            datasets[d] = manual
        def report(value, direction=d):
            if progress:
                progress(dict(direction=direction, **value))
        datasets[d] = add_manual_rays(datasets[d], report)
    return document_result(new_document(datasets, payload.get('configuration_version', 0)))


def dispatch(method, payload, progress=None):
    if not isinstance(payload, dict):
        raise ValueError("Параметры должны быть JSON-объектом.")
    if payload.get("direction") is not None:
        selected_directions(payload["direction"])
    # All entry paths use the same versioned optics, including direct API
    # requests that bypass the file-preview/accept workflow.
    from .ring_migration import upgrade_ring_document
    payload = dict(payload)
    for key in ('document', 'active_configuration', 'candidate_configuration'):
        if payload.get(key) is not None and method != 'trace_static':
            payload[key] = upgrade_ring_document(payload[key])
    if method == "catalog":
        return catalog()
    if method == "trace":
        return trace(payload)
    if method == "trace_static":
        return trace_static(payload)
    if method == "calibrate":
        return calibrate(payload)
    if method == "optimize":
        return optimize(payload, progress)
    if method == "refine_manual":
        return refine_manual_configuration(payload, progress)
    if method == 'add_manual_rays':
        return add_manual_analysis_rays(payload, progress)
    if method == "save_configuration":
        from .state import save_document
        document = payload.get("document") or payload.get("active_configuration") or _historical_document(payload)
        return save_document(document, payload["path"])
    if method == "load_configuration":
        from .state import load_document
        return document_result(load_document(payload["path"]))
    if method == "import_legacy":
        from .legacy_import import import_legacy
        return document_result(import_legacy(payload["path"], payload.get("direction"),
                                             payload.get("coordinate_frame"), payload.get("kind", "real")))
    if method == "import_manual":
        from .manual_schema import import_manual
        return document_result(import_manual(payload["path"], payload.get("direction"),
                                             payload.get("coordinate_frame")))
    raise ValueError(f"Неизвестная операция: {method}")
