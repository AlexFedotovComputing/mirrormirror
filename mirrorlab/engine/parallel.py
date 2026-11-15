"""CPU process parallelism over disjoint subsets of one emitted ray grid."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os

import numpy as np

from .rays import RayBundle
from .tracer import RayTracer, TraceResult

_tracer = None


def _initialize(scene, options):
    global _tracer
    _tracer = RayTracer(scene, **options)


def _trace_chunk(rays):
    result = _tracer.trace(rays)
    return int(rays.ray_id.max()), result, os.getpid()


def trace_cpu(scene, rays, *, cpu_workers=6, **options):
    """Keep source weights unchanged and make descendant IDs globally unique."""
    workers = min(cpu_workers, os.cpu_count() or 1, max(1, rays.n_rays))
    # A global strongest-ray cutoff cannot be applied independently per partition.
    if options.get('max_active_rays') is not None:
        workers = 1
    if workers == 1:
        result = RayTracer(scene, **options).trace(rays)
        result.diagnostics.update(cpu_workers=1, worker_pids=[os.getpid()])
        return result
    chunks = [rays.subset_preserving_values(ids) for ids in
              np.array_split(np.arange(rays.n_rays), workers)]
    # Each process owns a partition; numerical libraries must not spawn another
    # full team of CPU threads inside every worker.
    names = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
    previous = {name: os.environ.get(name) for name in names}
    try:
        os.environ.update({name: '1' for name in names})
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_initialize, initargs=(scene, options)) as pool:
            parts = list(pool.map(_trace_chunk, chunks))
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    next_id = int(rays.ray_id.max()) + 1
    segments, hits, finals = [], [], []
    totals = dict(intersection_tests=0, possible_tests=0, unfinished_count=0)
    for initial_max, result, pid in parts:
        arrays = [result.final_rays.ray_id, result.final_rays.parent_id]
        for block in result.segments + result.detector_hits:
            arrays.extend(block[key] for key in ('ray_id', 'parent_id') if key in block)
        largest = max((int(a.max()) for a in arrays if a.size), default=initial_max)
        offset = next_id - initial_max - 1
        # Some detector arrays may share storage. Replace rather than mutate them.
        def remap(a):
            return np.where(a > initial_max, a + offset, a)
        result.final_rays.ray_id = remap(result.final_rays.ray_id)
        result.final_rays.parent_id = remap(result.final_rays.parent_id)
        for block in result.segments + result.detector_hits:
            for key in ('ray_id', 'parent_id'):
                if key in block:
                    block[key] = remap(block[key])
        next_id += max(0, largest - initial_max)
        finals.append(result.final_rays)
        segments.extend(result.segments)
        hits.extend(result.detector_hits)
        for key in totals:
            totals[key] += result.diagnostics[key]
    return TraceResult(RayBundle.concat(finals), segments, hits, parts[0][1].surface_names,
                       {**totals, 'terminated_by_interaction_limit': totals['unfinished_count'] > 0,
                        'collision_mode': options.get('collision_mode', 'exhaustive'),
                        'cpu_workers': workers, 'worker_pids': sorted({p[2] for p in parts})})
