"""Explicit, reversible application upgrade of the WEST-side distributor.

Portable state decoding stays lossless. The application upgrades old EAST
datasets on entry and records both optical snapshots; fitted bodies, survey
registration and measurement assignments are never recomputed here.
"""
from __future__ import annotations

import copy

import numpy as np


def upgrade_ring_dataset(dataset):
    from .ring_alignment import RING_ALIGNMENT_REVISION, apply_ring_alignment
    from .state import configuration_from_dataset, digest, snapshot

    if dataset['direction'] != 'west_to_east' or dataset.get('optics_snapshot') is None:
        return copy.deepcopy(dataset)
    previous = dataset['optics_snapshot']
    correction = previous.get('metadata', {}).get('ring_alignment')
    if correction is not None:
        if correction.get('revision') != RING_ALIGNMENT_REVISION:
            raise ValueError('Unsupported ring alignment revision')
        return copy.deepcopy(dataset)
    if digest(previous) != dataset.get('optics_sha256'):
        raise ValueError('Embedded optics hash mismatch before ring upgrade')
    config = configuration_from_dataset(dataset)
    from .calibration.physical import validate_reflections
    before = validate_reflections(config, dataset['mirrors'])
    apply_ring_alignment(config)
    after = validate_reflections(config, dataset['mirrors'])
    hit_errors, direction_errors = [], []
    for key, old in before['mirrors'].items():
        new = after['mirrors'][key]
        if old['status'] == 'validated' and new['status'] != 'validated':
            raise ValueError(f'Ring correction degraded the fitted physical path {key}')
        if 'outgoing_direction' in old:
            if 'outgoing_direction' not in new:
                raise ValueError(f'Ring correction lost the fitted reflected path {key}')
            hit_errors.append(float(np.linalg.norm(np.asarray(old['hit_m'])-new['hit_m'])))
            direction_errors.append(float(np.linalg.norm(np.asarray(old['outgoing_direction'])-new['outgoing_direction'])))
    if max(hit_errors, default=0) > 1e-8 or max(direction_errors, default=0) > 1e-9:
        raise ValueError('Ring correction changed fitted reflected rays')
    result = copy.deepcopy(dataset)
    result['ring_alignment_previous_optics'] = copy.deepcopy(previous)
    result['ring_alignment_previous_sha256'] = digest(previous)
    result['optics_snapshot'] = snapshot(config)
    result['optics_sha256'] = digest(result['optics_snapshot'])
    result['optics_origin'] = 'west_ring_alignment_2026_11_16'
    result['ring_alignment_validation'] = dict(
        validated_before=before['validated_count'], validated_after=after['validated_count'],
        unavailable_before=before['unavailable_count'], unavailable_after=after['unavailable_count'],
        max_hit_shift_m=max(hit_errors, default=0), max_outgoing_direction_error=max(direction_errors, default=0))
    # Stored physical paths include their upstream branches. Refresh these
    # reports after the optical change; never present stale paths as current.
    if result.get('physical_validation') is not None:
        result['physical_validation'] = after
    if result.get('diagnostics', {}).get('reflections') is not None:
        result['diagnostics']['reflections'] = after
    return result


def upgrade_ring_document(document):
    from .state import validate_document

    validate_document(document)
    result = copy.deepcopy(document)
    if 'west_to_east' in result['datasets']:
        result['datasets']['west_to_east'] = upgrade_ring_dataset(result['datasets']['west_to_east'])
        from .gas_volume import upgrade_gas_dataset
        result['datasets']['west_to_east'] = upgrade_gas_dataset(result['datasets']['west_to_east'])
        if result['datasets']['west_to_east'].get('optics_sha256') != document['datasets']['west_to_east'].get('optics_sha256'):
            result['configuration_version'] = document.get('configuration_version', 0)+1
    validate_document(result)
    return result
