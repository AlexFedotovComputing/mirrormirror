"""Nominal drift-volume inner boundary, MPD TPC TDR rev.07, table 1."""
INNER_RADIUS_M = .34
TDR_URL = 'https://mpd.jinr.ru/wp-content/uploads/2023/11/TpcTdr-v07.pdf'


def apply_gas_boundary(config):
    config.tracer_options['bundle_clip_inner_radius_m'] = INNER_RADIUS_M
    # This absorbing detector otherwise terminates rays at the old 0.5 m
    # radius before they can reach the actual clipping boundary.
    for surface in config.scene.build_surfaces():
        if surface.name == 'Cylindrical Screen 1':
            surface.radius = INNER_RADIUS_M
            surface.__post_init__()
    config.metadata['gas_inner_boundary'] = dict(radius_m=INNER_RADIUS_M, source=TDR_URL)


def upgrade_gas_dataset(dataset):
    import copy
    from .state import configuration_from_dataset, snapshot, digest
    from .calibration.physical import validate_reflections
    result = copy.deepcopy(dataset)
    if not result.get('optics_snapshot'):
        return result
    config = configuration_from_dataset(result)
    if (config.tracer_options.get('bundle_clip_inner_radius_m') == INNER_RADIUS_M and
        all(s.radius == INNER_RADIUS_M for s in config.scene.build_surfaces()
            if s.name == 'Cylindrical Screen 1')):
        return result
    result.setdefault('gas_boundary_previous_optics', copy.deepcopy(result['optics_snapshot']))
    apply_gas_boundary(config)
    result['optics_snapshot'] = snapshot(config)
    result['optics_sha256'] = digest(result['optics_snapshot'])
    if result.get('physical_validation') is not None or result.get('diagnostics', {}).get('reflections') is not None:
        report = validate_reflections(config, result['mirrors'])
        if result.get('physical_validation') is not None:
            result['physical_validation'] = report
        if result.get('diagnostics', {}).get('reflections') is not None:
            result['diagnostics']['reflections'] = report
    return result
