"""Bounded report-only translation for the native checked compatibility bridge."""
from .safety import ContractError, canonical, integer, keys

MAX_PROJECTION_REQUEST_BYTES = 4096
MAX_PROJECTION_REPORT_BYTES = 1024 * 1024
MAX_PROJECTION_WORKER_BYTES = 2 * 1024 * 1024
MAX_VIOLATION_IDS = 128
PROJECTION_LIMITS = {
    'max_nodes': 256, 'max_edges': 1024, 'max_source_points': 4096,
    'max_output_points': 8192, 'max_crosslinks': 16384,
    'max_metadata_bytes': 1024 * 1024, 'max_report_bytes': MAX_PROJECTION_REPORT_BYTES,
}
REPORT_FIELDS = frozenset('schema_version scope source_authority representation geometry_policy short_edge_policy missing_material_policy length_unit coordinate_frame declaration_origin length_scope unit_conversion frame_transform dimension coordinate_components source_node_order edges nonempty_internal_point_edge_ids projected_internal_point_edge_ids omitted_edge_ids missing_material_edge_ids defaulted_material_edge_ids unrepresented_node_ids counts metadata field_disposition junction_rule crosslink_strength_and_stiffness short_chord_threshold not_qualified'.split())


def checked_report(report, request):
    """Validate transport identity/shape, without recalculating geometry or losses."""
    if (type(report) is not dict or set(report) != REPORT_FIELDS or
            report['schema_version'] != 'fibernet.fiber_network_projection.v1' or
            report['scope'] != 'conversion_time_correspondence_to_returned_network_only' or
            any(report[key] != request[key] for key in ('length_unit', 'coordinate_frame',
                'geometry_policy', 'short_edge_policy', 'missing_material_policy')) or
            type(report['edges']) is not list or type(report['counts']) is not dict):
        raise ProjectionContractError('PROJECTION_OUTPUT_CONTRACT_FAILURE', 'internal')
    raw = canonical(report)
    if len(raw) > MAX_PROJECTION_REPORT_BYTES:
        raise ProjectionContractError('PROJECTION_RESULT_LIMIT', 'resource')
    return raw


class ProjectionContractError(ContractError):
    def __init__(self, code, category='contract'):
        super().__init__(code)
        self.code, self.category = code, category


def projection_request(graph_info, arguments):
    """Only approved declarations and fixed ceilings enter the native profile."""
    keys(arguments, {'graph_id', 'geometry_policy', 'short_edge_policy'}, set())
    if (graph_info['node_count'] > PROJECTION_LIMITS['max_nodes'] or
            graph_info['edge_count'] > PROJECTION_LIMITS['max_edges']):
        raise ProjectionContractError('PROJECTION_GRAPH_LIMIT', 'resource')
    if graph_info['length_unit'] not in ('m', 'cm', 'mm', 'um', 'nm'):
        raise ProjectionContractError('DECLARED_LENGTH_UNIT_REQUIRED')
    if graph_info['frame_id'] == 'unknown':
        raise ProjectionContractError('DECLARED_FRAME_REQUIRED')
    geometry = arguments.get('geometry_policy', 'reject_internal_points')
    short = arguments.get('short_edge_policy', 'reject')
    if (type(geometry) is not str or geometry not in ('reject_internal_points', 'endpoint_chord') or
            type(short) is not str or short not in ('reject', 'omit_legacy')):
        raise ProjectionContractError('INVALID_PROJECTION_INPUT')
    request = {'length_unit': graph_info['length_unit'],
               'coordinate_frame': graph_info['frame_id'],
               'geometry_policy': geometry, 'short_edge_policy': short,
               'missing_material_policy': 'reject', 'limits': dict(PROJECTION_LIMITS)}
    if len(canonical(request)) > MAX_PROJECTION_REQUEST_BYTES:
        raise ProjectionContractError('PROJECTION_REQUEST_LIMIT', 'resource')
    return request


def project_report(data, request):
    """One profiled public dispatch; the native owner alone performs projection.

    The returned FiberNetwork is intentionally not transferred or serialized.
    Admission limits apply after the existing bounded saved-graph loader.
    """
    from fibernet.core.structure_graph import StructureGraph
    from fibernet.core.fiber_network_profile import (
        FiberNetworkProfile, FiberNetworkProjectionLimits, FiberNetworkProjectionError)
    try:
        expected = projection_request(
            {'node_count': len(data['nodes']), 'edge_count': len(data['edges']),
             'length_unit': request['length_unit'], 'frame_id': request['coordinate_frame']},
            {name: request[name] for name in ('geometry_policy', 'short_edge_policy')})
        if canonical(request) != canonical(expected):
            raise ProjectionContractError('INVALID_PROJECTION_INPUT')
        graph = StructureGraph.from_dict(data)
        profile = FiberNetworkProfile(**{**request,
            'limits': FiberNetworkProjectionLimits(**request['limits'])})
        network, report = graph.to_fiber_network(profile=profile)
        # No network.to_dict(): legacy crosslinks contain nonfinite sentinels.
        del network
        checked_report(report, request)
        return {'projection_report': report}
    except ProjectionContractError as exc:
        return {'error_code': exc.code, 'error_category': exc.category}
    except FiberNetworkProjectionError as exc:
        if set(exc.violations) - {'internal_points', 'short_edges', 'missing_materials'}:
            return {'error_code': 'PROJECTION_OUTPUT_CONTRACT_FAILURE', 'error_category': 'internal'}
        try:
            for ids in exc.violations.values():
                if len(ids) > PROJECTION_LIMITS['max_edges']:
                    raise ContractError('violation count limit')
                for value in ids: integer(value)
        except ContractError:
            return {'error_code': 'PROJECTION_OUTPUT_CONTRACT_FAILURE', 'error_category': 'internal'}
        violations = [{'kind': name, 'edge_ids': list(ids[:MAX_VIOLATION_IDS]),
                       'total_count': len(ids), 'truncated': len(ids) > MAX_VIOLATION_IDS}
                      for name, ids in exc.violations.items()]
        return {'error_code': 'PROJECTION_POLICY_REJECTED', 'error_category': 'domain',
                'diagnostics': {'violations': violations}}
    except MemoryError:
        return {'error_code': 'PROJECTION_LIMIT', 'error_category': 'resource'}
    except OverflowError:
        return {'error_code': 'PROJECTION_NUMERIC_RANGE', 'error_category': 'domain'}
    except (ValueError, TypeError, KeyError, ZeroDivisionError):
        return {'error_code': 'INVALID_PROJECTION_INPUT', 'error_category': 'domain'}
