"""Strict prepare-only translation to the owning StructureGraph API.

This module contains no solver, graph replacement, or transport equations.
Scientific imports stay in the confined worker.
"""
import re

from .safety import (ContractError, canonical, check_tree, graph_preflight,
                     integer, keys, number, sha, strict_json, vector)

MAX_PREPARATION_NODES = 1536
MAX_PREPARATION_EDGES = 2048
MAX_PREPARATION_METADATA_BYTES = 1024 * 1024
MAX_PREPARATION_REPORT_BYTES = 2 * 1024 * 1024
MAX_PREPARATION_SOURCE_BYTES = 4 * 1024 * 1024
MAX_PREPARATION_REQUEST_BYTES = 1024 * 1024
MAX_PREPARATION_RESULT_BYTES = 4 * 1024 * 1024
MAX_PREPARATION_ENVELOPE_BYTES = MAX_PREPARATION_RESULT_BYTES + 4096
NATIVE_LIMITS = dict(max_nodes=MAX_PREPARATION_NODES, max_edges=MAX_PREPARATION_EDGES,
                     max_metadata_bytes=MAX_PREPARATION_METADATA_BYTES,
                     max_report_bytes=MAX_PREPARATION_REPORT_BYTES)
PREPARATION_LIMITS = dict(NATIVE_LIMITS, max_source_bytes=MAX_PREPARATION_SOURCE_BYTES,
                         max_request_bytes=MAX_PREPARATION_REQUEST_BYTES,
                         max_result_bytes=MAX_PREPARATION_RESULT_BYTES)
BUNDLE_SCHEMA = 'fibernet.mcp.deformed_transport_bundle/1'
_ERROR_CATEGORIES = {
    'INVALID_PREPARATION_INPUT': 'contract',
    'DECLARED_LENGTH_UNIT_REQUIRED': 'contract',
    'PREPARATION_UNIT_CONFLICT': 'contract',
    'PREPARATION_FRAME_CONFLICT': 'contract',
    'DUPLICATE_TRANSLATION_ID': 'contract',
    'DUPLICATE_CONDUCTANCE_ID': 'contract',
    'PREPARATION_MAP_COUNT': 'contract',
    'PREPARATION_SOURCE_ROUNDTRIP': 'contract',
    'PREPARATION_GRAPH_LIMIT': 'resource',
    'PREPARATION_SOURCE_LIMIT': 'resource',
    'PREPARATION_REQUEST_LIMIT': 'resource',
    'PREPARATION_RESULT_LIMIT': 'resource',
    'PREPARATION_RESOURCE_LIMIT': 'resource',
    'PREPARATION_REJECTED': 'domain',
    'CORE_DEPENDENCY_UNAVAILABLE': 'internal',
    'NATIVE_PREPARATION_INVARIANT': 'internal',
}


class PreparationContractError(ContractError):
    def __init__(self, code, category=None):
        self.code = code
        self.category = _ERROR_CATEGORIES.get(code, 'contract') if category is None else category
        super().__init__(code)


def _records(rows, id_field, value_field, count, duplicate_code, *, translations=False):
    if type(rows) is not list or len(rows) != count:
        raise PreparationContractError('PREPARATION_MAP_COUNT')
    seen = set()
    for row in rows:
        keys(row, {id_field, value_field}, {id_field, value_field})
        integer(row[id_field])
        if row[id_field] in seen:
            raise PreparationContractError(duplicate_code)
        seen.add(row[id_field])
        if translations:
            vector(row[value_field], 2)
        else:
            number(row[value_field], positive=True)
    # Do not coerce values or decide native complete-map membership here.
    return {row[id_field]: row[value_field] for row in rows}


def preparation_request(graph_info, arguments, source_bytes=None):
    """Admit bounded wire records without filling defaults or coercing numbers."""
    try:
        keys(arguments, {'graph_id', 'length_unit', 'translations_m', 'reference_conductance_S'},
             {'graph_id', 'length_unit', 'translations_m', 'reference_conductance_S'})
        check_tree(arguments)
        if not (0 < graph_info['node_count'] <= MAX_PREPARATION_NODES and
                0 < graph_info['edge_count'] <= MAX_PREPARATION_EDGES):
            raise PreparationContractError('PREPARATION_GRAPH_LIMIT')
        if source_bytes is not None and source_bytes > MAX_PREPARATION_SOURCE_BYTES:
            raise PreparationContractError('PREPARATION_SOURCE_LIMIT')
        unit = arguments['length_unit']
        if type(unit) is not str or unit not in ('m', 'mm', 'um'):
            raise PreparationContractError('DECLARED_LENGTH_UNIT_REQUIRED')
        if unit != graph_info['length_unit']:
            raise PreparationContractError('PREPARATION_UNIT_CONFLICT')
        _records(arguments['translations_m'], 'node_id', 'value_m', graph_info['node_count'],
                 'DUPLICATE_TRANSLATION_ID', translations=True)
        _records(arguments['reference_conductance_S'], 'edge_id', 'value_S', graph_info['edge_count'],
                 'DUPLICATE_CONDUCTANCE_ID')
        request = {key: arguments[key] for key in
                   ('length_unit', 'translations_m', 'reference_conductance_S')}
        request['frame_id'] = graph_info['frame_id']
        if len(canonical(request)) > MAX_PREPARATION_REQUEST_BYTES:
            raise PreparationContractError('PREPARATION_REQUEST_LIMIT')
        return request
    except PreparationContractError:
        raise
    except (ContractError, TypeError, KeyError, ValueError, OverflowError):
        raise PreparationContractError('INVALID_PREPARATION_INPUT') from None


def _ordered_record(record):
    return dict(record, nodes=sorted(record['nodes'], key=lambda row: row['id']),
                edges=sorted(record['edges'], key=lambda row: row['id']))


def _lossless_decode(data, graph_class):
    graph = graph_class.from_dict(data)
    # Value equality permits exactly equal int/float JSON numbers, not coercion
    # of IDs or loss of nested keys, material fields, flags, or counters.
    if (graph.to_dict() != _ordered_record(data) or
            list(graph.nodes) != [row['id'] for row in data['nodes']] or
            list(graph.edges) != [row['id'] for row in data['edges']]):
        raise PreparationContractError('PREPARATION_SOURCE_ROUNDTRIP')
    return graph



def validate_bundle(bundle, core_provenance, source_sha256):
    """Validate the immutable wire snapshot, without recalculating physics."""
    try:
        keys(bundle, {'schema', 'source_sha256', 'core_provenance', 'derived_graph',
                      'conductance_S', 'binding_report'},
             {'schema', 'source_sha256', 'core_provenance', 'derived_graph',
              'conductance_S', 'binding_report'})
        if (bundle['schema'] != BUNDLE_SCHEMA or bundle['source_sha256'] != source_sha256 or
                canonical(bundle['core_provenance']) != canonical(core_provenance)):
            raise ContractError('bundle identity mismatch')
        graph = graph_preflight(bundle['derived_graph'])
        if not (0 < len(graph['nodes']) <= MAX_PREPARATION_NODES and
                0 < len(graph['edges']) <= MAX_PREPARATION_EDGES):
            raise ContractError('bundle graph limit')
        rows = _records(bundle['conductance_S'], 'edge_id', 'value_S', len(graph['edges']),
                        'DUPLICATE_CONDUCTANCE_ID')
        if set(rows) != {row['id'] for row in graph['edges']}:
            raise ContractError('bundle edge membership mismatch')
        report = bundle['binding_report']
        expected_fields = {
            'area_policy', 'arithmetic', 'box_boundary_tiling_interpretation', 'conductance_S',
            'conductance_unit', 'conductivity_policy', 'coordinate_frame', 'declaration_origin',
            'derived_full_graph_sha256', 'derived_graph_order_envelope_sha256', 'derived_lengths_m',
            'derived_node_AABB_m', 'displacement_scale', 'edge_ids', 'endpoints_by_id', 'feedback',
            'frame_transform', 'frozen_sigma_A_S_m', 'geometry_policy', 'graph_field_disposition',
            'input_kind', 'input_sha256', 'length_unit', 'mechanical_result_admitted', 'node_ids',
            'reference_box_size_m', 'reference_conductance_S', 'reference_lengths_m', 'schema',
            'scope', 'size_admission', 'solver_executed', 'source_edge_insertion_order',
            'source_full_graph_sha256', 'source_graph_order_envelope_sha256',
            'source_node_insertion_order', 'translation_frame', 'translation_unit', 'translations_m'}
        keys(report, expected_fields, expected_fields)
        if (report['schema'] != 'fibernet.deformed_transport_preparation/1'
                or report['scope'] != 'preparation_time_correspondence_only'
                or report['solver_executed'] is not False
                or report['mechanical_result_admitted'] is not False
                or report['feedback'] is not False
                or report['derived_full_graph_sha256'] != sha(canonical(graph))):
            raise ContractError('bundle report mismatch')
        for field in ('source_full_graph_sha256', 'derived_full_graph_sha256',
                      'source_graph_order_envelope_sha256', 'derived_graph_order_envelope_sha256', 'input_sha256'):
            if type(report[field]) is not str or not re.fullmatch('[a-f0-9]{64}', report[field]):
                raise ContractError('invalid native hash')
        node_ids=sorted(row['id'] for row in graph['nodes'])
        edge_ids=sorted(row['id'] for row in graph['edges'])
        for field, expected in (('node_ids',node_ids), ('edge_ids',edge_ids),
                                ('source_node_insertion_order',node_ids), ('source_edge_insertion_order',edge_ids)):
            values=report[field]
            if type(values) is not list or len(values)!=len(expected):
                raise ContractError('invalid native IDs')
            for value in values: integer(value)
            if (values if field in ('node_ids','edge_ids') else sorted(values)) != expected:
                raise ContractError('native ID membership mismatch')
        if list(rows) != report['source_edge_insertion_order']:
            raise ContractError('conductance order mismatch')
        if report['conductance_S'] != [rows[eid] for eid in edge_ids]:
            raise ContractError('native conductance correspondence mismatch')
        for field in ('reference_conductance_S','conductance_S','reference_lengths_m',
                      'derived_lengths_m','frozen_sigma_A_S_m'):
            values=report[field]
            if type(values) is not list or len(values)!=len(edge_ids):
                raise ContractError('native edge array shape mismatch')
            for value in values: number(value, positive=True)
        translations=report['translations_m']
        if type(translations) is not list or len(translations)!=len(node_ids):
            raise ContractError('native translation shape mismatch')
        for row in translations: vector(row,2)
        edges={row['id']:row for row in graph['edges']}
        if report['endpoints_by_id'] != [[edges[eid]['node_i'],edges[eid]['node_j']] for eid in edge_ids]:
            raise ContractError('native endpoint correspondence mismatch')
        for endpoints in report['endpoints_by_id']:
            for value in endpoints: integer(value)
        expected_order_hash=sha(canonical({'graph':graph,
            'node_insertion_order':report['source_node_insertion_order'],
            'edge_insertion_order':report['source_edge_insertion_order']}))
        if report['derived_graph_order_envelope_sha256']!=expected_order_hash:
            raise ContractError('native order binding mismatch')
        admission=report['size_admission']
        if (type(admission) is not dict or set(admission)!=set(NATIVE_LIMITS)|{
                'fixed_ceilings','max_spatial_buckets','metadata_material_bytes_charged','scope'}
                or any(type(admission[k]) is not int or admission[k]!=v for k,v in NATIVE_LIMITS.items())
                or type(admission['max_spatial_buckets']) is not int
                or admission['max_spatial_buckets']!=4*MAX_PREPARATION_NODES
                or admission['scope']!='payload_bounds_not_peak_memory_or_physical_qualification'):
            raise ContractError('native admission binding mismatch')
        ceilings={'max_nodes':4096,'max_edges':16384,'max_metadata_bytes':4194304,'max_report_bytes':16777216}
        if (type(admission['fixed_ceilings']) is not dict or set(admission['fixed_ceilings'])!=set(ceilings)
                or any(type(admission['fixed_ceilings'][k]) is not int or admission['fixed_ceilings'][k]!=v for k,v in ceilings.items())):
            raise ContractError('native ceiling binding mismatch')
        constants={'area_policy':'fixed_effective_area','geometry_policy':'endpoint_chord',
            'conductivity_policy':'frozen_reference_sigma_A','conductance_unit':'S',
            'translation_unit':'m','translation_frame':'source_graph_xy_axes',
            'input_kind':'caller_prescribed_translations','frame_transform':'none',
            'arithmetic':'binary64 actual stored coordinates and endpoint chords',
            'graph_field_disposition':'all serialized fields preserved except node positions; lookup remains consistent; materials/radii not interpreted',
            'box_boundary_tiling_interpretation':'unchanged reference/source provenance only; not updated electrical boundaries or periodic physics'}
        if any(report[k]!=v for k,v in constants.items()):
            raise ContractError('native policy binding mismatch')
        if (report['length_unit'] not in ('m','mm','um') or type(report['displacement_scale']) is not int
                or report['displacement_scale']!=1):
            raise ContractError('native units mismatch')
        frame=report['coordinate_frame']
        if frame is not None and (type(frame) is not str or not frame.strip()):
            raise ContractError('native frame mismatch')
        declarations={'length_unit':'caller_asserted_not_certified',
            'coordinate_frame':'stored_label_not_authenticated' if frame else 'undeclared',
            'translations':'caller_prescribed_in_source_graph_xy_axes'}
        if report['declaration_origin']!=declarations:
            raise ContractError('native declaration mismatch')
        if report['reference_box_size_m'] is not None: vector(report['reference_box_size_m'],3)
        aabb=report['derived_node_AABB_m']
        if type(aabb) is not list or len(aabb)!=2: raise ContractError('native AABB shape mismatch')
        for row in aabb: vector(row,3)
        integer(admission['metadata_material_bytes_charged'])
        if admission['metadata_material_bytes_charged']>MAX_PREPARATION_METADATA_BYTES:
            raise ContractError('native metadata limit')
        if len(canonical(report)) > MAX_PREPARATION_REPORT_BYTES:
            raise PreparationContractError('PREPARATION_RESULT_LIMIT')
        check_tree(bundle)
        raw = canonical(bundle)
        if len(raw) > MAX_PREPARATION_RESULT_BYTES:
            raise PreparationContractError('PREPARATION_RESULT_LIMIT')
        return raw
    except PreparationContractError as exc:
        if exc.code == 'PREPARATION_RESULT_LIMIT': raise
        raise PreparationContractError('NATIVE_PREPARATION_INVARIANT') from None
    except (ContractError, TypeError, KeyError, ValueError, OverflowError):
        raise PreparationContractError('NATIVE_PREPARATION_INVARIANT') from None


def validate_worker_result(result, core_provenance, source_sha256):
    try:
        if type(result) is not dict:
            raise ContractError('invalid worker result')
        if set(result) == {'error_code', 'error_category', '_core_provenance'}:
            if (type(result['error_code']) is not str or type(result['error_category']) is not str
                    or _ERROR_CATEGORIES.get(result['error_code']) != result['error_category']):
                raise ContractError('invalid worker failure')
        elif set(result) == {'preparation', '_core_provenance'}:
            validate_bundle(result['preparation'], core_provenance, source_sha256)
        else:
            raise ContractError('invalid worker result shape')
        if canonical(result['_core_provenance']) != canonical(core_provenance):
            raise ContractError('invalid worker provenance')
        return result
    except PreparationContractError:
        raise
    except (ContractError, TypeError, KeyError, ValueError):
        raise PreparationContractError('NATIVE_PREPARATION_INVARIANT') from None


def prepare_transport(data, request, core_provenance, source_sha256):
    """Call the native owner exactly once, returning its unmodified outputs."""
    try:
        keys(request, {'length_unit', 'frame_id', 'translations_m', 'reference_conductance_S'},
             {'length_unit', 'frame_id', 'translations_m', 'reference_conductance_S'})
        graph_preflight(data)
        info = {'node_count': len(data['nodes']), 'edge_count': len(data['edges']),
                'length_unit': request['length_unit'], 'frame_id': request['frame_id']}
        args = {key: value for key, value in request.items() if key != 'frame_id'}
        args['graph_id'] = '0' * 32
        preparation_request(info, args)
        from fibernet.core.structure_graph import StructureGraph
        from fibernet.sim.deformed_transport import prepare_deformed_transport
        graph = _lossless_decode(data, StructureGraph)
        translations = _records(request['translations_m'], 'node_id', 'value_m', len(data['nodes']),
                                'DUPLICATE_TRANSLATION_ID', translations=True)
        conductance = _records(request['reference_conductance_S'], 'edge_id', 'value_S', len(data['edges']),
                               'DUPLICATE_CONDUCTANCE_ID')
        prepared = prepare_deformed_transport(
            graph, translations_m_by_id=translations, reference_conductance_S=conductance,
            length_unit=request['length_unit'], **NATIVE_LIMITS)
        try:
            derived, values, report = prepared
            if type(derived) is not StructureGraph or type(values) is not dict or type(report) is not dict:
                raise PreparationContractError('NATIVE_PREPARATION_INVARIANT')
            bundle = {'schema': BUNDLE_SCHEMA, 'source_sha256': source_sha256,
                      'core_provenance': core_provenance, 'derived_graph': derived.to_dict(),
                      'conductance_S': [{'edge_id': eid, 'value_S': value} for eid, value in values.items()],
                      'binding_report': report}
            raw = validate_bundle(bundle, core_provenance, source_sha256)
            if (request['frame_id'] != 'unknown' and report['coordinate_frame'] is not None
                    and request['frame_id'] != report['coordinate_frame']):
                raise PreparationContractError('PREPARATION_FRAME_CONFLICT')
            # Detach the snapshot while native return values are still owned.
            return {'preparation': strict_json(raw)}
        except PreparationContractError:
            raise
        except (AttributeError, ValueError, TypeError, KeyError, OverflowError):
            raise PreparationContractError('NATIVE_PREPARATION_INVARIANT') from None

    except PreparationContractError as exc:
        return {'error_code': exc.code, 'error_category': exc.category}
    except ImportError:
        return {'error_code': 'CORE_DEPENDENCY_UNAVAILABLE', 'error_category': 'internal'}
    except MemoryError:
        return {'error_code': 'PREPARATION_RESOURCE_LIMIT', 'error_category': 'resource'}
    except RuntimeError:
        return {'error_code': 'NATIVE_PREPARATION_INVARIANT', 'error_category': 'internal'}
    except (ValueError, TypeError, KeyError, OverflowError, ZeroDivisionError):
        return {'error_code': 'PREPARATION_REJECTED', 'error_category': 'domain'}
