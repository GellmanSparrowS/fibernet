"""Bounded wire contract for the existing prescribed-temperature native owner.

No assembly, solver, temperature inference, material inference or equation is
implemented here. Scientific imports occur only in the confined worker route.
"""
import re

from .safety import (ContractError, canonical, check_tree, integer, keys, number,
                     sha, vector)

PROFILE = 'thermoelastic_2d_network/1'
MAX_SNAPSHOT_SOURCE_BYTES = 1024 * 1024
MAX_SNAPSHOT_REQUEST_BYTES = 64 * 1024
MAX_SNAPSHOT_RESULT_BYTES = 1024 * 1024
MAX_SNAPSHOT_ENVELOPE_BYTES = MAX_SNAPSHOT_RESULT_BYTES + 4096
SNAPSHOT_LIMITS = {'max_nodes': 192, 'max_edges': 256,
                  'max_source_bytes': MAX_SNAPSHOT_SOURCE_BYTES,
                  'max_request_bytes': MAX_SNAPSHOT_REQUEST_BYTES,
                  'max_result_bytes': MAX_SNAPSHOT_RESULT_BYTES}
ARGUMENTS = {'graph_id', 'length_unit', 'E_Pa', 'stress_free_temperature_K',
             'temperature_K', 'expansion_alpha_per_K', 'constraints', 'loads'}
BINDING = {'frame_id', 'frame_provenance', 'unit_provenance'}
IDENTITY_FIELDS = {'profile', 'node_ids', 'edge_ids', 'original_geometry',
    'endpoints_by_id', 'positions_m', 'radii_m', 'length_unit', 'E_Pa',
    'stress_free_temperature_K', 'temperature_K', 'expansion_alpha_per_K',
    'constraint_mask', 'prescribed_values', 'loads'}
SNAPSHOT_FIELDS = IDENTITY_FIELDS | {'input_sha256', 'units', 'interpolation',
    'numerical_regularization', 'total_axial_strain', 'thermal_axial_strain',
    'elastic_axial_strain', 'displacement', 'sigma_axial', 'sigma_bending',
    'sigma_total', 'moments', 'edge_forces', 'reactions', 'free_dof_indices',
    'free_residual', 'free_acceptance_bound', 'component_balance',
    'component_acceptance_bound', 'applicability', 'balance_origin_node_id'}
UNITS = {'length': 'm', 'rotation': 'rad', 'force': 'N', 'moment': 'N*m',
         'stress': 'Pa', 'temperature': 'K', 'alpha': '1/K'}
_ERROR_CATEGORIES = {
    'INVALID_THERMOELASTIC_INPUT': 'contract', 'SNAPSHOT_UNIT_CONFLICT': 'contract',
    'SNAPSHOT_GRAPH_LIMIT': 'resource', 'SNAPSHOT_SOURCE_LIMIT': 'resource',
    'SNAPSHOT_REQUEST_LIMIT': 'resource', 'SNAPSHOT_RESULT_LIMIT': 'resource',
    'SNAPSHOT_RESOURCE_LIMIT': 'resource', 'SNAPSHOT_SOURCE_ROUNDTRIP': 'contract',
    'SNAPSHOT_DOMAIN_REJECTED': 'domain', 'CORE_DEPENDENCY_UNAVAILABLE': 'internal',
    'NATIVE_SNAPSHOT_INVARIANT': 'internal', 'OUTPUT_CONTRACT_FAILURE': 'internal'}


class SnapshotContractError(ContractError):
    def __init__(self, code):
        self.code = code
        self.category = _ERROR_CATEGORIES[code]
        super().__init__(code)


def _rows(rows, ids, id_key, fields, *, complete=False, nullable=False, positive=False):
    if type(rows) is not list or len(rows) > len(ids):
        raise ContractError('record count')
    result = {}
    for row in rows:
        keys(row, {id_key, *fields}, {id_key, *fields})
        nid = row[id_key]
        integer(nid)
        if nid not in ids or nid in result:
            raise ContractError('duplicate or unknown ID')
        values = []
        for field in fields:
            value = row[field]
            if value is not None or not nullable:
                number(value, positive=positive)
            values.append(value)
        result[nid] = values[0] if len(fields) == 1 else values
    if complete and set(result) != set(ids):
        raise ContractError('incomplete ID coverage')
    return result


def _maps(arguments, data):
    nodes = {row['id'] for row in data['nodes']}
    edges = {row['id'] for row in data['edges']}
    return (
        _rows(arguments['temperature_K'], nodes, 'node_id', ['value_K'], complete=True, positive=True),
        _rows(arguments['expansion_alpha_per_K'], edges, 'edge_id', ['value_per_K'], complete=True),
        _rows(arguments['constraints'], nodes, 'node_id', ['ux_m', 'uy_m', 'theta_rad'], nullable=True),
        _rows(arguments['loads'], nodes, 'node_id', ['Fx_N', 'Fy_N', 'Mz_Nm']))


def snapshot_request(graph_info, arguments, data, source_bytes):
    """Validate IDs/units/budgets before spawning; leave mechanics to native."""
    try:
        if not (2 <= graph_info['node_count'] <= 192 and 1 <= graph_info['edge_count'] <= 256):
            raise SnapshotContractError('SNAPSHOT_GRAPH_LIMIT')
        if source_bytes > MAX_SNAPSHOT_SOURCE_BYTES:
            raise SnapshotContractError('SNAPSHOT_SOURCE_LIMIT')
        keys(arguments, ARGUMENTS, ARGUMENTS)
        check_tree(arguments)
        if type(arguments['graph_id']) is not str or not re.fullmatch('[a-f0-9]{32}', arguments['graph_id']):
            raise ContractError('invalid handle')
        if arguments['length_unit'] not in ('m', 'mm', 'um') or arguments['length_unit'] != graph_info['length_unit']:
            raise SnapshotContractError('SNAPSHOT_UNIT_CONFLICT')
        number(arguments['E_Pa'], positive=True)
        number(arguments['stress_free_temperature_K'], positive=True)
        if (len(data['nodes']), len(data['edges'])) != (graph_info['node_count'], graph_info['edge_count']):
            raise ContractError('source count mismatch')
        _maps(arguments, data)
        if (type(graph_info['frame_id']) is not str or not re.fullmatch('[A-Za-z0-9_-]{1,64}', graph_info['frame_id'])
                or graph_info['frame_provenance'] not in ('unknown', 'operator_asserted')
                or graph_info['unit_provenance'] != 'operator_asserted'):
            raise ContractError('invalid operator binding')
        request = dict(arguments, **{key: graph_info[key] for key in BINDING})
        if len(canonical(request)) > MAX_SNAPSHOT_REQUEST_BYTES:
            raise SnapshotContractError('SNAPSHOT_REQUEST_LIMIT')
        return request
    except SnapshotContractError:
        raise
    except (ContractError, KeyError, TypeError, ValueError, OverflowError):
        raise SnapshotContractError('INVALID_THERMOELASTIC_INPUT') from None


def _array(value, dimensions, *, positive=False, boolean=False):
    if not dimensions:
        if boolean:
            if type(value) is not bool:
                raise ContractError('expected bool')
        else:
            number(value, positive=positive)
        return
    if type(value) is not list or len(value) != dimensions[0]:
        raise ContractError('native shape')
    for row in value:
        _array(row, dimensions[1:], positive=positive, boolean=boolean)


def validate_snapshot(value, data, request):
    """Check finite complete wire layout and input binding, not physics again."""
    try:
        keys(value, SNAPSHOT_FIELDS, SNAPSHOT_FIELDS)
        check_tree(value)
        nodes = sorted(row['id'] for row in data['nodes'])
        edges = sorted(data['edges'], key=lambda row: row['id'])
        eids = [row['id'] for row in edges]
        n, e = len(nodes), len(eids)
        for field, expected in [('node_ids', nodes), ('edge_ids', eids)]:
            if type(value[field]) is not list:
                raise ContractError('native IDs')
            for item in value[field]: integer(item)
            if value[field] != expected: raise ContractError('native IDs')
        if value['profile'] != PROFILE or value['numerical_regularization'] is not False:
            raise ContractError('native profile')
        if value['units'] != UNITS or value['interpolation'] != 'linear_endpoint_temperature_mean_axial/1':
            raise ContractError('native units/interpolation')
        for field in ('E_Pa', 'stress_free_temperature_K'):
            number(value[field], positive=True)
            if value[field] != request[field]: raise ContractError('native scalar binding')
        if value['length_unit'] != request['length_unit']:
            raise ContractError('native unit binding')
        temperature, alpha, constraints, loads = _maps(request, data)
        _array(value['temperature_K'], [n], positive=True)
        _array(value['expansion_alpha_per_K'], [e])
        if value['temperature_K'] != [temperature[i] for i in nodes] or value['expansion_alpha_per_K'] != [alpha[i] for i in eids]:
            raise ContractError('native field binding')
        for field in ('displacement', 'loads', 'prescribed_values', 'reactions'): _array(value[field], [n, 3])
        _array(value['constraint_mask'], [n, 3], boolean=True)
        prescribed = [constraints.get(i, [None, None, None]) for i in nodes]
        mask = [[x is not None for x in row] for row in prescribed]
        if (value['constraint_mask'] != mask or
                value['prescribed_values'] != [[0. if x is None else x for x in row] for row in prescribed] or
                value['loads'] != [loads.get(i, [0., 0., 0.]) for i in nodes]):
            raise ContractError('native boundary binding')
        free = [3*i+k for i, row in enumerate(mask) for k, fixed in enumerate(row) if not fixed]
        if type(value['free_dof_indices']) is not list: raise ContractError('native free IDs')
        for item in value['free_dof_indices']: integer(item)
        if value['free_dof_indices'] != free: raise ContractError('native free IDs')
        _array(value['free_residual'], [len(free)])
        _array(value['free_acceptance_bound'], [len(free)], positive=True)
        _array(value['component_balance'], [3])
        _array(value['component_acceptance_bound'], [3], positive=True)
        for field in ('total_axial_strain', 'thermal_axial_strain', 'elastic_axial_strain', 'sigma_axial', 'sigma_bending', 'sigma_total'): _array(value[field], [e])
        _array(value['moments'], [e, 2]); _array(value['edge_forces'], [e, 3])
        _array(value['positions_m'], [n, 2]); _array(value['radii_m'], [e], positive=True)
        xyz = [row['position'] for row in sorted(data['nodes'], key=lambda row: row['id'])]
        radii = [row['radius'] for row in edges]
        original = {'dimension': data['dimension'], 'positions_xyz': xyz, 'radii': radii,
                    'internal_points': [row.get('internal_points') for row in edges]}
        geometry=value['original_geometry']
        keys(geometry, set(original), set(original))
        if type(geometry['dimension']) is not int or geometry['dimension'] != 2: raise ContractError('native dimension')
        _array(geometry['positions_xyz'], [n,3]); _array(geometry['radii'], [e], positive=True)
        if type(geometry['internal_points']) is not list or len(geometry['internal_points']) != e:
            raise ContractError('native internal points')
        for points in geometry['internal_points']:
            if points is not None:
                if type(points) is not list: raise ContractError('native internal points')
                _array(points, [len(points),3])
        if geometry != original: raise ContractError('native source geometry')
        scale = {'m': 1., 'mm': .001, 'um': .000001}[request['length_unit']]
        if (value['positions_m'] != [[x*scale, y*scale] for x, y, z in xyz] or
                value['radii_m'] != [r*scale for r in radii]):
            raise ContractError('native SI geometry')
        for pair in value['endpoints_by_id']:
            if type(pair) is not list or len(pair) != 2: raise ContractError('native endpoints')
            for endpoint in pair: integer(endpoint)
        if value['endpoints_by_id'] != [[row['node_i'], row['node_j']] for row in edges]:
            raise ContractError('native incidence')
        integer(value['balance_origin_node_id'])
        roots = [nid for nid in nodes if constraints.get(nid) == [0., 0., 0.]]
        if value['balance_origin_node_id'] != (roots[0] if roots else nodes[0]):
            raise ContractError('native origin')
        identity = {key: value[key] for key in IDENTITY_FIELDS}
        if value['input_sha256'] != sha(canonical(identity)):
            raise ContractError('native input identity')
        a = value['applicability']
        keys(a, {'model','stability_qualified','fiber_signs','endpoint_order','edge_rows','ceilings'},
                {'model','stability_qualified','fiber_signs','endpoint_order','edge_rows','ceilings'})
        if type(a['fiber_signs']) is not list or any(type(x) is not int for x in a['fiber_signs']):
            raise ContractError('native fiber signs')
        if (a['model'] != 'linear_planar_welded_Euler_Bernoulli' or a['stability_qualified'] is not False
                or a['fiber_signs'] != [-1, 1] or a['endpoint_order'] != ['node_i','node_j']
                or a['ceilings'] != {'thermal_strain': 1e-5, 'mean_total_axial_strain': 1e-4,
                                    'rotation_and_slope_rad': 1e-3, 'extreme_fiber_strain': 1e-4}):
            raise ContractError('native applicability contract')
        if type(a['edge_rows']) is not list or len(a['edge_rows']) != e: raise ContractError('native applicability rows')
        fields = {'transverse_polynomial_normalized_x_m','slope_sample_x_over_L','slope_rad',
                  'transverse_chord_slope_rad','endpoint_curvature_per_m','endpoint_thermal_strain',
                  'endpoint_total_fiber_strain','endpoint_elastic_fiber_strain'}
        for row in a['edge_rows']:
            keys(row, fields, fields)
            locations = row['slope_sample_x_over_L']
            if type(locations) is not list or len(locations) not in (2, 3): raise ContractError('native sample count')
            _array(locations, [len(locations)])
            if locations[0] != 0 or locations[-1] != 1 or any(a >= b for a,b in zip(locations,locations[1:])):
                raise ContractError('native sample positions')
            _array(row['slope_rad'], [len(locations)])
            _array(row['transverse_polynomial_normalized_x_m'], [4])
            number(row['transverse_chord_slope_rad'])
            for field in ('endpoint_curvature_per_m', 'endpoint_thermal_strain'): _array(row[field], [2])
            for field in ('endpoint_total_fiber_strain', 'endpoint_elastic_fiber_strain'): _array(row[field], [2,2])
        if len(canonical(value)) > MAX_SNAPSHOT_RESULT_BYTES:
            raise SnapshotContractError('SNAPSHOT_RESULT_LIMIT')
    except SnapshotContractError:
        raise
    except (ContractError, KeyError, TypeError, ValueError, OverflowError):
        raise SnapshotContractError('NATIVE_SNAPSHOT_INVARIANT') from None


def validate_worker_snapshot(result, data, request, source_sha256, core_provenance):
    binding = {'source_sha256', 'request_sha256', '_core_provenance'}
    error = 'error_code' in result
    required = binding | ({'error_code','error_category'} if error else {'snapshot'})
    try:
        keys(result, required, required)
        if canonical(result['_core_provenance']) != canonical(core_provenance):
            raise ContractError('worker core binding')
        if result['source_sha256'] != source_sha256 or result['request_sha256'] != sha(canonical(request)):
            raise ContractError('worker request binding')
        if error:
            if result['error_code'] not in _ERROR_CATEGORIES or result['error_category'] != _ERROR_CATEGORIES[result['error_code']]:
                raise ContractError('worker error contract')
        else:
            validate_snapshot(result['snapshot'], data, request)
    except SnapshotContractError:
        raise
    except (ContractError, KeyError, TypeError, ValueError):
        raise SnapshotContractError('NATIVE_SNAPSHOT_INVARIANT') from None


def solve_snapshot(data, request, source_sha256):
    """The sole physics call; original native snapshot is returned unmodified."""
    binding = {'source_sha256': source_sha256, 'request_sha256': sha(canonical(request))}
    try:
        keys(request, ARGUMENTS | BINDING, ARGUMENTS | BINDING)
        info = dict(node_count=len(data['nodes']), edge_count=len(data['edges']),
                    length_unit=request['length_unit'], **{key:request[key] for key in BINDING})
        if snapshot_request(info, {key:request[key] for key in ARGUMENTS}, data, len(canonical(data))) != request:
            raise SnapshotContractError('INVALID_THERMOELASTIC_INPUT')
        from fibernet.core.structure_graph import StructureGraph
        from fibernet.sim.thermoelastic import solve_thermoelastic_network_snapshot
        graph = StructureGraph.from_dict(data)
        ordered = dict(data, nodes=sorted(data['nodes'], key=lambda row:row['id']),
                       edges=sorted(data['edges'], key=lambda row:row['id']))
        if (graph.to_dict() != ordered or list(graph.nodes) != [r['id'] for r in data['nodes']]
                or list(graph.edges) != [r['id'] for r in data['edges']]):
            raise SnapshotContractError('SNAPSHOT_SOURCE_ROUNDTRIP')
        temperature, alpha, constraints, loads = _maps(request, data)
        result = solve_thermoelastic_network_snapshot(graph, length_unit=request['length_unit'],
            E_Pa=request['E_Pa'], stress_free_temperature_K=request['stress_free_temperature_K'],
            temperature_K=temperature, expansion_alpha_per_K=alpha,
            constraints=constraints, loads=loads).to_dict()
        validate_snapshot(result, data, request)
        return dict(binding, snapshot=result)
    except SnapshotContractError as error:
        return dict(binding, error_code=error.code, error_category=error.category)
    except ImportError:
        return dict(binding, error_code='CORE_DEPENDENCY_UNAVAILABLE', error_category='internal')
    except (ValueError, TypeError, KeyError, ArithmeticError, Warning):
        return dict(binding, error_code='SNAPSHOT_DOMAIN_REJECTED', error_category='domain')
