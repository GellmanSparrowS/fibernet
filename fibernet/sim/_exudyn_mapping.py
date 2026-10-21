"""Pure SI compilation of a straight shared-node chain from StructureGraph.

Circular-section stiffness and mass are derived from explicit SI material data.
No mechanics solver, contact evaluator, native provider or graph regeneration.
Expanded-endpoint/welded graphs are outside this preparation profile.
"""
import math

from fibernet.core.structure_graph import StructureGraph
from fibernet.units import LENGTH_FACTORS
from ._exudyn_contract import (
    COMPILER, PROFILE, ExudynContactRequest, ExudynContractError,
    _canonical, _decode, _fail, _id, _real,
)


def _snapshot_graph(graph):
    if not isinstance(graph, StructureGraph):
        raise TypeError('graph must be StructureGraph')
    if not isinstance(graph.nodes, dict) or not isinstance(graph.edges, dict):
        _fail('invalid_graph', 'graph nodes and edges must be dictionaries')
    if not 2 <= len(graph.nodes) <= 257 or not 1 <= len(graph.edges) <= 256:
        _fail('invalid_graph', 'straight-chain preparation admits 2..257 nodes and 1..256 edges')
    for key, node in graph.nodes.items():
        nid = _id(key)
        if _id(getattr(node, 'node_id', None)) != nid:
            _fail('invalid_id', 'node dictionary key and node_id disagree', nid)
    for key, edge in graph.edges.items():
        eid = _id(key)
        if _id(getattr(edge, 'edge_id', None)) != eid:
            _fail('invalid_id', 'edge dictionary key and edge_id disagree', eid)
        _id(getattr(edge, 'node_i', None))
        _id(getattr(edge, 'node_j', None))
    try:
        snapshot = graph.to_dict()
    except (ValueError, TypeError, AttributeError, OverflowError) as exc:
        raise ExudynContractError('invalid_graph', 'graph cannot produce valid source provenance') from exc
    # Preserve numeric IDs, including signed in-memory IDs, without from_dict().
    snapshot['next_node_id'] = _id(snapshot['next_node_id'], 'next node ID')
    snapshot['next_edge_id'] = _id(snapshot['next_edge_id'], 'next edge ID')
    snapshot['tolerance'] = _real(snapshot['tolerance'], 'graph tolerance', nonnegative=True)
    for node in snapshot['nodes']:
        node['id'] = _id(node['id'])
    for edge in snapshot['edges']:
        edge['radius'] = _real(edge['radius'], 'edge radius', positive=True)
        for field in ('id', 'node_i', 'node_j'):
            edge[field] = _id(edge[field])
    return _decode(_canonical(snapshot))


def _position(value, nid):
    if type(value) is not list or len(value) != 3:
        _fail('unsupported_geometry', 'each node needs three coordinates', nid)
    p = tuple(_real(x, 'node position') for x in value)
    if p[1] != 0 or p[2] != 0 or abs(p[0]) > 1:
        _fail('unsupported_geometry', 'profile requires y=z=0 and |x|<=1 m', nid)
    return p


def _compile_section(edge, shear):
    eid = edge['id']
    radius = _real(edge.get('radius'), 'edge radius', positive=True)
    material = edge.get('material')
    if type(material) is not dict:
        _fail('unsupported_material', 'explicit serialized Material required', eid)
    if material.get('stiffness_tensor') is not None:
        _fail('unsupported_material', 'anisotropic stiffness is outside this profile', eid)
    E = _real(material.get('youngs_modulus'), 'Young modulus', positive=True)
    G = _real(material.get('shear_modulus'), 'shear modulus', positive=True)
    rho = _real(material.get('density'), 'density', positive=True)
    nu = _real(material.get('poissons_ratio'), 'Poisson ratio')
    if not -1 < nu < .5 or not math.isclose(G, E / (2 * (1 + nu)), rel_tol=1e-12, abs_tol=0):
        _fail('unsupported_material', 'inconsistent isotropic E/G/nu', eid)
    try:
        area = math.pi * radius**2
        inertia = math.pi * radius**4 / 4
        polar = 2 * inertia
        stiffness = [E*area, shear[0]*G*area, shear[1]*G*area,
                     G*polar, E*inertia, E*inertia]
        mass_per_length = rho*area
        rotational_inertia = [rho*polar, rho*inertia, rho*inertia]
    except OverflowError as exc:
        raise ExudynContractError('nonfinite', 'section intermediate overflow', eid) from exc
    values = [area, inertia, polar, mass_per_length] + stiffness + rotational_inertia
    if not all(math.isfinite(x) and x > 0 for x in values):
        _fail('unsupported_material', 'derived section is nonpositive/nonfinite', eid)
    return {'source_edge_id': eid, 'radius_m': radius, 'area_m2': area,
            'Iy_m4': inertia, 'Iz_m4': inertia, 'J_m4': polar,
            'stiffness_diagonal_SI': stiffness,
            'mass_per_length_kg_m': mass_per_length,
            'inertia_per_length_diagonal_kg_m': rotational_inertia,
            'shear_correction': list(shear), 'material': material,
            'section_damping': 'zero', 'prestress': 'none'}


def _compile_snapshot(source, request):
    if type(request) is not ExudynContactRequest:
        raise TypeError('request must be ExudynContactRequest')
    if type(source) is not dict or source.get('version') != '3.1' or source.get('dimension') not in (2, 3):
        _fail('invalid_graph', 'a StructureGraph schema-3.1 source record is required')
    nodes, edges = source.get('nodes'), source.get('edges')
    if type(nodes) is not list or type(edges) is not list or not 2 <= len(nodes) <= 257 or not 1 <= len(edges) <= 256:
        _fail('invalid_graph', 'invalid or out-of-scope graph counts')
    if any(type(row) is not dict for row in nodes + edges):
        _fail('invalid_graph', 'all source node/edge records must be objects')
    positions = {}
    for node in nodes:
        if type(node) is not dict:
            _fail('invalid_graph', 'invalid node record')
        nid = _id(node.get('id'))
        if nid in positions:
            _fail('invalid_id', 'duplicate node ID', nid)
        positions[nid] = _position(node.get('position'), nid)
    tolerance = _real(source.get('tolerance'), 'graph tolerance', nonnegative=True)
    box = source.get('box_size')
    if box is not None:
        if type(box) is not list or len(box) != 3:
            _fail('invalid_graph', 'box_size must be null or three finite coordinates')
        for coordinate in box:
            _real(coordinate, 'box_size')
    # The existing graph-JSON reader imposes independent allocator invariants.
    for kind, rows in (('node', nodes), ('edge', edges)):
        counter = _id(source.get('next_' + kind + '_id'), 'next ' + kind + ' ID')
        maximum = max(_id(row.get('id')) for row in rows if type(row) is dict)
        if counter < max(0, maximum + 1):
            _fail('invalid_graph', 'next ' + kind + ' ID must be nonnegative and exceed stored IDs')
    # Explicitly reuse the core SI length factor, without a permissive fallback.
    if request.length_unit != 'm' or LENGTH_FACTORS['m'] != 1.0:
        _fail('invalid_units', 'strict SI graph length contract required')
    chain = sorted(positions, key=lambda n: positions[n][0])
    if any(positions[a][0] >= positions[b][0] for a, b in zip(chain, chain[1:])):
        _fail('unsupported_geometry', 'distinct ordered chain coordinates are required')
    if set(request.clamped_node_ids) != {chain[0], chain[-1]}:
        _fail('invalid_boundary', 'clamps must identify the two physical chain ends')
    if len(edges) != len(nodes)-1:
        _fail('unsupported_geometry', 'profile requires a simple path; no branches/cycles/parallel edges')
    ordinals = {nid: i for i, nid in enumerate(chain)}
    edge_by_pair = {}
    edge_ids = set()
    for edge in edges:
        if type(edge) is not dict:
            _fail('invalid_graph', 'invalid edge record')
        eid, a, b = (_id(edge.get(k)) for k in ('id', 'node_i', 'node_j'))
        if eid in edge_ids:
            _fail('invalid_id', 'duplicate edge ID', eid)
        edge_ids.add(eid)
        if a not in positions or b not in positions or a == b:
            _fail('invalid_graph', 'edge incidence is invalid', eid)
        if type(edge.get('segments')) is not int or edge['segments'] != 1:
            _fail('unsupported_refinement', 'require explicit segments=1; no implicit resampling', eid)
        if edge.get('internal_points') is not None:
            _fail('unsupported_geometry', 'internal points/curves are not supported', eid)
        pair = frozenset((a, b))
        if pair in edge_by_pair:
            _fail('unsupported_geometry', 'parallel physical edges are outside the chain profile', eid)
        if abs(ordinals[a]-ordinals[b]) != 1:
            _fail('unsupported_geometry', 'every edge must connect adjacent positions on one path', eid)
        edge_by_pair[pair] = edge
    expected_pairs = [frozenset((a, b)) for a, b in zip(chain, chain[1:])]
    if set(edge_by_pair) != set(expected_pairs):
        _fail('unsupported_geometry', 'path is disconnected or contains an isolated node')
    nq = 6 * len(chain) + 3
    sphere_dofs = list(range(nq-3, nq))
    node_records = []
    translation_dofs, rotation_dofs = [], []
    for ordinal, nid in enumerate(chain):
        translations = list(range(6*ordinal, 6*ordinal+3))
        rotations = list(range(6*ordinal+3, 6*ordinal+6))
        translation_dofs.extend(translations)
        rotation_dofs.extend(rotations)
        node_records.append({'source_node_id': nid, 'compilation_node_ordinal': ordinal,
                             'reference_position_m': list(positions[nid]),
                             'reference_rxyz_rad': [0., 0., 0.],
                             'translation_dofs': translations, 'rxyz_dofs': rotations,
                             'native_node_index': None})
    sections, beams = [], []
    mass_terms = []
    for ordinal, (left, right, pair) in enumerate(zip(chain, chain[1:], expected_pairs)):
        edge = edge_by_pair[pair]
        eid = edge['id']
        length = positions[right][0] - positions[left][0]
        if not 1e-6 <= length <= .01:
            _fail('unsupported_geometry', 'reference element length must be in [1 um, 10 mm]', eid)
        section = _compile_section(edge, request.shear_correction)
        sections.append(section)
        mass = section['mass_per_length_kg_m'] * length
        radius = section['radius_m'] + request.sphere.radius_m
        if not math.isfinite(mass) or mass <= 0 or not math.isfinite(radius):
            _fail('nonfinite', 'derived reference mass/exclusion radius is invalid', eid)
        mass_terms.append(mass)
        orientation = 1 if edge['node_i'] == left else -1
        beams.append({'source_edge_id': eid, 'compilation_beam_ordinal': ordinal,
                      'compilation_node_ordinals': [ordinal, ordinal+1],
                      'source_node_ids_native_order': [left, right],
                      'source_orientation': orientation,
                      'source_coordinate_at_native_endpoints': [0., 1.] if orientation == 1 else [1., 0.],
                      'material_interval': [0., 1.], 'reference_length_m': length,
                      'reference_mass_kg': mass, 'section_ordinal': ordinal,
                      'contact_exclusion_radius_m': radius,
                      'contact_dofs': list(range(6*ordinal, 6*ordinal+12))+sphere_dofs,
                      'contact_mass_matrix': 'zero_15_by_15', 'native_object_index': None})
    try:
        fiber_mass = math.fsum(mass_terms)
    except OverflowError as exc:
        raise ExudynContractError('nonfinite', 'reference mass sum overflow') from exc
    if not math.isfinite(fiber_mass + request.sphere.mass_kg):
        _fail('nonfinite', 'total reference mass overflow')
    clamp_dofs = list(range(6)) + list(range(6*(len(chain)-1), 6*len(chain)))
    steps = sum(p.steps for p in request.phases)
    records = steps + len(request.phases)  # all accepted states and phase-start records
    return {'schema': 'fibernet.exudyn_shared_chain_map/1', 'compiler': COMPILER,
            'profile': PROFILE, 'units': {'length': 'm', 'time': 's', 'mass': 'kg',
                                        'force': 'N', 'moment': 'N*m', 'rotation': 'rad'},
            'node_ids': sorted(positions), 'edge_ids': sorted(edge_ids), 'chain_order': chain,
            'nodes': node_records, 'beams': beams, 'sections': sections,
            'counts': {'physical_nodes': len(nodes), 'physical_edges': len(edges),
                       'beam_nodes': len(nodes), 'sphere_nodes': 1, 'ODE2': nq, 'AE': 12, 'welds': 0},
            'sphere': {'compilation_node_ordinal': len(chain), 'translation_dofs': sphere_dofs,
                       'reference_position_m': list(request.sphere.center_m),
                       'initial_velocity_m_s': list(request.sphere.velocity_m_s),
                       'radius_m': request.sphere.radius_m, 'mass_kg': request.sphere.mass_kg,
                       'native_node_index': None, 'native_object_index': None},
            'dof_blocks': {'fiber_translation': translation_dofs, 'fiber_rxyz': rotation_dofs,
                           'sphere_translation': sphere_dofs},
            'constraints': {'clamped_node_ids': list(request.clamped_node_ids),
                            'zero_coordinate_dofs': clamp_dofs, 'frame': 'world + zero reference Rxyz',
                            'algebraic_equations': 12},
            'loads': [], 'gravity_m_s2': list(request.gravity_m_s2),
            'reference_fiber_mass_kg': fiber_mass,
            'total_reference_mass_kg': fiber_mass + request.sphere.mass_kg,
            'source_graph_json_roundtrip_supported': all(i >= 0 for i in list(positions)+list(edge_ids)),
            'forecast': {'accepted_states': steps, 'phase_start_records': len(request.phases),
                         'raw_state_records': records,
                         'q_qt_qdd_AE_float64_payload_bytes': records*8*(3*nq+12),
                         'one_dense_mass_matrix_float64_bytes': 8*nq*nq,
                         'scope': 'named numeric payloads only; excludes headers/contact/diagnostics/trace/metadata; not a total memory or storage estimate'},
            'execution_available': False, 'restart_supported': False,
            'qualification': 'source_supported_not_natively_qualified'}


def compile_straight_chain(graph, request):
    """Return detached derived records; prepare_exudyn_contact adds identities."""
    return _compile_snapshot(_snapshot_graph(graph), request)
