"""Small synthetic serialization fixtures; no native simulation/authorship claim."""
import math
from fibernet.core.structure_graph import StructureGraph
from fibernet.sim.archived_ge import SI, raw_json, sha

def fixture():
    source = dict(version='3.1', dimension=3, tolerance=1e-09, next_node_id=92, next_edge_id=78, box_size=None, metadata={}, nodes=[dict(id=91, position=[1.0, 0.0, 0.0], boundary=[False] * 6, metadata={}), dict(id=11, position=[0.0, 0.0, 0.0], boundary=[True] * 6, metadata={})], edges=[dict(id=77, node_i=91, node_j=11, radius=0.01, segments=1, internal_points=None, metadata={}), dict(id=5, node_i=11, node_j=91, radius=0.01, segments=1, internal_points=None, metadata={})])
    g = StructureGraph.from_dict(source)
    source = g.to_dict()
    refs = {}
    ltg = {}
    for j, (nid, pos, angle) in enumerate([(10, [0.0, 0.0, 0.0], 0.0), (20, [1.0, 0.0, 0.0], 0.0), (30, [0.0, 0.0, 0.0], math.pi), (40, [1.0, 0.0, 0.0], math.pi)]):
        refs[str(nid)] = dict(nodeType='RigidBodyRxyz', referenceCoordinates=pos + [0.0, 0.0, angle])
        ltg[str(nid)] = list(range(6 * j, 6 * j + 6))
    beams = [dict(parent_edge_id=eid, segment_index=0, segments=1, nodeNumbers=ns, length=1.0, radius_m=0.01, section_id='explicit-test-section', sectionData={'massPerLength': 1.0, 'stiffnessMatrix': [[1.0 if i == j else 0.0 for j in range(6)] for i in range(6)], 'inertia': [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)], 'dampingMatrix': [[0.0] * 6 for i in range(6)]}) for eid, ns in [(5, [10, 20]), (77, [30, 40])]]
    p = dict(model='fibernet.new-ge-reference-lobatto-contact/1', subdivisions=1, new_nodes=[], source_graph=source, beams=beams, retained_markers=[], retained_objects=[], contact_sites=[dict(parent_edge_id=e, station=j, node_id=([10, 20] if e == 5 else [30, 40])[j], reference_weight_m=0.5) for e in [5, 77] for j in [0, 1]])
    b = dict(reference_frames=refs, layout={'node_ltg_ode2': ltg}, assembly={'reference_plan': p, 'beams': [dict(parent_edge_id=e, segment=0, object=i) for i, e in enumerate([5, 77])], 'retained_object_map': {}, 'node_map': {'10': 0, '20': 1, '30': 2, '40': 3}})
    h = dict(schema='fibernet.new-nodal-contact-state-trace/1', binding=b, ode2_size=24, ae_size=0, runtime_identity={'provider': 'test-only-no-native'}, static_metadata={'method': 'test-GA'})
    s = dict(schema='fibernet.complete-accepted-state/1', header_sha256=sha(raw_json(h)), step=1, time_s=0.001, q=[0.0] * 24, v=[0.0] * 24, a=[0.0] * 24, ae=[], diagnostics=dict(algorithmic_acceleration=[2.0] * 24, start_of_step_algorithmic_acceleration=[1.0] * 24, data_coordinates=[], ode1_coordinates=[], newton_converged=True, discontinuous_success=True, linear_solver_failed=False))
    return (g, source, h, s)

def arguments(source, h, s):
    gb, hb, sb = map(raw_json, (source, h, s))
    sections = [dict(parent_edge_id=v['parent_edge_id'], segment_index=v['segment_index'], section_id=v['section_id'], radius_m=v['radius_m'], sectionData=v['sectionData']) for v in h['binding']['assembly']['reference_plan']['beams']]
    return dict(source_graph_bytes=gb, header_bytes=hb, state_bytes=sb, expected_sha256=dict(graph=sha(gb), header=sha(hb), state=sha(sb)), section_override=dict(policy='archived-native-sectionData-overrides-source-Material', reason='Explicit synthetic archived section, not derived from placeholder graph Material', units=SI, source_graph_sha256=sha(gb), binding_sha256=sha(raw_json(h['binding'])), native_sections_sha256=sha(raw_json(sections))), units=SI, occurrence_tolerance_m=1e-12)
