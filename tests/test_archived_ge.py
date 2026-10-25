"""Portable archived-n1 projection contracts; no optional native provider."""
import unittest, json
from fibernet.core.structure_graph import StructureGraph
from fibernet.sim.archived_ge import import_archived_ge_state, raw_json, sha, SI
from _archived_ge_fixtures import fixture, arguments

class Tests(unittest.TestCase):

    def test_sparse_parallel_reordered_reversed_frames(self):
        g, a, h, s = fixture()
        g._nodes = dict(reversed(list(g.nodes.items())))
        g._edges = dict(reversed(list(g.edges.items())))
        r = import_archived_ge_state(g, **arguments(a, h, s))
        self.assertEqual(r.translations()['node_ids'], [11, 91])
        self.assertEqual(r.translations()['edge_ids'], [5, 77])
        self.assertFalse(r.sidecar()['edge_mapping']['77']['orientation_matches_source'])
        self.assertEqual(r.sidecar()['node_mapping']['11'], [10, 30])
        self.assertNotEqual(r.sidecar()['binding']['reference_frames']['10'], r.sidecar()['binding']['reference_frames']['30'])
        self.assertEqual(r.sidecar()['clamps'], [])

    def test_material_mutation_rejected(self):
        g, a, h, s = fixture()
        g.edges[5].material.youngs_modulus += 1
        self.assertRaises(ValueError, import_archived_ge_state, g, **arguments(a, h, s))

    def test_frame_hash_mutation_rejected(self):
        g, a, h, s = fixture()
        kw = arguments(a, h, s)
        h['binding']['reference_frames']['10']['referenceCoordinates'][3] = 0.1
        kw['header_bytes'] = raw_json(h)
        self.assertRaises(ValueError, import_archived_ge_state, g, **kw)

    def test_units_and_override_rejected(self):
        g, a, h, s = fixture()
        kw = arguments(a, h, s)
        kw['units'] = {**SI, 'length': 'mm'}
        self.assertRaises(ValueError, import_archived_ge_state, g, **kw)
        kw = arguments(a, h, s)
        kw['section_override']['native_sections_sha256'] = '0' * 64
        self.assertRaises(ValueError, import_archived_ge_state, g, **kw)

    def test_curves_and_twist_rejected(self):
        for kind in ['curve', 'twist']:
            g, a, h, s = fixture()
            if kind == 'curve':
                a['edges'][0]['internal_points'] = [[0.5, 0.1, 0.0]]
                g = StructureGraph.from_dict(a)
            else:
                h['binding']['reference_frames']['20']['referenceCoordinates'][3] = 0.1
            s['header_sha256'] = sha(raw_json(h))
            self.assertRaises(ValueError, import_archived_ge_state, g, **arguments(a, h, s))

    def test_weld_disagreement_rejected_not_averaged(self):
        g, a, h, s = fixture()
        s['q'][12] = 1e-06
        self.assertRaises(ValueError, import_archived_ge_state, g, **arguments(a, h, s))

    def test_missing_or_phase_wrong_algorithmic_rejected(self):
        for field in ['algorithmic_acceleration', 'start_of_step_algorithmic_acceleration']:
            g, a, h, s = fixture()
            s['diagnostics'][field] = None
            self.assertRaises(ValueError, import_archived_ge_state, g, **arguments(a, h, s))

    def test_detached_bytes_preserve_native_state(self):
        g, a, h, s = fixture()
        s['q'][3] = -0.0
        r = import_archived_ge_state(g, **arguments(a, h, s))
        s['q'][3] = 7.0
        x = r.sidecar()
        self.assertEqual(x['native_state']['q'][3].hex(), (-0.0).hex())
        x['native_state']['q'][3] = 9.0
        self.assertEqual(r.sidecar()['native_state']['q'][3].hex(), (-0.0).hex())
        self.assertEqual(r.original_state_bytes, arguments(a, h, json.loads(r.original_state_bytes))['state_bytes'])

def linked():
    g, a, h, s = fixture()
    h.update(schema='fibernet.linked-nodal-contact-state-trace/1', initial_global_step=100, initial_absolute_time_s=0.1, parent_link={'state_sha256': 'parent-token'}, clock=dict(absolute_time=True, step_size_s=0.001, absolute_time_tolerance_s=1e-15))
    s.update(schema='fibernet.linked-complete-accepted-state/1', step=101, global_step=101, local_step=1, time_s=0.101, anchor=False, parent_state_sha256='parent-token')
    return (g, a, h, s)

def call(g, a, h, s):
    s['header_sha256'] = sha(raw_json(h))
    return import_archived_ge_state(g, **arguments(a, h, s))

class PhaseTests(unittest.TestCase):

    def test_valid_linked_nonanchor_and_true_anchor(self):
        g, a, h, s = linked()
        self.assertEqual(call(g, a, h, s).translations()['global_step'], 101)
        s.update(step=100, global_step=100, local_step=0, time_s=0.1, anchor=True)
        s['diagnostics']['start_of_step_algorithmic_acceleration'] = None
        self.assertEqual(call(g, a, h, s).translations()['global_step'], 100)

    def test_resealed_anchor_bypass_rejected(self):
        for value in [True, 'yes', 1]:
            g, a, h, s = linked()
            s['anchor'] = value
            s['diagnostics']['newton_converged'] = False
            s['diagnostics']['start_of_step_algorithmic_acceleration'] = None
            self.assertRaises(ValueError, call, g, a, h, s)
        g, a, h, s = fixture()
        s['anchor'] = True
        self.assertRaises(ValueError, call, g, a, h, s)

    def test_resealed_clock_and_indices_rejected(self):
        for field, value in [('step_size_s', 0.0), ('step_size_s', -0.001), ('step_size_s', True), ('absolute_time_tolerance_s', 1.0), ('absolute_time_tolerance_s', -1.0)]:
            g, a, h, s = linked()
            h['clock'][field] = value
            self.assertRaises(ValueError, call, g, a, h, s)
        for field, value in [('local_step', -1), ('local_step', True), ('time_s', -0.1)]:
            g, a, h, s = linked()
            s[field] = value
            self.assertRaises(ValueError, call, g, a, h, s)

class PublicShapeTests(unittest.TestCase):
    def check_rejected(self, mutate):
        graph, source, header, state = fixture()
        mutate(source, header, state)
        state['header_sha256'] = sha(raw_json(header))
        with self.assertRaises(ValueError):
            import_archived_ge_state(graph, **arguments(source, header, state))

    def test_schema_and_missing_fields(self):
        for mutate in [
            lambda a,h,s: h.update(schema='unknown'),
            lambda a,h,s: s.update(schema='unknown'),
            lambda a,h,s: h.pop('ode2_size'),
            lambda a,h,s: h.update(ode2_size='24'),
            lambda a,h,s: h.update(ae_size=200001),
        ]:
            self.check_rejected(mutate)

    def test_colliding_ltg_and_native_ids(self):
        for mutate in [
            lambda a,h,s: h['binding']['layout']['node_ltg_ode2'].update({'20':[0,1,2,3,4,5]}),
            lambda a,h,s: h['binding']['assembly']['node_map'].update({'20':0}),
            lambda a,h,s: h['binding']['assembly']['beams'][1].update(object=0),
        ]:
            self.check_rejected(mutate)

    def test_unmapped_clamp_is_not_exported_as_null(self):
        def bad(a,h,s):
            p=h['binding']['assembly']['reference_plan']
            p['retained_markers']=[{'old_id':7,'fields':{'nodeNumber':999,'coordinate':0}}]
            p['retained_objects']=[{'old_id':8,'fields':{'objectType':'ConnectorCoordinate','activeConnector':True,'markerNumbers':[6,7]}}]
            h['binding']['assembly']['retained_object_map']={'8':4}
        self.check_rejected(bad)

    def test_reference_and_section_shapes(self):
        for mutate in [
            lambda a,h,s: h['binding']['reference_frames']['10'].update(referenceCoordinates=[0,0,0]),
            lambda a,h,s: h['binding']['assembly']['reference_plan']['beams'][0]['sectionData'].update(inertia=[[1.]]),
            lambda a,h,s: h['binding']['assembly']['reference_plan']['beams'][0]['sectionData'].update(massPerLength=0.),
        ]:
            self.check_rejected(mutate)

    def test_graph_type_and_bad_bytes(self):
        g,a,h,s=fixture();kw=arguments(a,h,s)
        with self.assertRaises(TypeError):import_archived_ge_state(object(),**kw)
        kw['header_bytes']='not bytes'
        with self.assertRaises(ValueError):import_archived_ge_state(g,**kw)

    def test_source_immutability(self):
        g,a,h,s=fixture();before=raw_json(g.to_dict());kw=arguments(a,h,s)
        result=import_archived_ge_state(g,**kw)
        self.assertEqual(raw_json(g.to_dict()),before)
        self.assertEqual(result.original_header_bytes,kw['header_bytes'])
        self.assertEqual(result.original_state_bytes,kw['state_bytes'])

    def test_duplicate_json_keys_and_nonfinite_constants(self):
        g,a,h,s=fixture();kw=arguments(a,h,s)
        for raw in [b'{"schema":"a","schema":"b"}',b'{"x":NaN}']:
            bad=dict(kw);bad['header_bytes']=raw;bad['expected_sha256']={**kw['expected_sha256'],'header':sha(raw)}
            with self.assertRaises(ValueError):import_archived_ge_state(g,**bad)

    def test_projection_overflow_is_a_documented_value_error(self):
        g,a,h,s=fixture();kw=arguments(a,h,s);kw['occurrence_tolerance_m']=10**400
        with self.assertRaises(ValueError):import_archived_ge_state(g,**kw)
        g,a,h,s=linked();h['clock']['step_size_s']=10**400;s['header_sha256']=sha(raw_json(h))
        with self.assertRaises(ValueError):import_archived_ge_state(g,**arguments(a,h,s))
