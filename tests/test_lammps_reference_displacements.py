"""Synthetic result records only; no LAMMPS provider or solver."""
import copy,hashlib,json,unittest
import math
from decimal import Decimal, localcontext
from pathlib import Path
import subprocess
import struct
import sys
from types import SimpleNamespace
import numpy as np
from fibernet.core.structure_graph import StructureGraph
from fibernet.analysis.geometry_paths import compare_lammps_node_displacements as compare

def fixture():
    initial=StructureGraph(dimension=3);a=initial.add_node([1000.,0.,0.]);b=initial.add_node([2000.,0.,0.]);initial.add_edge(a,b)
    d=initial.to_dict();mapping={a:42,b:7}
    for n in d['nodes']:n['id']=mapping[n['id']]
    for e in d['edges']:e['node_i']=mapping[e['node_i']];e['node_j']=mapping[e['node_j']]
    d['nodes'].reverse();d.pop('next_node_id');initial=StructureGraph.from_dict(d)
    reference=initial.copy();reference.nodes[42].position=np.array([.5,0.,0.]);reference.nodes[7].position=np.array([1.5,0.,0.])
    digest=hashlib.sha256(json.dumps(initial.to_dict(),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    result=SimpleNamespace(metadata={'backend':'lammps','node_ids':[42,7], 'original_graph_length_unit':'um','original_graph_length_to_m':1e-6,'field_units':{'deformed_positions':'um'},'source_identity':{'graph_sha256':digest}},deformed_positions=np.array([[1100.,0.,0.],[2300.,0.,0.]]))
    return reference,initial,result

class DisplacementComparison(unittest.TestCase):
    def test_alignment_units_and_detachment(self):
        r,g,s=fixture();out=compare(r,g,s,reference_length_unit='mm',initial_length_unit='um')
        self.assertEqual(out['node_ids'],(7,42));self.assertEqual(out['units'],'m')
        np.testing.assert_allclose(out['from_initial_m'][:,0],[.0003,.0001],atol=1e-18)
        np.testing.assert_allclose(out['from_reference_m'][:,0],[.0008,.0006],atol=1e-18)
        out['from_initial_m'][0,0]=8;self.assertEqual(s.deformed_positions[1,0],2300.)
    def test_fails_closed_for_maps_units_identity_and_positions(self):
        cases=['missing','duplicate','foreign','unit','digest','masked','nonfinite','shape','record','mixed_bool']
        for case in cases:
            with self.subTest(case=case):
                r,g,s=fixture()
                if case=='missing':del s.metadata['node_ids']
                elif case=='duplicate':s.metadata['node_ids']=[42,42]
                elif case=='foreign':s.metadata['node_ids']=[42,88]
                elif case=='unit':s.metadata['field_units']['deformed_positions']='m'
                elif case=='digest':g.nodes[42].position[0]+=1
                elif case=='masked':s.deformed_positions=np.ma.array(s.deformed_positions,mask=False)
                elif case=='nonfinite':s.deformed_positions[0,0]=np.nan
                elif case=='shape':s.deformed_positions=np.ones((2,2))
                elif case=='record':r.nodes[42].node_id=9
                elif case=='mixed_bool':s.deformed_positions=[np.array([True,False,False]),[2300.,0.,0.]]
                with self.assertRaises((ValueError,TypeError)):compare(r,g,s,reference_length_unit='mm',initial_length_unit='um')
    def test_all_units_and_equal_reference_consistency(self):
        from fibernet.units import LENGTH_FACTORS
        for unit,scale in LENGTH_FACTORS.items():
            with self.subTest(unit=unit):
                _,g,s=fixture();s.metadata['original_graph_length_unit']=unit
                s.metadata['original_graph_length_to_m']=scale;s.metadata['field_units']['deformed_positions']=unit
                out=compare(g.copy(),g,s,reference_length_unit=unit,initial_length_unit=unit)
                np.testing.assert_array_equal(out['from_initial_m'],out['from_reference_m'])

    def test_elementwise_large_difference_fallback(self):
        _,g,s=fixture();g.nodes[42].position[0]=1e308;s.deformed_positions[0,0]=-1e308
        s.metadata['original_graph_length_unit']='mm';s.metadata['original_graph_length_to_m']=1e-3
        s.metadata['field_units']['deformed_positions']='mm'
        s.metadata['source_identity']['graph_sha256']=hashlib.sha256(json.dumps(g.to_dict(),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        out=compare(g,g,s,reference_length_unit='mm',initial_length_unit='mm')
        self.assertTrue(np.isfinite(out['from_initial_m']).all())
        self.assertAlmostEqual(out['from_initial_m'][1,0]/1e305,-2.)
        np.testing.assert_array_equal(out['from_initial_m'],out['from_reference_m'])

    def test_real_adapter_projection_from_synthetic_record(self):
        from fibernet.integrations import lammps_backend as b
        from fibernet.sim import _lammps_result as adapter
        r,g,_=fixture()
        p=b.prepare_harmonic_beads(g,length_unit='um',maximum_bond_length_m=.01,
            masses_kg={7:1.,42:1.},bonds={0:b.HarmonicBond(2.,.001)},
            run=b.RunSpec(1e-6,0,1,((-.01,.01),)*3))
        positions=tuple(tuple(g.nodes[i].position*1e-6) for i in p.node_ids)
        zeros=((0.,0.,0.),)*2
        frame=b.TrajectoryFrame(0,0.,p.node_ids,positions,zeros,zeros)
        thermo=b.ThermoFrame(0,0.,0.,0.,(0.,)*6)
        raw=b.LAMMPSRunResult(p.identity,(frame,),(thermo,),'')
        result=adapter.to_sim_result(p,raw,b.RuntimeSpec('cli_serial_static','/mock/python','/mock/lmp'),0.,False)
        out=compare(r,g,result,reference_length_unit='mm',initial_length_unit='um')
        np.testing.assert_allclose(out['from_initial_m'],0,atol=1e-18)
        np.testing.assert_allclose(out['from_reference_m'][:,0],[.0005,.0005],atol=1e-18)

    def test_cg_synthetic_detailed_save_load(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from fibernet.integrations import lammps_backend as b
        from fibernet.integrations._lammps_cg import CGRequest, CGAngle, MorsePair, prepare_cg
        from fibernet.sim import _lammps_result as adapter
        from fibernet.sim.accelerated import SimResult
        g=StructureGraph(dimension=3)
        for p in ((0,0,0),(1,0,0),(1,1,0),(0,0,2)):g.add_node(p)
        g.add_edge(0,1);g.add_edge(1,2)
        q=CGRequest('m',3,{n:1. for n in g.nodes},{e:b.HarmonicBond(2.,1.) for e in g.edges},
                    {9:CGAngle(0,1,2,0,1,3.)},3,{0:MorsePair(1.,2.,1.,3.)},b.RunSpec(.001,0,1,((-5.,5.),)*3))
        prepared=prepare_cg(g,q);points=tuple(tuple(g.nodes[i].position) for i in prepared.node_ids)
        zeros=((0.,0.,0.),)*4;frame=b.TrajectoryFrame(0,0.,prepared.node_ids,points,zeros,zeros)
        thermo=b.ThermoFrame(0,0.,0.,0.,(0.,)*6);raw=b.LAMMPSRunResult(prepared.identity,(frame,),(thermo,),'')
        result=adapter.to_sim_result(prepared,raw,b.RuntimeSpec('cli_serial_static','/mock/python','/mock/lmp'),0.,True)
        with TemporaryDirectory() as td:
            path=Path(td)/'detailed.json';result.save(str(path),detailed=True);loaded=SimResult.load(str(path))
            out=compare(g,g,loaded,reference_length_unit='m',initial_length_unit='m')
        self.assertEqual(out['node_ids'],(0,1,2,3));np.testing.assert_array_equal(out['from_initial_m'],np.zeros((4,3)))
        np.testing.assert_array_equal(out['from_initial_m'],out['from_reference_m'])

    def test_overflow_and_reference_id_mismatch(self):
        r,g,s=fixture();r.nodes[42].position[0]=-1e308;s.deformed_positions[0,0]=1e308
        # m-unit source produces overflow on subtraction, without touching input.
        s.metadata['field_units']['deformed_positions']='m';s.metadata['original_graph_length_unit']='m';s.metadata['original_graph_length_to_m']=1.
        with self.assertRaises(OverflowError):compare(r,g,s,reference_length_unit='m',initial_length_unit='m')
        r,g,s=fixture();r.nodes.pop(7)
        with self.assertRaises(ValueError):compare(r,g,s,reference_length_unit='mm',initial_length_unit='um')

class DisplacementComparisonIntegration(unittest.TestCase):
    def check_float32_source_cases(self, require_gradual=False):
        for negative in (False, True):
            for form in ('array', 'big_endian', 'row_arrays', 'mixed_scalars'):
                with self.subTest(negative=negative, form=form):
                    _, g, s = fixture()
                    for node in g.nodes.values():
                        node.position[:] = 0.
                    dtype = '>f4' if form == 'big_endian' else '=f4'
                    values = np.zeros((2, 3), dtype=dtype)
                    unsigned = np.dtype('u4').newbyteorder(values.dtype.byteorder)
                    values.view(unsigned)[0, 0] = 1 | (int(negative) << 31)
                    preserved = bool(values[0, 0] != 0.)
                    if require_gradual:
                        self.assertTrue(preserved, 'clean process must preserve float32 source')
                    if form == 'row_arrays':
                        s.deformed_positions = [values[0], values[1]]
                    elif form == 'mixed_scalars':
                        s.deformed_positions = [[values[0, 0], 0., 0.], [0., 0., 0.]]
                    else:
                        s.deformed_positions = values
                    s.metadata.update(original_graph_length_unit='m', original_graph_length_to_m=1.)
                    s.metadata['field_units']['deformed_positions'] = 'm'
                    s.metadata['source_identity']['graph_sha256'] = hashlib.sha256(
                        json.dumps(g.to_dict(), sort_keys=True, separators=(',', ':'),
                                   allow_nan=False).encode()).hexdigest()
                    if not preserved:
                        with self.assertRaisesRegex(OverflowError, 'representable'):
                            compare(g, g, s, reference_length_unit='m', initial_length_unit='m')
                    else:
                        out = compare(g, g, s, reference_length_unit='m', initial_length_unit='m')
                        expected = float.fromhex(('-' if negative else '')+'0x1p-149')
                        self.assertEqual(out['from_initial_m'][1, 0], expected)
                        np.testing.assert_array_equal(out['from_initial_m'], out['from_reference_m'])

    def check_subnormal_cases(self, require_gradual=False):
        self.check_float32_source_cases(require_gradual)
        for negative in (False, True):
            sign = '-' if negative else ''
            a = float.fromhex(sign+'0x1p-1022')
            b = float.fromhex(sign+'0x1.0000000000001p-1022')
            # float.fromhex can itself flush this fixture under FTZ.
            tiny = struct.unpack('>d', struct.pack('>Q', 1 | (int(negative) << 63)))[0]
            for case in ('same_unit', 'mixed_unit', 'stored_subnormal', 'converted_subnormal'):
                with self.subTest(case=case, negative=negative):
                    _, g, s = fixture()
                    if case == 'same_unit':
                        initial, final, reference, ref_unit = a, b, a, 'm'
                        probe = (np.array([b])-np.array([a]))*1.
                    elif case == 'mixed_unit':
                        initial, final, reference, ref_unit = b, b, a*1000., 'mm'
                        probe = np.array([b])-np.array([reference])*.001
                    elif case == 'stored_subnormal':
                        initial, final, reference, ref_unit = 0., 0., tiny, 'm'
                        probe = np.array([tiny])*1.
                    else:
                        initial = final = -1e-305 if negative else 1e-305
                        reference, ref_unit = a, 'm'
                        probe = np.array([final])*.001
                    g.nodes[42].position[0] = initial
                    s.deformed_positions[0, 0] = final
                    r = g.copy()
                    r.nodes[42].position[0] = reference
                    unit, factor = ('mm', .001) if case == 'converted_subnormal' else ('m', 1.)
                    s.metadata.update(original_graph_length_unit=unit, original_graph_length_to_m=factor)
                    s.metadata['field_units']['deformed_positions'] = unit
                    s.metadata['source_identity']['graph_sha256'] = hashlib.sha256(
                        json.dumps(g.to_dict(), sort_keys=True, separators=(',', ':'),
                                   allow_nan=False).encode()).hexdigest()
                    preserved = int(probe.view(np.uint64)[0]) & ((1 << 63)-1) == 1
                    if case == 'converted_subnormal':
                        # This also detects DAZ treating a stored nonzero value
                        # as zero before subtraction could yield a wrong nonzero.
                        preserved = bool(probe[0] != 0.)
                    if require_gradual:
                        self.assertTrue(preserved, 'clean process must preserve this subnormal')
                    if not preserved:
                        with self.assertRaisesRegex(OverflowError, 'representable'):
                            compare(r, g, s, reference_length_unit=ref_unit, initial_length_unit=unit)
                    else:
                        out = compare(r, g, s, reference_length_unit=ref_unit, initial_length_unit=unit)
                        expected_negative = not negative if case == 'stored_subnormal' else negative
                        expected_bits = 1 | (int(expected_negative) << 63)
                        if case == 'converted_subnormal':
                            with localcontext() as context:
                                context.prec = 800
                                scaled = float(Decimal(final)*Decimal(.001))
                                expected = float(Decimal(scaled)-Decimal(reference))
                            expected_bits = int(np.array([expected]).view(np.uint64)[0])
                        key = 'from_initial_m' if case == 'same_unit' else 'from_reference_m'
                        self.assertEqual(int(out[key].view(np.uint64)[1, 0]), expected_bits)

    def test_subnormal_loss_respects_active_runtime(self):
        self.check_subnormal_cases()

    def test_subnormal_positive_control_in_clean_process(self):
        root = Path(__file__).resolve().parents[1]
        code = (
            'import sys; '
            f'sys.path[:0] = [{str(root)!r}, {str(root / "tests")!r}]; '
            'from test_lammps_reference_displacements import DisplacementComparisonIntegration; '
            'DisplacementComparisonIntegration().check_subnormal_cases(require_gradual=True)'
        )
        completed = subprocess.run([sys.executable, '-c', code], capture_output=True,
                                   text=True, timeout=30, check=False)
        self.assertEqual(completed.returncode, 0, completed.stdout+completed.stderr)

    def test_all_unit_pairs_against_independent_si_values(self):
        # Deliberately independent of fibernet.units and the implementation.
        scales = {'m': 1., 'mm': .001, 'um': 1e-6, 'µm': 1e-6,
                  'nm': 1e-9, 'cm': .01, 'km': 1000., 'in': .0254,
                  'ft': .3048, 'Å': 1e-10, 'angstrom': 1e-10}
        for initial_unit, initial_scale in scales.items():
            for reference_unit, reference_scale in scales.items():
                with self.subTest(initial=initial_unit, reference=reference_unit):
                    r, g, s = fixture()
                    s.metadata.update(original_graph_length_unit=initial_unit,
                                      original_graph_length_to_m=initial_scale)
                    s.metadata['field_units']['deformed_positions'] = initial_unit
                    out = compare(r, g, s, reference_length_unit=reference_unit,
                                  initial_length_unit=initial_unit)
                    self.assertEqual(out['node_ids'], (7, 42))
                    with localcontext() as context:
                        context.prec = 80
                        a, b = Decimal(str(initial_scale)), Decimal(str(reference_scale))
                        expected = [float(Decimal(2300)*a-Decimal('1.5')*b),
                                    float(Decimal(1100)*a-Decimal('.5')*b)]
                    np.testing.assert_allclose(out['from_reference_m'][:, 0], expected,
                                               rtol=3e-15, atol=0.)
                    np.testing.assert_allclose(out['from_initial_m'][:, 0],
                                               [300*initial_scale, 100*initial_scale],
                                               rtol=3e-15, atol=0.)
                    np.testing.assert_array_equal(out['from_reference_m'][:, 1:], 0.)

    def test_parser_projection_and_edge_history_share_stable_ids(self):
        from fibernet.integrations import lammps_backend as b
        from fibernet.sim import _lammps_result as adapter
        from test_lammps_backend_contract import fixture as protocol_fixture
        r, g, _ = fixture()
        p = b.prepare_harmonic_beads(g, length_unit='um', maximum_bond_length_m=.01,
            masses_kg={7: 1., 42: 1.}, bonds={0: b.HarmonicBond(2., .001)},
            run=b.RunSpec(1e-6, 1, 1, ((-.01, .01),)*3))
        dump, thermo = protocol_fixture(p)
        # Change only the final synthetic dump frame. Raw atom rows are reversed.
        lines = dump.splitlines()
        header = max(i for i, line in enumerate(lines) if line.startswith('ITEM: ATOMS'))
        atom_to_node = {int(atom): node for atom, node in p.manifest()['atom_to_node_id'].items()}
        for i in range(header+1, len(lines)):
            fields = lines[i].split()
            node = atom_to_node[int(fields[0])]
            fields[2] = '.0023' if node == 7 else '.0011'
            lines[i] = ' '.join(fields)
        raw = b.parse_outputs(p, '\n'.join(lines)+'\n', thermo)
        result = adapter.to_sim_result(p, raw,
            b.RuntimeSpec('cli_serial_static', '/inert/python', '/inert/lmp'), 0., True)
        out = compare(r, g, result, reference_length_unit='mm', initial_length_unit='um')
        history = b.edge_length_history(p, raw, 0)
        self.assertEqual(out['node_ids'], (7, 42))
        np.testing.assert_allclose(out['from_initial_m'][:, 0], [.0003, .0001], rtol=3e-15)
        np.testing.assert_allclose(out['from_reference_m'][:, 0], [.0008, .0006], rtol=3e-15)
        self.assertTrue(math.isclose(history.lengths_m[-1], .0012, rel_tol=3e-15))
        self.assertEqual(result.metadata['field_units']['max_displacement'], 'um')
        self.assertTrue(math.isclose(result.max_displacement, 300., rel_tol=3e-15))
        self.assertEqual((history.node_i, history.node_j), (42, 7))

    def test_profile_bound_detailed_save_load_and_summary_rejection(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from fibernet.sim.accelerated import SimResult
        from fibernet.sim import _lammps_result as adapter
        from test_lammps_displacement_reference import prepared_case, typed_raw, RUNTIME
        for kind in ('harmonic', 'profile', 'cg'):
            with self.subTest(kind=kind), TemporaryDirectory() as directory:
                p = prepared_case(kind)
                g = StructureGraph.from_dict(json.loads(p.graph_json))
                reference = g.copy()
                for node in reference.nodes.values():
                    node.position[0] -= .5
                result = adapter.to_sim_result(p, typed_raw(p), RUNTIME, 0., True)
                path = Path(directory)/'result.json'
                profile = 'harmonic_sampled_v1' if kind == 'profile' else None
                result.save(path, detailed=True)
                loaded = SimResult.load(path, profile=profile)
                out = compare(reference, g, loaded,
                              reference_length_unit='mm', initial_length_unit='mm')
                np.testing.assert_allclose(out['from_initial_m'][:, 0], [0., .00002],
                                           rtol=3e-14, atol=0.)
                np.testing.assert_allclose(out['from_reference_m'][:, 0], [.0005, .00052],
                                           rtol=3e-15, atol=0.)
                if profile:
                    with self.assertRaises(ValueError):
                        SimResult.load(path)
                result.save(path, detailed=False)
                summary = SimResult.load(path, profile=profile)
                with self.assertRaises(ValueError):
                    compare(reference, g, summary,
                            reference_length_unit='mm', initial_length_unit='mm')

    def test_small_relative_changes_preserve_difference_before_scaling(self):
        for magnitude, unit, scale in ((1e-170, 'm', 1.), (1e100, 'nm', 1e-9),
                                       (1e308, 'mm', 1e-3)):
            with self.subTest(magnitude=magnitude, unit=unit):
                _, g, s = fixture()
                g.nodes[42].position[0] = magnitude
                final = math.nextafter(magnitude, math.inf)
                s.deformed_positions[0, 0] = final
                s.metadata.update(original_graph_length_unit=unit,
                                  original_graph_length_to_m=scale)
                s.metadata['field_units']['deformed_positions'] = unit
                s.metadata['source_identity']['graph_sha256'] = hashlib.sha256(
                    json.dumps(g.to_dict(), sort_keys=True, separators=(',', ':'),
                               allow_nan=False).encode()).hexdigest()
                out = compare(g, g, s, reference_length_unit=unit, initial_length_unit=unit)
                with localcontext() as context:
                    context.prec = 800
                    expected = float((Decimal(final)-Decimal(magnitude))*Decimal(scale))
                self.assertGreater(expected, 0.)
                self.assertEqual(out['from_initial_m'][1, 0], expected)
                np.testing.assert_array_equal(out['from_initial_m'], out['from_reference_m'])

    def test_reference_correspondence_is_explicit_not_inferred(self):
        r, g, s = fixture()
        # Matching IDs permit a caller-chosen frame; the helper does not register it.
        for node in r.nodes.values():
            node.position += np.array([1., 2., 3.])
        before = copy.deepcopy((r.to_dict(), g.to_dict(), s))
        out = compare(r, g, s, reference_length_unit='mm', initial_length_unit='um')
        np.testing.assert_allclose(out['from_reference_m'],
            [[-.0002, -.002, -.003], [-.0004, -.002, -.003]], rtol=3e-15, atol=0.)
        self.assertEqual((r.to_dict(), g.to_dict()), before[:2])
        np.testing.assert_array_equal(s.deformed_positions, before[2].deformed_positions)
        for key in ('from_initial_m', 'from_reference_m'):
            self.assertFalse(np.shares_memory(out[key], s.deformed_positions))
            self.assertFalse(np.shares_memory(out[key], r.nodes[7].position))
        self.assertFalse(np.shares_memory(out['from_initial_m'], out['from_reference_m']))


if __name__=='__main__':unittest.main()
