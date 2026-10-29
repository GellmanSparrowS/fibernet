"""Preparation adapter contract tests with stubbed native objects only.

Run directly with unittest. Checked-in retained fixtures make every check
portable and mandatory. These tests never execute FiberNet or a solver.
Real protocol parity must use an independently retained native result, rather
than deriving its expected values by making another preparation call.
"""
import asyncio
import ast
import copy
from contextlib import contextmanager
import gzip
import importlib.abc
import io
import json
import os
import runpy
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import anyio

from fibernet_mcp import preparation as prep
from fibernet_mcp.safety import ContractError, canonical, read_at, read_file, sha, strict_json
from fibernet_mcp import server

ROOT = Path(__file__).resolve().parents[1]
if bool(os.environ.get('FIBERNET_PREPARATION_TEST_BUNDLE')) != bool(os.environ.get('FIBERNET_PREPARATION_TEST_SOURCE')):
    raise RuntimeError('Supply both retained preparation fixture overrides or neither')
SAVED_BUNDLE = os.environ.get('FIBERNET_PREPARATION_TEST_BUNDLE', str(ROOT/'tests/fixtures/preparation/shear-bundle.json.gz'))
SAVED_SOURCE = os.environ.get('FIBERNET_PREPARATION_TEST_SOURCE', str(ROOT/'tests/fixtures/preparation/source.json.gz'))


def fixture_bytes(path, expected_bytes, expected_sha256):
    """Accept raw or gzip bytes, bounded before and during decompression."""
    with Path(path).open('rb') as stream:
        encoded = stream.read(expected_bytes + 1025)
    if len(encoded) > expected_bytes + 1024:
        raise RuntimeError('Retained preparation fixture encoded size exceeded')
    try:
        if encoded.startswith(b'\x1f\x8b'):
            with gzip.GzipFile(fileobj=io.BytesIO(encoded), mode='rb') as stream:
                raw = stream.read(expected_bytes + 1)
        else:
            raw = encoded
    except (OSError, EOFError) as error:
        raise RuntimeError('Invalid retained preparation fixture encoding') from error
    if len(raw) != expected_bytes or sha(raw) != expected_sha256:
        raise RuntimeError('Retained preparation fixture identity mismatch')
    return raw


SOURCE_BYTES = fixture_bytes(SAVED_SOURCE, 746048,
    '2d0550429aba412bad6d84b8522e65dc74f6b38c666c22c94927f8a2d943a014')
BUNDLE_BYTES = fixture_bytes(SAVED_BUNDLE, 1154891,
    'b5cb015bf5e765dbb481a514886a732e7ae207a39e80bc1300f0a710b6155016')
PROVENANCE = json.loads(BUNDLE_BYTES)['core_provenance']


def tiny_record():
    """The retained manual stable-ID identity fixture; no generated physics."""
    nodes = [{'id': nid, 'position': pos, 'boundary': [False] * 6,
              'metadata': {'label': str(nid)}} for nid, pos in
             [(42, [0., 0., 0.]), (105, [5., 0., 0.]), (7, [2., 0., 0.])]]
    material = {'name': 'manual-test-material', 'density': 1234.,
                'youngs_modulus': 2e6, 'poissons_ratio': .25,
                'extra': {'calibration': [1., 2.]}}
    edges = [{'id': eid, 'node_i': a, 'node_j': b, 'radius': radius,
              'segments': segments, 'internal_points': points,
              'material': copy.deepcopy(material), 'metadata': {'label': str(eid)}}
             for eid, a, b, radius, segments, points in
             [(307, 7, 42, .03, 7, None), (9, 7, 105, .02, 3, None),
              (61, 42, 7, .01, 5, [])]]
    return {'version': '3.1', 'dimension': 2, 'tolerance': 1e-8,
            'box_size': [6., 2., 0.], 'next_node_id': 1000, 'next_edge_id': 5000,
            'metadata': {'fixture': 'manual-only', 'coordinate_frame': 'lab_xyz',
                         'nested': {'values': [2, 3]}}, 'nodes': nodes, 'edges': edges}


def wire_request():
    return {'graph_id': '1' * 32, 'length_unit': 'm',
            'translations_m': [{'node_id': nid, 'value_m': value}
                               for nid, value in [(105, [1.5, 0.]), (42, [0., 0.]), (7, [.5, 0.])]],
            'reference_conductance_S': [{'edge_id': eid, 'value_S': value}
                                        for eid, value in [(61, 10.), (307, 4.), (9, 6.)]]}


def graph_info(**changes):
    return dict({'node_count': 3, 'edge_count': 3, 'length_unit': 'm',
                 'frame_id': 'lab_xyz'}, **changes)


def sorted_record(data):
    result = copy.deepcopy(data)
    result['nodes'].sort(key=lambda row: row['id'])
    result['edges'].sort(key=lambda row: row['id'])
    return result


class FakeGraph:
    """A decoder spy, not a substitute scientific graph implementation."""
    decode_change = None
    decoded = []

    def __init__(self, record):
        self.record = copy.deepcopy(record)
        self.metadata = self.record.get('metadata', {})
        self.nodes = {r['id']: types.SimpleNamespace(metadata=r.get('metadata', {}))
                      for r in self.record['nodes']}
        self.edges = {r['id']: types.SimpleNamespace(metadata=r.get('metadata', {}))
                      for r in self.record['edges']}

    @classmethod
    def from_dict(cls, data):
        graph = cls(data)
        if cls.decode_change is not None:
            cls.decode_change(graph)
        cls.decoded.append(graph)
        return graph

    def to_dict(self):
        return sorted_record(self.record)


def stub_return(data=None):
    derived = FakeGraph(tiny_record() if data is None else data)
    values = {307: 7., 9: 11., 61: 13.}  # Deliberately opaque stub outputs.
    report = {'schema': 'fibernet.deformed_transport_preparation/1',
              'scope': 'preparation_time_correspondence_only',
              'solver_executed': False, 'mechanical_result_admitted': False,
              'feedback': False, 'coordinate_frame': 'lab_xyz', 'derived_full_graph_sha256': sha(canonical(derived.to_dict())),
              'opaque_native_field': {'retain': [7, 'exactly']}}
    return derived, values, report


@contextmanager
def native_stub(side_effect=None, returned=None):
    modules = {name: types.ModuleType(name) for name in
               ('fibernet', 'fibernet.core', 'fibernet.core.structure_graph',
                'fibernet.sim', 'fibernet.sim.deformed_transport')}
    for name in ('fibernet', 'fibernet.core', 'fibernet.sim'):
        modules[name].__path__ = []
    modules['fibernet.core.structure_graph'].StructureGraph = FakeGraph
    owner = Mock(side_effect=side_effect, return_value=returned or stub_return())
    modules['fibernet.sim.deformed_transport'].prepare_deformed_transport = owner
    FakeGraph.decoded = []
    # Isolate argument forwarding/ownership. Full bundle validation is tested
    # separately against supplied, independently retained native bytes.
    with patch.dict(sys.modules, modules), patch.object(prep, 'validate_bundle',
            side_effect=lambda value, provenance, source: canonical(value)):
        yield owner


def bundle():
    return json.loads(BUNDLE_BYTES)


@contextmanager
def managed_store():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / 'source.json'
        source.write_bytes(SOURCE_BYTES)
        manifest = root / 'manifest.json'
        manifest.write_bytes(canonical({'source_tree': server.TREE, 'graphs': [{
            'path': str(source), 'sha256': sha(source.read_bytes()),
            'length_unit': 'm', 'frame_id': 'lab_xyz'}]}))
        output = root / 'output'
        output.mkdir()
        sentinel = output / 'unrelated.txt'
        sentinel.write_text('keep')
        verified = types.SimpleNamespace(provenance=lambda: copy.deepcopy(PROVENANCE))
        store = server.ManagedStore(str(manifest), str(output), verified)
        try:
            yield store, next(iter(store.graphs)), output
        finally:
            store.close()
            assert sentinel.read_text() == 'keep'
            assert sorted(p.name for p in output.iterdir()) == ['unrelated.txt']


class WireContractTests(unittest.TestCase):
    def test_O2_wire_rejection_matrix(self):
        cases = []
        def case(label, change, code='INVALID_PREPARATION_INPUT'):
            cases.append((label, change, code))
        case('duplicate node', lambda a: a['translations_m'].__setitem__(1, copy.deepcopy(a['translations_m'][0])), 'DUPLICATE_TRANSLATION_ID')
        case('duplicate edge', lambda a: a['reference_conductance_S'].__setitem__(1, copy.deepcopy(a['reference_conductance_S'][0])), 'DUPLICATE_CONDUCTANCE_ID')
        case('missing node', lambda a: a['translations_m'].pop(), 'PREPARATION_MAP_COUNT')
        case('missing edge', lambda a: a['reference_conductance_S'].pop(), 'PREPARATION_MAP_COUNT')
        for bad in (True, 42., '42', -1, 2**53):
            case('node ID ' + repr(bad), lambda a, b=bad: a['translations_m'][0].update(node_id=b))
            case('edge ID ' + repr(bad), lambda a, b=bad: a['reference_conductance_S'][0].update(edge_id=b))
        for bad in ([0.], [0., 0., 0.], [True, 0.], ['0', 0.], [float('nan'), 0.], [float('inf'), 0.]):
            case('vector ' + repr(bad), lambda a, b=bad: a['translations_m'][0].update(value_m=b))
        for bad in (True, 0., -1., '1', float('nan'), float('inf')):
            case('G ' + repr(bad), lambda a, b=bad: a['reference_conductance_S'][0].update(value_S=b))
        for bad in ('cm', 'nm', 'unspecified', '', True):
            case('unit ' + repr(bad), lambda a, b=bad: a.update(length_unit=b), 'DECLARED_LENGTH_UNIT_REQUIRED')
        case('conflicting unit', lambda a: a.update(length_unit='mm'), 'PREPARATION_UNIT_CONFLICT')
        case('extra option', lambda a: a.update(solve=True))
        case('extra row key', lambda a: a['translations_m'][0].update(rotation=0))
        for label, change, code in cases:
            with self.subTest(label=label):
                args = wire_request(); change(args)
                with self.assertRaises(prep.PreparationContractError) as error:
                    prep.preparation_request(graph_info(), args)
                self.assertEqual(error.exception.code, code)

    def test_O2_cardinality_does_not_prejudge_native_membership(self):
        args = wire_request()
        args['translations_m'][0]['node_id'] = 2**53 - 1
        args['reference_conductance_S'][0]['edge_id'] = 2**53 - 1
        actual = prep.preparation_request(graph_info(), args)
        self.assertIs(type(actual['translations_m'][0]['node_id']), int)
        self.assertEqual(actual['translations_m'], args['translations_m'])
        self.assertEqual(actual['reference_conductance_S'], args['reference_conductance_S'])

    def test_O2_count_source_request_bounds(self):
        # Fixture storage encoding cannot change or silently regenerate the
        # accepted native bytes. Raw environment overrides remain supported.
        raw = b'retained fixture'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'fixture'
            for encoded in (raw, gzip.compress(raw, mtime=0)):
                path.write_bytes(encoded)
                self.assertEqual(fixture_bytes(path,len(raw),sha(raw)),raw)
            for encoded in (raw+b'changed', gzip.compress(raw*2,mtime=0),
                            b'\x1f\x8btruncated', b'x'*(len(raw)+1025)):
                path.write_bytes(encoded)
                with self.assertRaises(RuntimeError):
                    fixture_bytes(path,len(raw),sha(raw))
        for info in (graph_info(node_count=1537), graph_info(edge_count=2049),
                     graph_info(node_count=0), graph_info(edge_count=0)):
            with self.subTest(info=info), self.assertRaises(prep.PreparationContractError) as error:
                prep.preparation_request(info, wire_request())
            self.assertEqual(error.exception.code, 'PREPARATION_GRAPH_LIMIT')
        with self.assertRaises(prep.PreparationContractError) as error:
            prep.preparation_request(graph_info(), wire_request(), 4 * 1024**2 + 1)
        self.assertEqual(error.exception.code, 'PREPARATION_SOURCE_LIMIT')
        with patch.object(prep, 'MAX_PREPARATION_REQUEST_BYTES', 1):
            with self.assertRaises(prep.PreparationContractError) as error:
                prep.preparation_request(graph_info(), wire_request())
        self.assertEqual(error.exception.code, 'PREPARATION_REQUEST_LIMIT')
        self.assertEqual(prep.NATIVE_LIMITS, {'max_nodes':1536, 'max_edges':2048,
            'max_metadata_bytes':1048576, 'max_report_bytes':2097152})

    def test_O2_no_electrical_normalization_on_preparation(self):
        for field, scalar in [('translations_m','value_m'),('reference_conductance_S','value_S')]:
            args=wire_request()
            args[field][0][scalar]=[2**53+1,0] if field=='translations_m' else 2**53+1
            before=copy.deepcopy(args)
            rpc={'jsonrpc':'2.0','id':1,'method':'tools/call','params':{
                'name':'fibernet_prepare_deformed_transport','arguments':args}}
            with self.subTest(field=field), native_stub() as owner:
                with self.assertRaises(ContractError): strict_json(canonical(rpc),electrical_rpc=True)
                with self.assertRaises(prep.PreparationContractError): prep.preparation_request(graph_info(),args)
                owner.assert_not_called()
            self.assertEqual(args,before)
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'):
            with self.assertRaises(ContractError): strict_json(raw)

    def test_O2_bundle_bounds_identity_and_nested_depth(self):
        value = bundle()
        value['source_sha256'] = 'b'*64
        self.assertEqual(prep.validate_bundle(value, PROVENANCE, 'b'*64), canonical(value))
        for field, replacement in [('source_sha256','c'*64), ('schema','bad'),
                                   ('core_provenance',{})]:
            changed = copy.deepcopy(value); changed[field] = replacement
            with self.subTest(field=field), self.assertRaises(prep.PreparationContractError):
                prep.validate_bundle(changed, PROVENANCE, 'b'*64)
        with patch.object(prep, 'MAX_PREPARATION_RESULT_BYTES', 1):
            with self.assertRaises(prep.PreparationContractError) as error:
                prep.validate_bundle(value, PROVENANCE, 'b'*64)
        self.assertEqual(error.exception.code, 'PREPARATION_RESULT_LIMIT')
        deep = 'x'
        for _ in range(18): deep = [deep]
        def rebound(changed, nested):
            changed['derived_graph']['metadata']['nested'] = nested
            report = changed['binding_report']
            report['derived_full_graph_sha256'] = sha(canonical(changed['derived_graph']))
            report['derived_graph_order_envelope_sha256'] = sha(canonical({
                'graph':changed['derived_graph'],
                'node_insertion_order':report['source_node_insertion_order'],
                'edge_insertion_order':report['source_edge_insertion_order']}))
            return changed
        changed = rebound(copy.deepcopy(value),deep)
        with self.assertRaises(prep.PreparationContractError):
            prep.validate_bundle(changed, PROVENANCE, 'b'*64)
        # A depth-16 bundle is valid on its own but cannot pass the existing
        # depth-16 worker-envelope parser after two wrapping object levels.
        nested='x'
        for _ in range(13): nested=[nested]
        boundary = rebound(copy.deepcopy(value),nested)
        prep.validate_bundle(boundary,PROVENANCE,'b'*64)
        with self.assertRaises(ContractError):
            strict_json(canonical({'core_provenance':PROVENANCE,
                'result':{'preparation':boundary}}))
        for changed in ({'preparation':value, '_core_provenance':PROVENANCE, 'private':'path'},
                        {'preparation':value, '_core_provenance':{}},
                        {'error_code':'NO_SUCH_CODE', 'error_category':'domain', '_core_provenance':PROVENANCE}):
            with self.assertRaises(prep.PreparationContractError):
                prep.validate_worker_result(changed, PROVENANCE, 'b'*64)


class DelegationTests(unittest.TestCase):
    def invoke(self, data=None, args=None):
        data = tiny_record() if data is None else data
        args = wire_request() if args is None else args
        request = prep.preparation_request(graph_info(), args)
        return prep.prepare_transport(data, request, PROVENANCE, sha(canonical(data)))

    def test_O3_O5_exactly_one_owner_call_and_detached_outputs(self):
        data = tiny_record(); args = wire_request()
        before = canonical({'data':data, 'args':args})
        returned = stub_return()
        with native_stub(returned=returned) as owner:
            result = self.invoke(data, args)
        self.assertEqual(owner.call_count, 1)
        graph = FakeGraph.decoded[0]
        self.assertIs(owner.call_args.args[0], graph)
        self.assertEqual(list(graph.nodes), [42,105,7])
        self.assertEqual(list(graph.edges), [307,9,61])
        self.assertEqual(owner.call_args.kwargs, {
            'translations_m_by_id': {105:[1.5,0.],42:[0.,0.],7:[.5,0.]},
            'reference_conductance_S': {61:10.,307:4.,9:6.}, 'length_unit':'m',
            'max_nodes':1536,'max_edges':2048,'max_metadata_bytes':1048576,'max_report_bytes':2097152})
        out = result['preparation']
        self.assertEqual([r['edge_id'] for r in out['conductance_S']], [307,9,61])
        self.assertEqual(out['binding_report'], returned[2])
        self.assertEqual(out['derived_graph'], returned[0].to_dict())
        returned[2]['opaque_native_field']['retain'].append('later mutation')
        returned[0].record['metadata']['mutated'] = True
        self.assertNotIn('later mutation', out['binding_report']['opaque_native_field']['retain'])
        self.assertNotIn('mutated', out['derived_graph']['metadata'])
        self.assertEqual(canonical({'data':data,'args':args}), before)

    def test_O5_safe_physical_integers_reach_owner_without_float_coercion(self):
        args=wire_request()
        args['translations_m'][0]['value_m']=[2**53-1,0]
        args['reference_conductance_S'][0]['value_S']=2**53-1
        with native_stub() as owner:
            self.invoke(args=args)
        translation=owner.call_args.kwargs['translations_m_by_id'][105][0]
        conductance=owner.call_args.kwargs['reference_conductance_S'][61]
        self.assertIs(type(translation),int); self.assertIs(type(conductance),int)
        self.assertEqual(translation,2**53-1); self.assertEqual(conductance,2**53-1)

    def test_O5_exception_categories_are_not_exception_text(self):
        for exception, code, category in [
            (ValueError,'PREPARATION_REJECTED','domain'),
            (TypeError,'PREPARATION_REJECTED','domain'),
            (MemoryError,'PREPARATION_RESOURCE_LIMIT','resource'),
            (RuntimeError,'NATIVE_PREPARATION_INVARIANT','internal'),
            (ImportError,'CORE_DEPENDENCY_UNAVAILABLE','internal')]:
            with self.subTest(exception=exception.__name__), native_stub(side_effect=exception('private/path and stack')) as owner:
                result = self.invoke()
            self.assertEqual(owner.call_count, 1)
            self.assertEqual(result, {'error_code':code, 'error_category':category})
            self.assertNotIn('private/path', json.dumps(result))

    def test_O6_lossless_decode_refuses_every_unrequested_change(self):
        mutations = [
            ('omitted metadata filled', lambda g:g.record['metadata'].update(default=True)),
            ('material field lost', lambda g:g.record['edges'][0]['material'].pop('extra')),
            ('node metadata lost', lambda g:g.record['nodes'][0].update(metadata={})),
            ('counter changed', lambda g:g.record.update(next_node_id=1001)),
            ('boundary altered', lambda g:g.record['nodes'][0]['boundary'].__setitem__(0,True)),
            ('node insertion reordered', lambda g:setattr(g,'nodes',dict(reversed(list(g.nodes.items()))))),
            ('edge insertion reordered', lambda g:setattr(g,'edges',dict(reversed(list(g.edges.items()))))),
        ]
        for label, mutate in mutations:
            data = tiny_record(); before = canonical(data)
            with self.subTest(label=label), patch.object(FakeGraph,'decode_change',staticmethod(mutate)), native_stub() as owner:
                result = self.invoke(data)
            self.assertEqual(result, {'error_code':'PREPARATION_SOURCE_ROUNDTRIP','error_category':'contract'})
            owner.assert_not_called()
            self.assertEqual(canonical(data), before)

    def test_O6_missing_source_fields_are_not_default_filled(self):
        cases = [('boundary', lambda d:d['nodes'][0].pop('boundary'),
                  lambda g:g.record['nodes'][0].update(boundary=[False]*6)),
                 ('counter', lambda d:d.pop('next_edge_id'),
                  lambda g:g.record.update(next_edge_id=5000)),
                 ('material', lambda d:d['edges'][0]['material'].pop('density'),
                  lambda g:g.record['edges'][0]['material'].update(density=1234.))]
        for label, omit, fill in cases:
            data=tiny_record(); omit(data); before=canonical(data)
            with self.subTest(label=label), patch.object(FakeGraph,'decode_change',staticmethod(fill)), native_stub() as owner:
                result=self.invoke(data)
            owner.assert_not_called()
            self.assertEqual(result['error_code'],'PREPARATION_SOURCE_ROUNDTRIP')
            self.assertEqual(canonical(data),before)

    def test_O6_stored_frame_is_not_injected_or_transformed(self):
        data=tiny_record(); data['metadata']['coordinate_frame']='other_frame'
        returned=stub_return(data); returned[2]['coordinate_frame']='other_frame'
        with native_stub(returned=returned) as owner:
            result=self.invoke(data)
        self.assertEqual(owner.call_count,1)
        self.assertEqual(result['error_code'],'PREPARATION_FRAME_CONFLICT')
        data=tiny_record(); data['metadata'].pop('coordinate_frame')
        returned=stub_return(data); returned[2]['coordinate_frame']=None
        with native_stub(returned=returned) as owner:
            result=self.invoke(data)
        self.assertEqual(owner.call_count,1)
        self.assertNotIn('coordinate_frame',result['preparation']['derived_graph']['metadata'])
        self.assertIsNone(result['preparation']['binding_report']['coordinate_frame'])

    def test_O9_source_has_no_scientific_implementation_or_new_provider_import(self):
        tree = ast.parse(Path(prep.__file__).read_text())
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module: imports.add(node.module)
        self.assertEqual(imports - {'safety','re'}, {'fibernet.core.structure_graph', 'fibernet.sim.deformed_transport'})
        owner_calls = [node for node in ast.walk(tree) if isinstance(node,ast.Call)
                       and isinstance(node.func,ast.Name) and node.func.id=='prepare_deformed_transport']
        self.assertEqual(len(owner_calls),1)
        for forbidden in ('numpy', 'scipy', 'torch', 'taichi', 'lammps'):
            self.assertFalse(any(name==forbidden or name.startswith(forbidden+'.') for name in imports))


class StoreAndCancellationTests(unittest.TestCase):
    def test_O7_failed_launch_cleans_owned_request(self):
        with managed_store() as (store,gid,output):
            sup = server.Supervisor(store)
            temporary=str(output/'operator-temporary-root')
            for operation in ('prepare_deformed_transport','linear_transport'):
                with self.subTest(operation=operation), patch.dict(os.environ,{'TMPDIR':temporary}):
                    with patch.object(server.subprocess,'Popen',side_effect=OSError('injected')) as launch:
                        with self.assertRaises(OSError):
                            anyio.run(sup.run,operation,gid,
                                      prep.preparation_request(graph_info(),wire_request()))
                    environment=launch.call_args.kwargs['env']
                    if operation=='prepare_deformed_transport':
                        self.assertEqual(environment.get('TMPDIR'),temporary)
                    else:
                        self.assertNotIn('TMPDIR',environment)
            self.assertIsNone(sup.active)
            self.assertFalse(list(Path(store.path).glob('request-*')))
            self.assertFalse(list(Path(store.path).glob('worker-*')))
            self.assertEqual(store.artifacts,{})

    def test_O7_source_hash_symlink_and_worker_provenance(self):
        with managed_store() as (store,gid,output):
            source = Path(store.path)/gid
            raw = source.read_bytes()
            source.write_bytes(raw+b' ')
            with self.assertRaises(ContractError): store.source(gid)
            source.write_bytes(raw)
            link = output/'symlink.json'; link.symlink_to(source)
            try:
                with self.assertRaises(OSError): read_file(str(link))
            finally: link.unlink()
            for payload in ({'core_provenance':{},'result':{}},
                            {'core_provenance':PROVENANCE,'result':{},'extra':True},
                            {'core_provenance':PROVENANCE,'result':{'_core_provenance':{}}}):
                with self.assertRaises(ContractError): server._verified_worker_result(payload,PROVENANCE)
            self.assertFalse(store.artifacts)

    def test_O7_worker_request_and_source_integrity_before_owner(self):
        # Execute only worker.main with OS-limit installation stubbed and native
        # preparation replaced. This launches no subprocess and imports no core.
        with patch('resource.setrlimit'):
            worker = runpy.run_module('fibernet_mcp.worker', run_name='preparation_worker_contract_test')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/'source.json'; request=root/'request.json'
            source.write_bytes(canonical(tiny_record()))
            request.write_bytes(canonical(prep.preparation_request(graph_info(),wire_request())))
            source_digest=sha(source.read_bytes()); request_digest=sha(request.read_bytes())
            verified=types.SimpleNamespace(provenance=lambda:PROVENANCE)
            worker['_verify_core_result']=lambda:verified
            # main retains its module globals, which are the returned mapping.
            main=worker['main']; main.__globals__['_verify_core_result']=lambda:verified
            for mode in ('source_hash','request_hash','request_symlink'):
                with self.subTest(mode=mode):
                    output=root/'result.json'
                    source_hash='0'*64 if mode=='source_hash' else source_digest
                    request_hash='0'*64 if mode=='request_hash' else request_digest
                    supplied=request
                    if mode=='request_symlink':
                        supplied=root/'request-link.json'; supplied.symlink_to(request)
                    argv=['worker','prepare_deformed_transport',str(source),str(output),source_hash,str(supplied),request_hash]
                    with patch.object(sys,'argv',argv), patch.object(prep,'prepare_transport') as owner:
                        with self.assertRaises((ValueError,OSError)): main()
                    owner.assert_not_called(); self.assertFalse(output.exists())
                    if supplied!=request: supplied.unlink()

    def test_O7_malformed_worker_output_cleans_temporaries(self):
        class Completed:
            pid=123456789
            returncode=0
            def poll(self): return self.returncode
            def wait(self): return self.returncode
        with managed_store() as (store,gid,output):
            sup=server.Supervisor(store)
            source_before=(Path(store.path)/gid).read_bytes()
            def launch(command,**kwargs):
                Path(command[command.index('prepare_deformed_transport')+2]).write_bytes(canonical({'core_provenance':{},'result':{}}))
                return Completed()
            with patch.object(server.subprocess,'Popen',side_effect=launch):
                result=anyio.run(sup.run,'prepare_deformed_transport',gid,
                    prep.preparation_request(graph_info(),wire_request()))
            self.assertEqual(result['error']['code'],'WORKER_PROVENANCE_MISMATCH')
            self.assertEqual((Path(store.path)/gid).read_bytes(),source_before)
            self.assertTrue(sup.audit[0]['reaped'])
            self.assertIsNone(sup.active); self.assertFalse(store.artifacts)
            self.assertFalse(list(Path(store.path).glob('worker-*')))
            self.assertFalse(list(Path(store.path).glob('request-*')))

    def test_O7_collisions_preserve_preexisting_files(self):
        for collision in ('request','worker'):
            with self.subTest(collision=collision), managed_store() as (store,gid,output):
                name=collision+'-'+'c'*32
                path=Path(store.path)/name; path.write_bytes(b'preexisting sentinel')
                sup=server.Supervisor(store)
                ids=[types.SimpleNamespace(hex='c'*32),types.SimpleNamespace(hex='c'*32)]
                try:
                    with patch.object(server.uuid,'uuid4',side_effect=ids), patch.object(server.subprocess,'Popen') as launch:
                        with self.assertRaises((ContractError,OSError)):
                            anyio.run(sup.run,'prepare_deformed_transport',gid,
                                prep.preparation_request(graph_info(),wire_request()))
                        launch.assert_not_called()
                    self.assertEqual(path.read_bytes(),b'preexisting sentinel')
                    self.assertFalse(store.artifacts); self.assertIsNone(sup.active)
                finally: path.unlink()

    def test_O7_partial_write_failure_and_budgets(self):
        with managed_store() as (store,gid,output):
            value = bundle(); value['source_sha256'] = store.graphs[gid]['source_sha256']
            before = set(os.listdir(store.path)); total = store.total
            staged = store.stage_artifact(gid,value,PROVENANCE,kind='deformed_transport_preparation')
            self.assertEqual(set(os.listdir(store.path)),before)
            self.assertFalse(store.artifacts)
            with patch('fibernet_mcp.safety.os.fsync',side_effect=OSError('write failure')):
                with self.assertRaises(OSError): store.commit_artifact(staged)
            self.assertEqual(set(os.listdir(store.path)),before)
            self.assertEqual(store.total,total)
            self.assertFalse(store.artifacts)
            with patch.object(server,'MAX_SESSION',total):
                with self.assertRaises(ContractError):
                    staged = store.stage_artifact(gid,value,PROVENANCE,kind='deformed_transport_preparation')
                    store.commit_artifact(staged)
            store.artifacts.update({str(i):{'kind':'different'} for i in range(64)})
            try:
                with self.assertRaises(ContractError): store.stage_artifact(gid,value,PROVENANCE,kind='deformed_transport_preparation')
            finally: store.artifacts.clear()
            self.assertEqual(set(os.listdir(store.path)),before)

    def test_O8_bundle_atomic_visibility_dedup_and_bytes(self):
        with managed_store() as (store,gid,output):
            value = bundle(); value['source_sha256'] = store.graphs[gid]['source_sha256']
            staged = store.stage_artifact(gid,value,PROVENANCE,kind='deformed_transport_preparation')
            self.assertFalse(store.artifacts)
            descriptor = store.commit_artifact(staged)
            self.assertEqual(len(store.artifacts),1)
            raw = read_at(store.fd,descriptor['artifact_id'])
            self.assertEqual(raw,canonical(value))
            self.assertEqual(sha(raw),descriptor['sha256'])
            value['binding_report']['caller_mutation'] = 'changed caller'
            self.assertEqual(read_at(store.fd,descriptor['artifact_id']),raw)
            duplicate = store.stage_artifact(gid,strict_json(raw),PROVENANCE,kind='deformed_transport_preparation')
            self.assertEqual(store.commit_artifact(duplicate),descriptor)
            self.assertEqual(len(store.artifacts),1)
            path = Path(store.path)/descriptor['artifact_id']
            path.write_bytes(raw+b' ')
            with self.assertRaises(ContractError):
                store.stage_artifact(gid,strict_json(raw),PROVENANCE,kind='deformed_transport_preparation')
            path.write_bytes(raw)

    def test_O8_reply_failure_and_pending_cancel_publish_nothing(self):
        async def exercise(store,gid):
            value = bundle(); value['source_sha256'] = store.graphs[gid]['source_sha256']
            with self.assertRaises(ValueError):
                await server.publish_preparation(store,gid,value,PROVENANCE,
                    lambda data: (_ for _ in ()).throw(ValueError('reply schema')))
            self.assertFalse(store.artifacts)
            # Model a cancellation delivered while cleanup has a shield active.
            with anyio.CancelScope() as scope:
                with anyio.CancelScope(shield=True):
                    scope.cancel()
                    await anyio.sleep(0)
                await server.publish_preparation(store,gid,value,PROVENANCE,lambda data:None)
                self.fail('pending cancellation must win before commit')
            self.assertTrue(scope.cancelled_caught)
            self.assertFalse(store.artifacts)
            response = await server.publish_preparation(store,gid,value,PROVENANCE,lambda data:None)
            self.assertEqual(response['status'],'prepared')
            self.assertEqual(len(store.artifacts),1)
            total = store.total
            with self.assertRaises(ValueError):
                await server.publish_preparation(store,gid,value,PROVENANCE,
                    lambda data: (_ for _ in ()).throw(ValueError('dedup reply schema')))
            self.assertEqual(store.total,total)
            self.assertEqual(len(store.artifacts),1)
            with anyio.CancelScope() as scope:
                scope.cancel()
                await anyio.sleep(0)
            self.assertEqual(store.info(response['artifact']['artifact_id']),response['artifact'])
        with managed_store() as (store,gid,output):
            anyio.run(exercise,store,gid)


if __name__ == '__main__':
    unittest.main(verbosity=2)
