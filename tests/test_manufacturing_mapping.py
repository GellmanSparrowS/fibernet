"""Portable source-contract tests; all numerical and graph endpoints are doubles.

Run from the source checkout, or set FIBERNET_SOURCE_ROOT to a source overlay
and FIBERNET_REFERENCE_ROOT to its unchanged reference files. No native scientific
package is imported. These tests do not qualify physical route generation.
"""
import ast
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import make_dataclass, replace
import csv
import errno
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import MappingProxyType, ModuleType, SimpleNamespace
import unittest
from _source_check_support import run_source_check

ROOT = Path(os.environ.get('FIBERNET_SOURCE_ROOT', Path(__file__).resolve().parents[1]))
REFERENCE = Path(os.environ.get('FIBERNET_REFERENCE_ROOT', ROOT))
FIXTURES = os.environ.get('FIBERNET_TEST_TMP')


def flatten(value):
    if isinstance(value, (list, tuple)):
        return [x for row in value for x in flatten(row)]
    return [value]


class Array:
    """Tiny binary-storage double, deliberately not a numerical implementation."""
    def __init__(self, value, dtype=None, *, shape=None, readonly=False):
        if isinstance(value, Array):
            dtype = dtype or value.dtype.str
            value = value.tolist()
        self.data = deepcopy(value)
        values = flatten(value)
        kind = ('f' if any(type(v) is float for v in values) else
                'b' if values and all(type(v) is bool for v in values) else 'i')
        code = {float: '<f8', int: '<i8', bool: '|b1'}.get(dtype, dtype)
        code = code or ('|b1' if kind == 'b' else '<' + kind + '8')
        self.dtype = SimpleNamespace(str=code, kind=code[-2], itemsize=int(code[-1]))
        cast = {'f': float, 'i': int, 'u': int, 'b': bool}[self.dtype.kind]
        def convert(v):
            return [convert(x) for x in v] if isinstance(v, (list, tuple)) else cast(v)
        self.data = convert(self.data)
        def dimensions(v):
            return (len(v),) + (dimensions(v[0]) if v else ()) if isinstance(v, list) else ()
        self.shape = tuple(shape) if shape is not None else dimensions(self.data)
        self.flags = SimpleNamespace(writeable=not readonly)
    @property
    def flat(self):
        return iter(flatten(self.data))
    @property
    def T(self):
        return Array([list(row) for row in zip(*self.data)], self.dtype.str)
    def tolist(self):
        return deepcopy(self.data)
    def tobytes(self):
        fmt = {'f': 'd', 'i': 'q', 'u': 'Q', 'b': '?'}[self.dtype.kind]
        return b''.join(struct.pack('<' + fmt, value) for value in self.flat)
    def setflags(self, *, write):
        self.flags.writeable = write
    def __len__(self):
        return len(self.data)
    def __iter__(self):
        return iter(self.data)
    def __getitem__(self, key):
        return self.data[key]


def array(value, dtype=None, copy=True):
    return Array(value, dtype)


def frozen(value):
    if isinstance(value, dict):
        return MappingProxyType({key: frozen(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(frozen(item) for item in value)
    return value


class StructureGraph:
    def __init__(self):
        # Deliberately sparse, reversed node insertion; physical edges parallel.
        self.nodes = {20: SimpleNamespace(position=[2., 0., 0.], metadata={'note': ['right']}),
                      5: SimpleNamespace(position=[1., 0., 0.], metadata={})}
        self.edges = {}
        for i, (eid, a, b) in enumerate([(901, 20, 5), (11, 5, 20), (500, 20, 5), (2, 5, 20)]):
            self.edges[eid] = SimpleNamespace(node_i=a, node_j=b, radius=None if i == 1 else .25,
                material=None if i == 1 else {'name': 'stored', 'E': 7., 'extra': {'tags': ['x']}},
                metadata={'rank': i}, internal_points=[], segments=None if i == 1 else 3)
        self.metadata = {'labels': ['source'], 'length_unit': 'm'}
    def edge_array(self):
        raise AssertionError('stable endpoints must never be treated as dense rows')


# Retain the exact GraphBatch field names, while replacing the numerical endpoint.
TREE = ast.parse((REFERENCE / 'fibernet/ml/graph_data.py').read_text(encoding='utf-8'))
def dataclass_double(name):
    node = next(x for x in TREE.body if isinstance(x, ast.ClassDef) and x.name == name)
    return make_dataclass(name, [(x.target.id, object) for x in node.body
                                if isinstance(x, ast.AnnAssign)], frozen=True)
GraphBatch = dataclass_double('GraphBatch')
GraphBatchProvenance = dataclass_double('GraphBatchProvenance')
CALLS = []


def ro(value, dtype=None, shape=None):
    return Array(value, dtype, shape=shape, readonly=True)


def graph_batch(graphs, **kwargs):
    CALLS.append(deepcopy(kwargs))
    assert len(graphs) == 1 and kwargs['include_provenance'] is True
    assert kwargs['edge_features'] == ('radius',)
    if (len(kwargs['graph_ids']) != 1 or type(kwargs['graph_ids'][0]) is not str
            or not kwargs['graph_ids'][0] or type(kwargs['coordinate_frame']) is not str
            or not kwargs['coordinate_frame']):
        raise ValueError('graph/frame identity is required')
    graph = graphs[0]
    ids = sorted(graph.nodes)
    rows = {nid: row for row, nid in enumerate(ids)}
    edges = list(graph.edges.values())
    ends = [[rows[e.node_i], rows[e.node_j]] for e in edges]
    materials = [e.material for e in edges]
    internal = [p for e in edges for p in e.internal_points]
    provenance = GraphBatchProvenance('fibernet.graph_batch.provenance.v1',
        ro([[False]*6 for _ in ids], bool), ro([False]*len(ids), bool), tuple('missing' for _ in ids),
        tuple(frozen(deepcopy(m)) for m in materials), ro([m is not None for m in materials], bool),
        tuple('stored_origin_unknown' if m is not None else 'missing' for m in materials),
        ro([e.segments or 0 for e in edges]), ro([e.segments is not None for e in edges], bool),
        tuple('stored_origin_unknown' if e.segments is not None else 'missing' for e in edges))
    return GraphBatch('fibernet.graph_batch.v2', tuple(kwargs['graph_ids']), ro([3]), kwargs['length_unit'],
        kwargs['coordinate_frame'], ro([0,len(ids)]), ro([0,len(edges)]), ro(ids), ro(list(graph.edges)),
        ro([graph.nodes[nid].position for nid in ids], float), ro(list(map(list, zip(*ends)))),
        ro(internal, float, (len(internal),3)), ro([0]+[0]*len(edges)), ro([bool(e.internal_points) for e in edges], bool),
        ro([[],[]]), ro([]), ro([]), ro([]), ('radius',), ro([[e.radius or 0.] for e in edges], float),
        ro([[e.radius is not None] for e in edges], bool), (frozen(deepcopy(graph.metadata)),),
        tuple(frozen(deepcopy(graph.nodes[nid].metadata)) for nid in ids),
        tuple(frozen(deepcopy(e.metadata)) for e in edges), provenance)


class ManufacturingNetwork:
    def __init__(self, reference, positions, edges):
        self.reference, self.positions, self.edges = reference, positions, edges
        if (len(positions) != 2 or not len(edges) or len(edges) % 2
                or any(sorted(e) != [0,1] for e in edges)):
            raise ValueError('test route requires a connected all-even two-node graph')
        self.route_nodes = Array([i % 2 for i in range(len(edges)+1)])
        self.route_edges = Array(list(range(len(edges))))
        self.topology_id = hashlib.sha256(edges.tobytes()).hexdigest()
        self.added_edges = self.bridges = 0


@contextmanager
def owned_export_file(path):
    fd, name = tempfile.mkstemp(prefix='owned-', dir=Path(path).parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def export_route(path, network, centers):
    # The byte-level legacy writer contract, with no actual solver/route endpoint.
    with owned_export_file(path) as temporary:
        with temporary.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['step','incoming_edge_id','node_id','x_mm','y_mm','z_mm'])
            for step, node in enumerate(network.route_nodes):
                writer.writerow([step, -1 if step == 0 else int(network.route_edges[step-1]), int(node), *centers[node]])


def module(name, **values):
    result = ModuleType(name)
    result.__dict__.update(values)
    result.__contract_double__ = True
    return result


def symlink_or_skip(case, link, target):
    """Skip only when the host cannot create the requested test symlink."""
    try:
        link.symlink_to(target)
    except NotImplementedError as error:
        case.skipTest('symlink creation is unsupported: ' + str(error))
    except OSError as error:
        if (error.errno in {errno.EPERM, errno.EACCES, errno.ENOSYS, errno.ENOTSUP}
                or getattr(error, 'winerror', None) == 1314):
            case.skipTest('symlink creation is unavailable or not permitted: ' + str(error))
        raise


class MappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved = sys.modules.copy()
        doubles = {name: module(name, __path__=[]) for name in
                   ('fibernet','fibernet.gen','fibernet.core','fibernet.ml')}
        doubles.update({'numpy': module('numpy', ndarray=Array, array=array,
                                       asarray=lambda value, dtype=None: value if isinstance(value,Array) and dtype is None else array(value,dtype)),
            'fibernet.core.structure_graph': module('fibernet.core.structure_graph', StructureGraph=StructureGraph),
            'fibernet.ml.graph_data': module('fibernet.ml.graph_data', GraphBatch=GraphBatch,
                                           GraphBatchProvenance=GraphBatchProvenance, graph_batch=graph_batch),
            'fibernet.gen.manufacturing': module('fibernet.gen.manufacturing', ManufacturingNetwork=ManufacturingNetwork),
            'fibernet.gen.solid_export': module('fibernet.gen.solid_export', export_route=export_route,
                                              _owned_export_file=owned_export_file)})
        sys.modules.update(doubles)
        spec = importlib.util.spec_from_file_location('fibernet.gen.manufacturing_bridge', ROOT / 'fibernet/gen/manufacturing_bridge.py')
        cls.bridge = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.bridge)
        cls.solid = doubles['fibernet.gen.solid_export']
    @classmethod
    def tearDownClass(cls):
        for name in list(sys.modules):
            if name == 'numpy' or name.startswith('fibernet'):
                if name in cls.saved:
                    sys.modules[name] = cls.saved[name]
                else:
                    sys.modules.pop(name, None)
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=FIXTURES)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.csv, self.sidecar = self.root/'route.csv', self.root/'route.json'
        self.graph = StructureGraph()
        self.solid.export_route, self.solid._owned_export_file = export_route, owned_export_file
    def adapt(self, **kwargs):
        options = dict(graph_id='sample', length_unit='m', coordinate_frame='stored')
        options.update(kwargs)
        return self.bridge.manufacturing_from_graph(self.graph, **options)
    def publish(self, network, snapshot, **kwargs):
        return self.bridge.export_mapped_route(self.csv, network, snapshot, sidecar_path=self.sidecar, **kwargs)
    def absent(self):
        self.assertFalse(self.csv.exists())
        self.assertFalse(self.sidecar.exists())
    def test_sparse_multiedge_snapshot_and_mapping(self):
        n,s = self.adapt()
        self.assertEqual(s.node_ids.tolist(), [5,20])
        self.assertEqual(s.edge_ids.tolist(), [901,11,500,2])
        self.assertEqual(n.edges.tolist(), [[1,0],[0,1],[1,0],[0,1]])
        m = self.publish(n,s,source_input_sha256='a'*64)
        self.assertEqual([r['incoming_edge_id'] for r in m['route']], [None,901,11,500,2])
        self.assertEqual([r['node_id'] for r in m['route']], [5,20,5,20,5])
        self.assertEqual([r['source_edge_direction'] for r in m['route']], [None,-1,-1,-1,-1])
        self.assertEqual(m['source_input_sha256_origin'], 'caller_asserted_not_file_verified')
        self.assertEqual(m['route_csv_sha256'],hashlib.sha256(self.csv.read_bytes()).hexdigest())
        self.assertEqual(json.loads(self.sidecar.read_bytes()),m)
        self.assertEqual(m['conversion']['factor'],1000.)
        self.assertEqual(m['source_snapshot']['graph_batch']['positions'],[[1.,0.,0.],[2.,0.,0.]])
        rows = list(csv.reader(io.StringIO(self.csv.read_text(encoding='utf-8'))))
        self.assertEqual(rows[1],['0','-1','0','1000.0','0.0','0.0'])
        self.assertEqual(s.edge_features.tolist(),[[.25],[0.],[.25],[.25]])
        self.assertEqual(s.edge_feature_available.tolist(),[[True],[False],[True],[True]])
        self.assertEqual(m['source_snapshot']['graph_batch']['provenance']['edge_segments'],[3,0,3,3])
    def test_detached_ownership_and_return_copy(self):
        n,s = self.adapt()
        self.graph.nodes[5].position[0] = 99.
        self.graph.metadata['labels'].append('mutated')
        self.graph.edges[901].material['extra']['tags'].append('mutated')
        self.assertEqual(n.positions.tolist()[0][0],1.)
        m = self.publish(n,s)
        m['source_snapshot']['graph_batch']['graph_metadata'][0]['labels'].append('return mutation')
        self.assertEqual(list(s.graph_metadata[0]['labels']),['source'])
        self.assertEqual(json.loads(self.sidecar.read_bytes())['source_snapshot']['graph_batch']['graph_metadata'][0]['labels'],['source'])
        self.assertEqual(list(s.provenance.edge_material_records[0]['extra']['tags']),['x'])
        for name in self.bridge._ARRAYS:
            self.assertFalse(getattr(n,name).flags.writeable)
        self.assertIsNot(n.reference,n.positions)
        self.assertIsNot(n.positions,s.positions)
    def test_snapshot_id_swap_same_geometry(self):
        n,s = self.adapt()
        other = replace(s,node_ids=ro([100,200]))
        with self.assertRaisesRegex(ValueError,'changed'):
            self.publish(n,other)
        self.absent()
    def test_snapshot_material_swap_same_geometry(self):
        n,s = self.adapt()
        p = replace(s.provenance,edge_material_records=(frozen({'E':9}),)+s.provenance.edge_material_records[1:])
        with self.assertRaisesRegex(ValueError,'changed'):
            self.publish(n,replace(s,provenance=p))
        self.absent()
    def test_each_network_array_mutation(self):
        for name in self.bridge._ARRAYS:
            with self.subTest(name=name):
                n,s = self.adapt()
                value = getattr(n,name)
                if isinstance(value.data[0],list): value.data[0][0] += 1
                else: value.data[0] += 1
                with self.assertRaisesRegex(ValueError,'changed'):
                    self.publish(n,s)
                self.absent()
    def test_snapshot_array_mutation(self):
        n,s = self.adapt()
        s.positions.data[0][0] = 99.
        with self.assertRaisesRegex(ValueError,'changed'): self.publish(n,s)
        self.absent()
    def test_writable_replacement_refused(self):
        n,s = self.adapt()
        n.positions = array(n.positions)
        with self.assertRaisesRegex(ValueError,'read-only'): self.publish(n,s)
        self.absent()
    def test_unbound_legacy_network(self):
        n,s = self.adapt()
        del n._manufacturing_snapshot_sha256
        with self.assertRaisesRegex(ValueError,'factory-bound'): self.publish(n,s)
        self.absent()
    def test_unit_and_frame_conflicts(self):
        cases = [{'length_unit':'unknown'}, {'nested':{'units':{'length':'mm'}}},
                 {'units':'unknown'}, {'units':None}, {'radius_unit':'mm'},
                 {'coordinate_frame':'other'}]
        for metadata in cases:
            with self.subTest(metadata=metadata):
                self.graph.metadata = metadata
                with self.assertRaises(ValueError): self.adapt()
        self.absent()
    def test_node_and_edge_units_are_checked(self):
        for owner in [self.graph.nodes[20],self.graph.edges[901]]:
            owner.metadata['units']={'coordinates':'mm'}
            with self.assertRaisesRegex(ValueError,'conflicts'): self.adapt()
            owner.metadata.pop('units')
    def test_mm_and_material_units_preserved(self):
        self.graph.metadata={'length_unit':'mm','units':{'stress':'Pa'}}
        self.graph.edges[901].material['declared_unit']='Pa'
        n,s = self.adapt(length_unit='mm')
        m = self.publish(n,s)
        self.assertEqual(m['conversion']['factor'],1.)
        self.assertEqual(m['source_snapshot']['graph_batch']['provenance']['edge_material_records'][0]['declared_unit'],'Pa')
        self.assertIn('not effective solver',m['material_scope'])
    def test_invalid_declared_unit(self):
        for unit in (None,'','unknown','cm','M',True):
            with self.subTest(unit=unit), self.assertRaises(ValueError): self.adapt(length_unit=unit)
    def test_internal_points_refused(self):
        self.graph.edges[901].internal_points=[[1.5,0.,0.]]
        with self.assertRaisesRegex(ValueError,'internal-point'): self.adapt()
    def test_no_topology_repair(self):
        del self.graph.edges[2]
        with self.assertRaisesRegex(ValueError,'all-even'): self.adapt()
        self.assertEqual(list(self.graph.edges),[901,11,500])
    def test_unsupported_precision_and_radius(self):
        self.graph.nodes[5].position=[2**53+1,0,0]
        with self.assertRaisesRegex(ValueError,'binary64'): self.adapt()
        self.graph.nodes[5].position=[1.,0.,0.]
        self.graph.edges[901].radius=True
        with self.assertRaisesRegex(ValueError,'numeric storage'): self.adapt()
    def test_mixed_sequence_integer_loss_refused_before_promotion(self):
        for value in ([2**53+1,0.,0.],(2**53+1,0.,0.),[-2**53-1,0.,0.]):
            self.graph.nodes[5].position=value
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'binary64'): self.adapt()
        self.absent()
    def test_nested_sequence_integer_loss_refused_before_promotion(self):
        with self.assertRaisesRegex(ValueError,'binary64'):
            self.bridge._exact_binary64([[2**53+1,0.],[0.,0.]],'test coordinates')
    def test_exact_mixed_sequence_is_retained(self):
        self.graph.nodes[5].position=[2**53,0.,0.]
        n,s=self.adapt()
        self.assertEqual(s.positions.tolist()[0][0],2**53)
    def test_smaller_floating_storage_refused_before_widening(self):
        for dtype in ('<f2','<f4'):
            self.graph.nodes[5].position=Array([1.,0.,0.],dtype)
            with self.subTest(dtype=dtype),self.assertRaisesRegex(ValueError,'numeric storage'): self.adapt()
        self.graph.nodes[5].position=[1.,0.,0.]
        self.graph.edges[901].radius=Array(.25,'<f4')
        with self.assertRaisesRegex(ValueError,'numeric storage'): self.adapt()
    def test_nonfinite_source_refused(self):
        self.graph.nodes[5].position=[float('inf'),0.,0.]
        with self.assertRaisesRegex(ValueError,'binary64'): self.adapt()
    def test_nonfinite_conversion_refused_before_write(self):
        self.graph.nodes[20].position=[1e308,0.,0.]
        n,s = self.adapt()
        with self.assertRaisesRegex(ValueError,'nonfinite'): self.publish(n,s)
        self.absent()
    def test_collapsed_conversion_refused_before_write(self):
        a=1.025
        for _ in range(200):
            b=math.nextafter(a,math.inf)
            if a*1000. == b*1000.: break
            a=b
        else: self.fail('no adjacent binary64 conversion collision found')
        self.graph.nodes[5].position=[a,0.,0.]
        self.graph.nodes[20].position=[b,0.,0.]
        n,s=self.adapt()
        with self.assertRaisesRegex(ValueError,'collapsed'): self.publish(n,s)
        self.absent()
    def test_binary64_identity_and_underflow_guard(self):
        tiny=struct.unpack('>d',struct.pack('>Q',1))[0]
        identity=self.bridge._float64_identity
        self.assertEqual(identity(tiny),1)
        self.assertEqual(identity(-0.),identity(0.))
        with self.assertRaisesRegex(ValueError,'nonzero coordinate'):
            self.bridge._check_conversion([[tiny,0.,0.],[0.,0.,0.]],[[0.,0.,0.],[0.,0.,0.]],[[0,1]])
        self.bridge._check_conversion([[-0.,0.,0.],[1.,0.,0.]],[[0.,0.,0.],[1000.,0.,0.]],[[0,1]])
    def test_invalid_hash_before_write(self):
        n,s=self.adapt()
        for sha in ('A'*64,'x',True,123):
            with self.subTest(sha=sha),self.assertRaises(ValueError): self.publish(n,s,source_input_sha256=sha)
        self.absent()
    def test_existing_destination_refusal(self):
        n,s=self.adapt()
        for target in (self.csv,self.sidecar):
            target.write_bytes(b'keep')
            with self.assertRaises(FileExistsError): self.publish(n,s)
            self.assertEqual(target.read_bytes(),b'keep')
            other=self.sidecar if target==self.csv else self.csv
            self.assertFalse(other.exists())
            target.unlink()
    def test_alias_and_missing_parent_refusal(self):
        n,s=self.adapt()
        with self.assertRaisesRegex(ValueError,'different'):
            self.bridge.export_mapped_route(self.csv,n,s,sidecar_path=self.root/'.'/self.csv.name)
        with self.assertRaisesRegex(ValueError,'parent'):
            self.bridge.export_mapped_route(self.csv,n,s,sidecar_path=self.root/'missing'/'map.json')
        self.absent()
    def test_existing_symlink_is_not_followed(self):
        n,s=self.adapt()
        target=self.root/'original'
        target.write_bytes(b'keep')
        symlink_or_skip(self, self.sidecar, target)
        with self.assertRaises(FileExistsError): self.publish(n,s)
        self.assertEqual(target.read_bytes(),b'keep')
        self.assertFalse(self.csv.exists())
    def test_export_failure_keeps_reservations(self):
        n,s=self.adapt()
        def fail(*args): raise OSError('writer failure')
        self.solid.export_route=fail
        with self.assertRaisesRegex(OSError,'writer failure'): self.publish(n,s)
        self.assertEqual(self.csv.read_bytes(),b'')
        self.assertEqual(self.sidecar.read_bytes(),b'')
        with self.assertRaises(FileExistsError): self.publish(n,s)
    def test_sidecar_failure_keeps_csv_and_empty_completion(self):
        n,s=self.adapt()
        @contextmanager
        def fail(path):
            raise OSError('sidecar failure')
            yield path
        self.solid._owned_export_file=fail
        with self.assertRaisesRegex(OSError,'sidecar failure'): self.publish(n,s)
        self.assertTrue(self.csv.read_bytes().startswith(b'step,'))
        self.assertEqual(self.sidecar.read_bytes(),b'')
    def test_racing_second_reservation_preserves_other_file(self):
        n,s=self.adapt()
        original=self.bridge._reserve
        def race(path):
            if path == self.sidecar: path.write_bytes(b'racing writer')
            return original(path)
        self.bridge._reserve=race
        try:
            with self.assertRaises(FileExistsError): self.publish(n,s)
        finally: self.bridge._reserve=original
        self.assertEqual(self.csv.read_bytes(),b'')
        self.assertEqual(self.sidecar.read_bytes(),b'racing writer')
    def test_legacy_byte_mismatch_has_no_completion(self):
        n,s=self.adapt()
        def wrong(path,*args): path.write_bytes(b'wrong writer')
        self.solid.export_route=wrong
        with self.assertRaisesRegex(RuntimeError,'bytes differ'): self.publish(n,s)
        self.assertEqual(self.csv.read_bytes(),b'wrong writer')
        self.assertEqual(self.sidecar.read_bytes(),b'')
    def test_source_lazy_imports_and_public_surface(self):
        if not getattr(sys, '_fibernet_source_check_child', False):
            run_source_check(__file__)
            return
        tree=ast.parse((ROOT/'fibernet/gen/manufacturing_bridge.py').read_text(encoding='utf-8'))
        public=[x.name for x in tree.body if isinstance(x,ast.FunctionDef) and not x.name.startswith('_')]
        self.assertEqual(public,['manufacturing_from_graph','export_mapped_route'])
        for node in tree.body:
            if isinstance(node,ast.Import): self.assertFalse(any(x.name.startswith(('numpy','fibernet')) for x in node.names))
            if isinstance(node,ast.ImportFrom): self.assertFalse((node.module or '').startswith(('numpy','fibernet')))
        init=(ROOT/'fibernet/__init__.py').read_text(encoding='utf-8')
        root_tree=ast.parse(init)
        added=next(node for node in root_tree.body if isinstance(node,ast.ImportFrom)
                   and node.module == 'fibernet.gen.manufacturing_bridge')
        self.assertEqual([item.name for item in added.names],public)
        root_tree.body.remove(added)
        exports=next(node.value for node in root_tree.body if isinstance(node,ast.Assign)
                     and any(isinstance(target,ast.Name) and target.id=='__all__' for target in node.targets))
        self.assertTrue(all(name in [item.value for item in exports.elts] for name in public))
        exports.elts=[item for item in exports.elts if item.value not in public]
        if REFERENCE != ROOT:
            reference_tree=ast.parse((REFERENCE/'fibernet/__init__.py').read_text(encoding='utf-8'))
            self.assertEqual(ast.dump(root_tree),ast.dump(reference_tree))
        self.assertTrue(getattr(sys.modules['numpy'],'__contract_double__',False))
        self.assertFalse(any(name in sys.modules for name in ('scipy','torch','sklearn')))
    def test_forward_and_reverse_source_edge_orientation(self):
        self.graph.edges[901].node_i,self.graph.edges[901].node_j=5,20
        n,s=self.adapt()
        mapping=self.publish(n,s)
        self.assertEqual([item['source_edge_direction'] for item in mapping['route']],[None,1,-1,-1,-1])
    def test_topology_binding_change(self):
        for name,value in [('topology_id','changed'),('added_edges',1),('bridges',1)]:
            n,s=self.adapt()
            setattr(n,name,value)
            with self.assertRaisesRegex(ValueError,'changed'): self.publish(n,s)
            self.absent()
    def test_non_json_provenance_refused(self):
        self.graph.metadata['unsupported']=object()
        with self.assertRaisesRegex(ValueError,'unsupported source provenance'): self.adapt()
        self.absent()
    def test_subclass_refused(self):
        class Subgraph(StructureGraph): pass
        self.graph=Subgraph()
        with self.assertRaisesRegex(TypeError,'ordinary StructureGraph'): self.adapt()


if __name__ == '__main__':
    unittest.main()
