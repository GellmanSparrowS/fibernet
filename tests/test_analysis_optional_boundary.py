"""Ordinary analysis isolates OpenCV; explicit image failures stay visible."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

PRELUDE = r'''
import importlib
import importlib.abc
import inspect
import sys

attempts = []
error = {
    'absent': ModuleNotFoundError('injected absent cv2', name='cv2'),
    'broken_runtime': RuntimeError('injected broken cv2 runtime'),
    'broken_import': ImportError('injected cv2 ABI/import failure', name='cv2'),
    'broken_nested': ModuleNotFoundError('injected missing cv2 dependency', name='cv2._native'),
}[MODE]
class BlockCV2(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'cv2':
            attempts.append(fullname)
            raise error
assert not any(n.split('.')[0] == 'cv2' for n in sys.modules)
sys.meta_path.insert(0, BlockCV2())
'''


def run_python(body, mode, tmp_path):
    spec = importlib.util.find_spec('fibernet')
    assert spec is not None and spec.origin is not None
    parent = str(Path(spec.origin).resolve().parent.parent)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH'] = os.pathsep.join([parent, env.get('PYTHONPATH', '')])
    code = 'MODE = ' + repr(mode) + '\n' + PRELUDE + '\n' + textwrap.dedent(body)
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('mode', ['absent', 'broken_runtime', 'broken_import', 'broken_nested'])
def test_ordinary_topology_never_probes_cv2(mode, tmp_path):
    run_python('''
        import fibernet as fn
        import fibernet.analysis as analysis
        from fibernet.analysis.topology_metrics import analyze_topology as owner
        from fibernet.analysis import analyze_topology
        assert owner is analyze_topology is analysis.analyze_topology
        graph = fn.StructureGraph(dimension=2)
        a = graph.add_node([0., 0., 0.])
        b = graph.add_node([1., 0., 0.])
        graph.add_edge(a, b, radius=.05)
        direct = analyze_topology(graph)
        from fibernet.api import analyze
        public = analyze(graph, kind='topology')
        assert direct == public
        assert (direct.n_nodes, direct.n_physical_edges, direct.n_components) == (2, 1, 1)
        assert attempts == []
        assert 'fibernet.analysis.graph_features' not in sys.modules
        assert not any(n.split('.')[0] == 'cv2' for n in sys.modules)
        assert analysis.__all__ == [
            'GraphFeatureExtractor', 'PercolationResult', 'analyze_tensile_recruitment',
            'compute_percolation', 'SnapshotFeatureExtractor', 'ContactConfig',
            'TopologyMetricsV1', 'analyze_topology', 'GeometryPathMetricsV1',
            'analyze_geometry_paths', 'LaplacianSpectrumV1', 'analyze_spectrum']
        assert set(analysis.__all__) <= set(dir(analysis))
        assert 'GraphFeatureExtractor' not in vars(analysis)
        try:
            getattr(analysis, 'does_not_exist')
        except AttributeError:
            pass
        else:
            raise AssertionError('unknown attribute accepted')
        analysis.GraphFeatureExtractor = object()
        importlib.reload(analysis)
        assert 'GraphFeatureExtractor' not in vars(analysis)
        assert attempts == []
    ''', mode, tmp_path)


def test_absent_cv2_retains_real_export_and_existing_image_errors(tmp_path):
    run_python('''
        import fibernet as fn
        import fibernet.analysis as analysis
        assert attempts == []
        cls = analysis.GraphFeatureExtractor
        owner = importlib.import_module('fibernet.analysis.graph_features')
        from fibernet.ml import GraphFeatureExtractor as ml_export
        assert cls is owner.GraphFeatureExtractor is fn.GraphFeatureExtractor is ml_export
        assert attempts == ['cv2'] and owner.HAS_CV2 is False
        assert vars(analysis)['GraphFeatureExtractor'] is cls
        importlib.reload(analysis)
        assert 'GraphFeatureExtractor' not in vars(analysis)
        assert analysis.GraphFeatureExtractor is cls
        assert attempts == ['cv2']
        assert inspect.signature(cls) == inspect.signature(owner.GraphFeatureExtractor)
        graph = fn.StructureGraph(dimension=2)
        a = graph.add_node([0., 0., 0.]); b = graph.add_node([1., 0., 0.])
        graph.add_edge(a, b, radius=.05)
        ext = cls(canvas_size=32)
        report = ext.extract_with_status(graph)
        for name in ext.MESH_FEATURES + ext.PORE_FEATURES + ext.CONTACT_MEASUREMENTS:
            assert report['status'][name] == {'state': 'unavailable', 'reason': 'dependency_missing.cv2'}
        try:
            ext.extract(graph)
        except owner.FeatureUnavailableError as caught:
            assert 'opencv-python-headless' in str(caught)
            assert set(caught.reasons.values()) == {'dependency_missing.cv2'}
        else:
            raise AssertionError('default image availability guard bypassed')
        assert len(ext.extract(graph, allow_unavailable=True)) == 94
        assert not any(n.split('.')[0] == 'cv2' for n in sys.modules)
    ''', 'absent', tmp_path)


@pytest.mark.parametrize('mode', ['broken_runtime', 'broken_import', 'broken_nested'])
def test_explicit_image_export_preserves_broken_provider_failure(mode, tmp_path):
    run_python('''
        import fibernet.analysis as analysis
        assert attempts == []
        for _ in range(2):
            try:
                getattr(analysis, 'GraphFeatureExtractor')
            except BaseException as caught:
                assert caught is error
            else:
                raise AssertionError('broken provider mislabeled as absent')
            assert 'GraphFeatureExtractor' not in vars(analysis)
            assert 'fibernet.analysis.graph_features' not in sys.modules
        assert attempts == ['cv2', 'cv2']
        assert not any(n.split('.')[0] == 'cv2' for n in sys.modules)
    ''', mode, tmp_path)
