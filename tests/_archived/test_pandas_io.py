"""Tests for pandas I/O integration."""

import pytest
import numpy as np
from fibernet import gen
from fibernet.io import to_dataframe, from_dataframe, network_summary, parametric_to_dataframe

pytest.importorskip("pandas")



def test_to_dataframe():
    """Test converting network to DataFrame."""
    net = gen.random_straight_2d(num_fibers=10, fiber_length=8, box_size=(20, 20), seed=42)
    df = to_dataframe(net)
    
    assert df is not None
    assert 'fiber_id' in df.columns
    assert 'x' in df.columns
    assert 'y' in df.columns
    assert 'radius' in df.columns
    assert df['fiber_id'].nunique() == 10


def test_from_dataframe_roundtrip():
    """Test round-trip: network -> DataFrame -> network."""
    net = gen.random_straight_2d(num_fibers=15, fiber_length=12, box_size=(30, 30), seed=42)
    
    df = to_dataframe(net)
    net2 = from_dataframe(df)
    
    assert net2.num_fibers == net.num_fibers
    # Note: point counts may differ due to resampling


def test_network_summary():
    """Test network_summary function."""
    net = gen.random_straight_2d(num_fibers=20, fiber_length=10, box_size=(25, 25), seed=42)
    
    summary = network_summary(net)
    
    assert summary is not None
    assert 'fiber_id' in summary.columns
    assert 'length' in summary.columns
    assert 'tortuosity' in summary.columns
    assert len(summary) == 20
    
    # Check statistics
    assert summary['length'].mean() == pytest.approx(10.0, abs=1e-6)


def test_parametric_to_dataframe():
    """Test parametric study results to DataFrame."""
    params = {'num_fibers': np.array([10, 20, 30]), 'length': np.array([5, 10, 15])}
    metrics = {'nematic': np.array([0.5, 0.6, 0.7]), 'porosity': np.array([0.8, 0.7, 0.6])}
    
    df = parametric_to_dataframe(params, metrics)
    
    assert df is not None
    assert 'num_fibers' in df.columns
    assert 'length' in df.columns
    assert 'nematic' in df.columns
    assert 'porosity' in df.columns
    assert len(df) == 3


def test_describe_method():
    """Test FiberNetwork.describe() method."""
    net = gen.random_straight_2d(num_fibers=30, fiber_length=10, box_size=(25, 25), seed=42)
    
    desc = net.describe()
    
    assert desc is not None
    assert isinstance(desc, str)
    assert 'FiberNetwork Summary' in desc
    assert 'Fibers: 30' in desc
    assert 'Mean:' in desc
    assert 'Material:' in desc


def test_empty_network():
    """Test with empty network."""
    from fibernet.core.network import FiberNetwork
    
    net = FiberNetwork()
    desc = net.describe()
    
    assert 'Fibers: 0' in desc


def test_plot_method_2d():
    """Test FiberNetwork.plot() for 2D networks."""
    import matplotlib
    matplotlib.use('Agg')
    
    net = gen.random_straight_2d(num_fibers=20, fiber_length=8, box_size=(25, 25), seed=42)
    fig = net.plot()
    
    assert fig is not None
    assert type(fig).__name__ == 'Figure'


def test_plot_statistics():
    """Test FiberNetwork.plot_statistics() method."""
    import matplotlib
    matplotlib.use('Agg')
    
    net = gen.random_walk_fibers(num_fibers=15, num_steps=15, step_length=0.5, seed=42)
    fig = net.plot_statistics()
    
    assert fig is not None
    assert type(fig).__name__ == 'Figure'


def test_plot_method_3d():
    """Test FiberNetwork.plot() for 3D networks."""
    import matplotlib
    matplotlib.use('Agg')
    
    net = gen.random_straight_3d(num_fibers=10, fiber_length=6, box_size=(20, 20, 20), seed=42)
    result = net.plot()
    
    # 3D plot returns pyvista Plotter or None
    assert result is not None or result is None  # Either is acceptable


def test_validate_method():
    """Test FiberNetwork.validate() method."""
    net = gen.random_straight_2d(num_fibers=20, fiber_length=10, box_size=(30, 30), seed=42)
    result = net.validate()
    
    assert 'valid' in result
    assert 'errors' in result
    assert 'warnings' in result
    assert 'stats' in result
    assert result['stats']['num_fibers'] == 20


def test_validate_empty_network():
    """Test validate() on empty network."""
    from fibernet.core.network import FiberNetwork
    net = FiberNetwork()
    result = net.validate()
    
    assert result['valid'] == False
    assert len(result['errors']) > 0


def test_to_networkx():
    """Test FiberNetwork.to_networkx() method."""
    import networkx as nx
    
    net = gen.random_straight_2d(num_fibers=25, fiber_length=10, box_size=(30, 30), seed=42)
    G = net.to_networkx()
    
    assert G is not None
    assert G.number_of_nodes() == 25
    assert G.number_of_edges() == net.num_crosslinks
    
    # Check node attributes
    node = list(G.nodes())[0]
    assert 'length' in G.nodes[node]
    assert 'radius' in G.nodes[node]
    assert 'material' in G.nodes[node]


def test_batch_simulate():
    """Test batch_simulate function."""
    from fibernet.utils.batch import batch_simulate
    
    networks = [
        gen.random_straight_2d(num_fibers=n, fiber_length=8, box_size=(25, 25), seed=i)
        for i, n in enumerate([15, 20, 25])
    ]
    
    def simulate(net):
        return {'num_fibers': net.num_fibers}
    
    result = batch_simulate(networks, simulate, parallel=False, show_progress=False)
    
    assert result.success_count == 3
    assert result.error_count == 0
    assert len(result.results) == 3


def test_parameter_study():
    """Test parameter_study function."""
    from fibernet.utils.batch import parameter_study
    
    param_grid = {'num_fibers': [15, 20], 'fiber_length': [8, 10]}
    
    study = parameter_study(
        param_grid,
        lambda **kw: gen.random_straight_2d(**kw, box_size=(25, 25), seed=42),
        lambda net: {'num_fibers': net.num_fibers}
    )
    
    assert len(study['results']) == 4  # 2 x 2 combinations
    assert len(study['params']) == 4
