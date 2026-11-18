"""Counts and ordering for the opt-in legacy sparse fiber adjacency."""

from enum import IntEnum
from unittest.mock import patch

import numpy as np
import pytest
from scipy.sparse import isspmatrix_csr
from scipy.sparse.csgraph import connected_components

from fibernet import Crosslink, Fiber, FiberNetwork
from fibernet.gen import ZigZagGenerator


def network_with_fibers(count):
    return FiberNetwork(fibers=[
        Fiber([[0., 0., 0.], [1., 0., 0.]], fiber_id=99 - i % 2)
        for i in range(count)
    ])


def link(first, second):
    return Crosslink(first, second, 0., 0., np.zeros(3))


def test_sparse_counts_every_record_and_uses_list_order():
    network = network_with_fibers(4)
    repeated = link(np.int32(0), np.uint64(1))
    network.crosslinks.extend([
        repeated, repeated, link(1, 0), link(1, 1), link(-4, np.int64(-2)),
    ])
    expected = [[0, 3, 1, 0], [3, 2, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]]
    dense = network.connectivity_matrix()
    csr = network.connectivity_matrix(sparse=np.bool_(True))
    assert isinstance(dense, np.ndarray) and isspmatrix_csr(csr)
    assert csr.dtype == dense.dtype == np.dtype(int)
    np.testing.assert_array_equal(dense, expected)
    np.testing.assert_array_equal(csr.toarray(), expected)
    np.testing.assert_array_equal(np.asarray(csr.sum(axis=1)).ravel(),
                                  network.degree_distribution())
    csr.data[:] = 0
    np.testing.assert_array_equal(network.connectivity_matrix(sparse=True).toarray(), expected)
    np.testing.assert_array_equal(network.connectivity_matrix(sparse=np.bool_(False)), expected)


@pytest.mark.parametrize('count', [0, 3])
def test_sparse_empty_and_isolated_fibers_keep_shape(count):
    csr = network_with_fibers(count).connectivity_matrix(sparse=True)
    assert isspmatrix_csr(csr)
    assert (csr.shape, csr.nnz, csr.dtype) == ((count, count), 0, np.dtype(int))


def test_sparse_uses_integer_indices_even_if_int_conversion_differs():
    class FiberRow(IntEnum):
        SECOND = 1

        def __int__(self):
            return 0

    class NumpyRow(np.int64):
        def __int__(self):
            return 0

    expected = [[0, 0, 0], [0, 0, 1], [0, 1, 0]]
    for index in (FiberRow.SECOND, NumpyRow(1)):
        network = network_with_fibers(3)
        network.crosslinks.append(link(index, 2))
        np.testing.assert_array_equal(network.connectivity_matrix(), expected)
        np.testing.assert_array_equal(network.connectivity_matrix(sparse=True).toarray(), expected)


@pytest.mark.parametrize('value', [True, np.bool_(False), 1., '1', None,
                                  np.array(1), np.array([1]), np.timedelta64(1, 'ns')])
def test_sparse_rejects_noninteger_coordinate_types(value):
    network = network_with_fibers(2)
    network.crosslinks.append(link(value, 0))
    with pytest.raises(TypeError, match='integer fiber indices'):
        network.connectivity_matrix(sparse=True)


@pytest.mark.parametrize('value', [-3, 2, np.uint64(2**64 - 1)])
def test_sparse_bounds_are_checked_before_index_dtype_conversion(value):
    network = network_with_fibers(2)
    network.crosslinks.append(link(0, value))
    with pytest.raises(IndexError, match='out of bounds'):
        network.connectivity_matrix(sparse=True)


@pytest.mark.parametrize('flag', [0, 1, None, 'True', np.array(True)])
def test_sparse_flag_is_boolean(flag):
    with pytest.raises(TypeError, match='boolean'):
        FiberNetwork().connectivity_matrix(sparse=flag)


def test_legacy_dense_advanced_indexing_is_unchanged():
    network = network_with_fibers(2)
    network.crosslinks.append(link(True, 0))
    np.testing.assert_array_equal(network.connectivity_matrix(), [[2, 2], [0, 0]])


def test_large_zigzag_sparse_connectivity_uses_source_graph_degrees():
    generator = ZigZagGenerator(n_cols=30, n_rows=30)
    graph = generator.generate()
    network = generator.to_fiber_network()
    assert (network.num_fibers, network.num_crosslinks) == (4500, 11430)
    degrees = [graph.degree[u] + graph.degree[v] - 2 for u, v in graph.edges()]

    dense_shape = (network.num_fibers, network.num_fibers)
    original_zeros = np.zeros

    def reject_dense_allocation(shape, *args, **kwargs):
        if isinstance(shape, tuple) and shape == dense_shape:
            pytest.fail('sparse connectivity attempted to allocate a dense matrix')
        return original_zeros(shape, *args, **kwargs)

    with patch('fibernet.core.network.np.zeros', side_effect=reject_dense_allocation):
        csr = network.connectivity_matrix(sparse=True)
    assert csr.shape == dense_shape
    assert csr.nnz == 2 * sum(degree * (degree - 1) // 2 for _, degree in graph.degree())
    np.testing.assert_array_equal(np.asarray(csr.sum(axis=1)).ravel(), degrees)
    assert connected_components(csr, directed=False, return_labels=False) == 1
    stored_bytes = csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes
    assert stored_bytes < (dense_shape[0] ** 2 * np.dtype(int).itemsize) // 200
