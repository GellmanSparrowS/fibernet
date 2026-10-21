"""Tiny exact-value checks of the current face-index guard, with no provider.

The current source guard is extracted so large logical vertex counts can test
INT32_MAX independently without allocating a multi-billion-row vertex array.
Only the current repaired guard executes; this is not a native-geometry test.
"""
import ast
from pathlib import Path
import sys

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / 'fibernet' / 'gen' / 'tube_solid.py'
DTYPES = ['i1', 'u1', '<i2', '>i2', '<u2', '>u2',
          '<i4', '>i4', '<u4', '>u4', '<i8', '>i8', '<u8', '>u8']
ERROR = 'mesh face index is outside the vertex table or int32 range'


@pytest.fixture(scope='module')
def range_guard():
    tree = ast.parse(SOURCE.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'tube_mesh')
    guards = [node for node in function.body
              if isinstance(node, ast.If) and ERROR in ast.unparse(node)]
    assert len(guards) == 1
    assert ast.unparse(guards[0].test) == 'indices.size'
    return compile(ast.Module(body=guards, type_ignores=[]), str(SOURCE), 'exec')


def accepted(range_guard, indices, vertex_count):
    try:
        exec(range_guard, {'np': np, 'indices': indices, 'properties': range(vertex_count)})
    except ValueError as error:
        assert str(error) == ERROR
        return False
    return True


@pytest.mark.parametrize('dtype', DTYPES)
def test_range_guard_matches_exact_integer_values(range_guard, dtype, request):
    comparisons = 0
    info = np.iinfo(dtype)
    values = sorted({int(value) for value in [
        info.min, info.max, -1, 0, 1, 3, 4, 2**15, 2**16-1,
        2**31-1, 2**31, 2**32, 2**63-1, 2**64-1,
    ] if info.min <= value <= info.max})
    vertex_counts = sorted({min(value, sys.maxsize) for value in
                            [0, 1, 4, 2**16, 2**31, 2**31+1, 2**33, sys.maxsize]})
    for value in values:
        storage = np.zeros((4, 6), dtype=dtype)
        storage[::2, ::2] = [[0, 1, value], [value, 1, 0]]
        positive_stride = storage[::2, ::2]
        layouts = [positive_stride, positive_stride[::-1, ::-1],
                   np.asfortranarray(positive_stride)]
        for indices in layouts:
            indices.setflags(write=False)
            before = indices.tobytes()
            exact = [int(item) for row in indices.tolist() for item in row]
            for vertex_count in vertex_counts:
                expected = all(0 <= item < vertex_count and item <= 2**31-1
                               for item in exact)
                assert accepted(range_guard, indices, vertex_count) == expected
                assert indices.tobytes() == before
                comparisons += 1
    for vertex_count in [0, 4, min(2**31+1, sys.maxsize)]:
        assert accepted(range_guard, np.empty((0, 3), dtype=dtype), vertex_count)
        comparisons += 1
    request.node.user_properties.append(("exact_value_comparisons", comparisons))
