import numpy as np
import pytest
from fibernet.ml import physical_learning as learning

class NoRng:
    def integers(self, *args, **kwargs):
        raise AssertionError('RNG must not be consumed')

@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('method', ['random', 'diversity', 'test_plugin'])
def test_nonfinite_history_rejected_before_selection(bad, method, monkeypatch):
    def forbidden(*args):
        raise AssertionError('plugin must not run')
    monkeypatch.setitem(learning.ACQUISITION_PLUGINS, 'test_plugin', forbidden)
    pool=np.array([[0.,0.],[2.,0.]])
    X=np.array([[bad,0.]])
    with pytest.raises(ValueError, match='X must be finite'):
        learning.select_candidate(pool,X,np.full((1,3),np.nan),method,NoRng())
    assert np.array_equal(pool,[[0.,0.],[2.,0.]])
    assert np.isnan(X[0,0]) if np.isnan(bad) else X[0,0]==bad

def test_finite_diversity_retains_unlabelled_y():
    assert learning.select_candidate([[1.,0.],[4.,0.]],[[0.,0.]],
        np.full((1,3),np.nan),'diversity',NoRng())==1

def test_zero_history_preserves_rng_sequence():
    actual=np.random.default_rng(27);reference=np.random.default_rng(27)
    expected=int(reference.integers(3))
    assert learning.select_candidate([[1.],[2.],[3.]],np.empty((0,1)),
        np.empty((0,3)),'committee',actual)==expected
    assert actual.bit_generator.state==reference.bit_generator.state

def test_finite_history_random_preserves_rng_sequence():
    actual=np.random.default_rng(31);reference=np.random.default_rng(31)
    expected=int(reference.integers(2))
    assert learning.select_candidate([[1.],[2.]],[[0.]],
        [[np.nan,np.nan,np.nan]],'random',actual)==expected
    assert actual.bit_generator.state==reference.bit_generator.state
