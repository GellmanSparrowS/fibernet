"""Report wrong target/feature axes before backend construction or refit."""
import numpy as np
import pytest
from fibernet.ml import PhysicalRegressor


def _elastic_bar_data():
    # Closed-form linear bars: force, strain energy and end translation.
    length=np.linspace(.01,.02,12)
    radius=np.linspace(5e-5,9e-5,12)
    strain=np.tile([.005,.01],6)
    area=np.pi*radius**2
    x=np.column_stack([length,area,strain])
    y=np.column_stack([1e6*area*strain,.5*1e6*area*length*strain**2,length*strain])
    return x,y


@pytest.mark.parametrize('layout',['two_targets','one_target','transposed'])
def test_target_shape_names_expected_and_received_axes(layout):
    x,y=_elastic_bar_data()
    bad={'two_targets':y[:,:2],'one_target':y[:,0],'transposed':y.T}[layout]
    model=PhysicalRegressor(key='ridge')
    with pytest.raises(ValueError) as exc:
        model.train(x,bad)
    assert str(exc.value)==f'training targets must have shape (12, 3); got {bad.shape}'
    assert not hasattr(model,'estimator')


@pytest.mark.parametrize('shape',[(36,),(1,12,3)])
def test_feature_shape_reports_received_axes(shape):
    x,y=_elastic_bar_data();x=x.reshape(shape)
    model=PhysicalRegressor(key='ridge')
    with pytest.raises(ValueError) as exc:
        model.train(x,y)
    assert str(exc.value)==f'training features must have shape (N, D); got {shape}'
    assert not hasattr(model,'estimator')


def test_wrong_target_count_preserves_existing_numpy_fit():
    x,y=_elastic_bar_data()
    model=PhysicalRegressor(key='mlp',seed=23)
    model.train(x,y,epochs=3,validation_indices=[10,11])
    estimator=model.estimator;before=model.predict(x).copy()
    with pytest.raises(ValueError,match=r'targets must have shape \(12, 3\); got \(12, 2\)'):
        model.train(x,y[:,:2])
    assert model.estimator is estimator
    np.testing.assert_array_equal(model.predict(x),before)
