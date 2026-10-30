import numpy as np
import pytest
import fibernet.ml.beam_frame_fem as module
from fibernet.ml.beam_frame_fem import BeamFrameFEM

class AssemblyReached(Exception):
    pass

def call(dim, fixed, prescribed):
    return getattr(BeamFrameFEM(), f'solve_{dim}d')(
        np.array([[0],[1]]), np.array([[0.,0.,0.],[1.,0.,0.]])[:,:dim],
        np.array([.05]), fixed_nodes=fixed, prescribed_disp=prescribed)

def guard(monkeypatch):
    def reached(*args, **kwargs):
        raise AssemblyReached('assembly sentinel; no solver execution')
    def forbidden(*args, **kwargs):
        raise AssertionError('linear solver must not run')
    monkeypatch.setattr(BeamFrameFEM,'build_stiffness_2d',reached)
    monkeypatch.setattr(BeamFrameFEM,'section_properties',reached)
    monkeypatch.setattr(module,'spsolve',forbidden)

@pytest.mark.parametrize('dim',[2,3])
@pytest.mark.parametrize('field',['fixed','prescribed'])
@pytest.mark.parametrize('bad',[.9,-.2,True,np.bool_(False),1.0,'1'])
def test_invalid_boundary_row_rejected_before_assembly(dim,field,bad,monkeypatch):
    guard(monkeypatch)
    fixed=[bad] if field=='fixed' else []
    prescribed={bad:(0.,)*dim} if field=='prescribed' else {}
    with pytest.raises(ValueError,match='invalid.*indices'):
        call(dim,fixed,prescribed)

@pytest.mark.parametrize('dim',[2,3])
@pytest.mark.parametrize('kind',['python','numpy','empty'])
def test_supported_rows_reach_original_assembly(dim,kind,monkeypatch):
    guard(monkeypatch)
    fixed,prescribed=([0],{1:(0.,)*dim}) if kind=='python' else ([np.int64(0)],{np.int32(1):(0.,)*dim})
    if kind=='empty':fixed,prescribed=[],{}
    with pytest.raises(AssemblyReached):call(dim,fixed,prescribed)
