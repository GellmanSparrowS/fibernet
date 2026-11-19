"""Animation coordinates and playback must match saved displacement rows."""
from collections import UserList, deque

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork
from fibernet.viz import visualize_deformation


def _network():
    net=FiberNetwork(dimension=3)
    net.add_fiber(Fiber.straight([0,0,0],[1,0,0],segments=1))
    net.add_fiber(Fiber.straight([2,0,0],[2,1,0],segments=2))
    return net


def _frames(net):
    n=sum(len(f.centerline) for f in net.fibers)
    a=np.zeros((n,3))
    b=a.copy();b[:2,0]=.2;b[2:,1]=np.linspace(.1,.7,n-2)
    return [a,b]


def test_distinct_fibers_use_their_own_rows_and_fixed_global_bounds():
    net=_network();frames=_frames(net)
    anim=visualize_deformation(net,frames,interval=300)
    try:
        bounds=anim._fig.axes[0].get_xlim(),anim._fig.axes[0].get_ylim()
        for step in range(2):
            anim._func(step)
            offset=0
            for fiber,line in zip(net.fibers,anim._fig.axes[0].lines):
                n=len(fiber.centerline)
                expected=fiber.centerline+frames[step][offset:offset+n];offset+=n
                np.testing.assert_array_equal(line.get_xydata(),expected[:,:2])
                assert np.all(expected[:,0]>bounds[0][0]) and np.all(expected[:,0]<bounds[0][1])
                assert np.all(expected[:,1]>bounds[1][0]) and np.all(expected[:,1]<bounds[1][1])
        assert anim.event_source.interval==300
    finally:plt.close(anim._fig)


@pytest.mark.parametrize('interval',[30,60,120,300,1500])
def test_gif_preserves_requested_slow_and_nondivisor_playback(tmp_path,interval):
    net=_network();path=tmp_path/'motion.gif'
    anim=visualize_deformation(net,_frames(net),interval=interval,save_path=path)
    try:
        with Image.open(path) as gif:
            assert gif.n_frames==2
            durations=[]
            for i in range(gif.n_frames):
                gif.seek(i);durations.append(gif.info['duration'])
            assert durations==[interval,interval]
    finally:plt.close(anim._fig)


@pytest.mark.parametrize('interval',[0,-1,np.nan,np.inf,True,'500'])
def test_bad_intervals_fail_before_creating_a_figure(interval):
    net=_network();before=plt.get_fignums()
    with pytest.raises(ValueError,match='interval'):
        visualize_deformation(net,_frames(net),interval=interval)
    assert plt.get_fignums()==before


@pytest.mark.parametrize('frames',[[],[np.zeros((2,3))],[np.full((5,3),np.nan)],
                                  [np.ma.array(np.zeros((5,3)),mask=False)]])
def test_wrong_or_missing_point_rows_reject_instead_of_repeating_prefix(frames):
    before=plt.get_fignums()
    with pytest.raises(ValueError):visualize_deformation(_network(),frames)
    assert plt.get_fignums()==before


@pytest.mark.parametrize('container', [list, tuple, deque, UserList])
def test_nested_missing_displacement_is_not_drawn_as_an_observation(container):
    net=_network();frame=_frames(net)[1]
    rows=[row.copy() for row in frame]
    rows[2]=np.ma.array(rows[2],mask=[True,False,False])
    with pytest.raises(ValueError,match='must not be masked'):
        visualize_deformation(net,[container(rows)])
