"""Canonical topology/route matrix and real STL/3MF closed-solid verification."""
import sys
import tempfile
from pathlib import Path
from dataclasses import replace
import xml.etree.ElementTree as ET
import zipfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fslab.structure import StructureFactory, all_unit_keys, CUSTOM_CELLS
from fslab.cell_rules import RULES
from fslab.manufacturing import compile_planar, compile_surface
from fslab.print_export import PrintSettings, build_solid, export_solid, export_route


def check_route(network):
    assert network.health['odd']==0 and network.health['components']==1
    assert sorted(network.route_edges)==list(range(len(network.edges)))
    assert network.route_nodes[0]==network.route_nodes[-1]
    actual=np.sort(np.column_stack([network.route_nodes[:-1],network.route_nodes[1:]]),axis=1)
    assert np.array_equal(actual,np.sort(network.edges[network.route_edges],axis=1))


def main():
    count=0
    CUSTOM_CELLS['mfg_custom_test']={'nodes':[[-.2,.5],[.5,.5],[1.2,.5],[.5,1.2],[.5,-.2]],'edges':[[0,1],[1,2],[1,3],[1,4]]}
    try:
        for unit in all_unit_keys():
            for rule in RULES:
                factory=StructureFactory(unit=unit,grid_x=2,grid_y=2,n_pts_per_side=2,expansion_rule=rule)
                original=compile_planar(factory)
                check_route(original)
                assert original.health['max_degree']<=8
                for amplitude in (.05,.6,1.):
                    changed=compile_planar(replace(factory,line_displacements=[[amplitude,-amplitude],[-amplitude,amplitude]],perturbation=.3))
                    assert np.array_equal(original.edges,changed.edges),(unit,rule)
                    assert np.array_equal(original.route_nodes,changed.route_nodes)
                    assert original.topology_id==changed.topology_id
                    count+=1
        CUSTOM_CELLS['mfg_high_degree']={'nodes':[[.5,.5]]+[[.5+.4*np.cos(t),.5+.4*np.sin(t)] for t in np.linspace(0,2*np.pi,10,endpoint=False)],'edges':[[0,k] for k in range(1,11)]}
        high=compile_planar(StructureFactory(unit='mfg_high_degree',grid_x=1,grid_y=1))
        assert high.health['max_degree']<=8
        check_route(high)
        CUSTOM_CELLS.pop('mfg_high_degree')
        square=compile_planar(StructureFactory(grid_x=2,grid_y=1,n_pts_per_side=0))
        assert len(square.edges)==14, 'shared or mirrored boundary multiplicity was lost'
        assert len(np.unique(np.sort(square.edges,axis=1),axis=0))==7
        vertices=np.array([[0,0,0],[1,0,0],[2,0,.2],[0,1,.1],[1,1,.2],[2,1,.4]])
        faces=[[0,1,4,3],[1,2,5,4]]
        for unit in all_unit_keys():
            factory=StructureFactory(unit=unit,n_pts_per_side=2)
            first=compile_surface(vertices,faces,factory)
            second=compile_surface(vertices,faces,replace(factory,line_displacements=[[.6,-.6],[.3,.5]]))
            check_route(first)
            assert np.array_equal(first.edges,second.edges)
            assert np.array_equal(first.route_nodes,second.route_nodes)
        factory=StructureFactory(unit='hexagon',grid_x=2,grid_y=2,n_pts_per_side=2,line_displacements=[[0.,.15],[0.,.15]])
        network=compile_planar(factory)
        solid=build_solid(network)
        assert np.allclose(np.ptp(solid.vertices,axis=0),[100.,100.,2.],atol=1e-4)
        assert solid.validate()>0
        curved=build_solid(compile_surface(vertices,faces,factory),PrintSettings(width=30,depth=30,curved=True,max_voxels=1000000))
        assert curved.validate()>0 and np.ptp(curved.vertices,axis=0)[2]>2
        try:
            build_solid(network,stop_cb=lambda:True)
            raise AssertionError('cancel ignored')
        except InterruptedError:
            pass
        with tempfile.TemporaryDirectory(prefix='manufacturing_test_') as folder:
            folder=Path(folder)
            export_solid(folder/'test.stl',solid)
            export_solid(folder/'test.3mf',solid)
            export_route(folder/'route.csv',network,solid.centers)
            raw=(folder/'test.stl').read_bytes()
            assert len(raw)==84+50*len(solid.faces)
            with zipfile.ZipFile(folder/'test.3mf') as archive:
                model=ET.fromstring(archive.read('3D/3dmodel.model'))
                assert model.attrib['unit']=='millimeter'
                assert len(model.findall('.//{*}triangle'))==len(solid.faces)
            assert len((folder/'route.csv').read_text().splitlines())==len(network.edges)+2
            from fslab.bambu import open_in_bambu
            fake=folder/'bambu-studio.exe'
            fake.write_bytes(b'test-only')
            from fslab import bambu
            launch, calls = bambu.subprocess.Popen, []
            try:
                bambu.subprocess.Popen = lambda *args,**kwargs: calls.append((args,kwargs))
                try:
                    open_in_bambu(folder/'test.3mf',fake)
                    raise AssertionError('3MF must not be handed to Bambu')
                except ValueError:
                    pass
                open_in_bambu(folder/'test.stl',fake)
                assert calls[0][0][0]==[str(fake),str((folder/'test.stl').resolve())]
                assert calls[0][1]['shell'] is False
                original_root = getattr(sys, '_MEIPASS', None)
                try:
                    sys._MEIPASS = str(folder)
                    open_in_bambu(folder/'test.stl',fake)
                    assert calls[-1][1]['cwd'] == str(folder)
                    assert str(folder) not in calls[-1][1]['env'].get('PATH','')
                finally:
                    if original_root is None:
                        del sys._MEIPASS
                    else:
                        sys._MEIPASS = original_root
            finally:
                bambu.subprocess.Popen = launch
        print('[manufacturing] %d planar deformation/rule cases + 12 curved topology cases PASS'%count)
        print('[manufacturing] shared twins, closed route, degree limit, exact mm envelope, STL/3MF, cancel and Bambu launch contract PASS')
    finally:
        CUSTOM_CELLS.pop('mfg_custom_test',None)
        CUSTOM_CELLS.pop('mfg_high_degree',None)


if __name__=='__main__':
    main()
