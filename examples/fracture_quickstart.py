"""Synthetic three-node axial fracture example, not material calibration."""
from fibernet import Material, StructureGraph
from fibernet.sim.fracture import FractureConfig, FractureSolver


def main():
    graph=StructureGraph(dimension=3)
    material=Material(youngs_modulus=1e9,poissons_ratio=.3)
    for x in (0,.05,.1):
        graph.add_node([x,0,0])
    for a,b in ((0,1),(1,2)):
        graph.add_edge(a,b,radius=.001,material=material)
    config=FractureConfig(young_modulus_Pa=1e9,poisson_ratio=.3,
        normal_strength_Pa=1e6,shear_strength_Pa=1e6,axis=0,
        maximum_displacement_m=.001)
    result=FractureSolver(graph,config,fixed_nodes=[0],loaded_nodes=[2],length_unit='m').run()
    print(result.termination_reason)
    print('Failed edge IDs:',[event['segment_id'] for event in result.events])
    print('Free interior-node absolute pose:',result.data['state']['positions_m']['1'])

if __name__=='__main__':
    main()
