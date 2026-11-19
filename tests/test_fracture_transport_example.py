"""Check the fracture-driven reference resistor example against independent circuits."""
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from examples.fracture_transport import _electrical_history, run
from fibernet import Material, StructureGraph
from fibernet.sim.fracture import FractureConfig, FractureSolver


def _dense(graph, edge_ids, active, node_ids, electrodes):
    """Build the nodal balance directly and choose a reference per floating set."""
    n = len(node_ids)
    index = {nid: row for row, nid in enumerate(node_ids)}
    matrix = np.zeros((n, n))
    adjacency = [set() for _ in node_ids]
    chords = []
    for eid in edge_ids:
        edge = graph.edges[eid]
        i, j = index[edge.node_i], index[edge.node_j]
        length = math.dist(graph.nodes[edge.node_i].position, graph.nodes[edge.node_j].position)
        g = math.pi * edge.radius**2 / length if eid in active else 0.0
        chords.append((i, j, g))
        if g:
            matrix[i, i] += g
            matrix[j, j] += g
            matrix[i, j] -= g
            matrix[j, i] -= g
            adjacency[i].add(j)
            adjacency[j].add(i)
    prescribed = {index[nid]: float(level == 'high_V') for level, ids in electrodes.items() for nid in ids}
    unseen = set(range(n))
    gauges = []
    while unseen:
        source = min(unseen)
        component = {source}
        frontier = [source]
        while frontier:
            fresh = adjacency[frontier.pop()] - component
            frontier.extend(fresh)
            component.update(fresh)
        if not component.intersection(prescribed):
            prescribed[source] = 0.0
            gauges.append(node_ids[source])
        unseen -= component
    fixed = sorted(prescribed)
    free = sorted(set(range(n)) - set(fixed))
    voltage = np.zeros(n)
    voltage[fixed] = [prescribed[row] for row in fixed]
    if free:
        voltage[free] = np.linalg.solve(matrix[np.ix_(free, free)],-matrix[np.ix_(free, fixed)] @ voltage[fixed])
    current = [g*(voltage[i]-voltage[j]) for i,j,g in chords]
    injection = matrix @ voltage
    effective = sum(injection[index[nid]] for nid in electrodes['high_V'])
    return voltage, current, [g for _,_,g in chords], effective, gauges


class FractureTransportTests(unittest.TestCase):
    def test_actually_broken_parallel_paths_keep_sparse_reordered_ids(self):
        # Two different-length series paths and one permanently floating rod.
        positions = {37:(0,0,0),2:(.02,0,0),111:(.04,0,0),
                     6:(0,.01,0),52:(.04,.01,0),17:(.12,.01,0),
                     103:(0,.07,0),8:(.025,.07,0)}
        endpoints = {54:(103,8),305:(52,17),98:(6,52),11:(2,111),241:(37,2)}
        radius = .0005
        material = Material(youngs_modulus=1e9,poissons_ratio=.3)
        graph = StructureGraph.from_dict(dict(
            dimension=3,
            nodes=[dict(id=nid,position=p) for nid,p in reversed(list(positions.items()))],
            edges=[dict(id=eid,node_i=i,node_j=j,radius=radius,material=material.to_dict())
                   for eid,(i,j) in endpoints.items()],
        ))
        snapshot = graph.to_dict()
        electrodes = dict(high_V=[37,6],low_V=[111,17])
        solver = FractureSolver(graph, FractureConfig(1e9,.3,1e6,1e6,axis=0,maximum_displacement_m=160e-6),
                                fixed_nodes=electrodes['high_V'],loaded_nodes=electrodes['low_V'],
                                length_unit='m')
        fracture = solver.run()
        order = [eid for eid,_,_ in solver.geometry['original_edges']]
        report = _electrical_history(graph,fracture,mask_edge_ids=order,electrodes=electrodes)
        self.assertEqual(report['edge_ids'],[54,305,98,11,241])
        self.assertEqual(report['fracture_mask_edge_ids'],[11,54,98,241,305])
        self.assertEqual(report['node_ids'],sorted(positions))
        self.assertEqual(report['fracture_binding'],fracture.data['binding'])
        self.assertEqual(graph.to_dict(),snapshot)

        rows = report['observations']
        short,long = (math.pi*radius**2/length for length in (.04,.12))
        np.testing.assert_allclose([row['effective_conductance_S'] for row in rows],
                                  [short+long,short+long,long,long,0.],rtol=1e-14,atol=0)
        np.testing.assert_allclose([row['displacement_m'] for row in rows],
                                  np.array([0.,40.,40.,120.,120.])*1e-6,rtol=2e-11,atol=0)
        self.assertEqual(rows[2]['removed_edge_ids'],[11,241])
        self.assertEqual(rows[4]['removed_edge_ids'],[98,305])
        self.assertEqual(rows[-1]['status'],'open_circuit')
        self.assertIsNone(rows[-1]['resistance_ohm'])
        for row in rows:
            self.assertIn(8,row['gauge_node_ids'])
            self.assertEqual(row['current_A'][0],0.0)
            self.assertTrue(all(row['node_injection_A'][report['node_ids'].index(nid)]==0.0
                                for nid in row['gauge_node_ids']))
        self.assertEqual(fracture.data['state']['intact_segment_ids'],[54])
        for nid in (8,103):
            self.assertIsNone(fracture.data['state']['positions_m'][str(nid)])

    def test_public_demo_preserves_avalanche_frames_and_matches_dense_solution(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'electrical'
            report = run(out)
            def read(name):
                return json.loads((out/name).read_bytes())
            graph = StructureGraph.load_json(out/'graph.json')
            saved = read('fracture-result.json')
            self.assertEqual(report,read('electrical-history.json'))
            for filename,digest in report['files_sha256'].items():
                self.assertEqual(digest,sha256((out/filename).read_bytes()).hexdigest())
            rows = report['observations']
            self.assertEqual(len(rows),len(saved['observations']))
            edge_ids, node_ids = report['edge_ids'],report['node_ids']
            for row,mechanical in zip(rows,saved['observations']):
                active = {eid for eid,en in zip(report['fracture_mask_edge_ids'],mechanical['mask']) if en}
                self.assertEqual(row['phase'],mechanical['phase'])
                self.assertEqual(row['cascade'],mechanical['cascade'])
                self.assertEqual(row['displacement_m'],mechanical['displacement_m'])
                self.assertEqual(row['loaded_reaction_force_N'],mechanical['loaded_reaction_force_N'])
                self.assertEqual(row['strain_energy_J'],mechanical['strain_energy_J'])
                voltage,current,g,effective,gauges = _dense(graph,edge_ids,active,node_ids,report['electrodes'])
                self.assertEqual(row['gauge_node_ids'],gauges)
                np.testing.assert_allclose(row['conductance_S'],g,rtol=1e-14,atol=0)
                np.testing.assert_allclose(row['voltage_V'],voltage,rtol=1e-12,atol=1e-14)
                np.testing.assert_allclose(row['current_A'],current,rtol=1e-10,atol=1e-15)
                self.assertAlmostEqual(row['effective_conductance_S'],effective,places=15)
                self.assertAlmostEqual(row['dissipated_power_W'],effective,places=15)
            changes = [r for r in rows if r['epoch']==2 and r['removed_edge_ids']]
            self.assertEqual([len(r['removed_edge_ids']) for r in changes],[1,2,2,3,1])
            self.assertEqual(len({r['displacement_m'] for r in changes}),1)
            self.assertTrue(all(a['effective_conductance_S']>=b['effective_conductance_S']-1e-15
                                for a,b in zip(rows,rows[1:])))
            with (out/'electrical-history.csv').open(newline='',encoding='utf-8') as f:
                csv_rows = list(csv.DictReader(f))
            self.assertEqual(len(csv_rows),len(rows))
            self.assertEqual(csv_rows[-1]['status'],'open_circuit')
            self.assertEqual(csv_rows[-1]['resistance_ohm'],'')
            self.assertEqual(float(csv_rows[-1]['effective_conductance_S']),0.0)
            with self.assertRaises(FileExistsError):
                run(out)
