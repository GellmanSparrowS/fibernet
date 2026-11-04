"""Tiny text/admission checks; no solver execution or deck acceptance claim."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from fibernet.core.fiber import Fiber, CrossSection
from fibernet.core.material import Material
from fibernet.core.network import FiberNetwork, Crosslink
from fibernet.io import fea_export as fea
from fibernet import io


# Independently authored spec-reader; intentionally not product parser code.
from pathlib import Path
from math import isfinite


def read_msh22_lines(path):
    data = Path(path).read_bytes()
    if len(data) > 1024 * 1024:
        raise ValueError('review fixture exceeds one MiB')
    lines = iter(line.strip() for line in data.decode('ascii').splitlines() if line.strip())
    def take():
        try:
            return next(lines)
        except StopIteration as exc:
            raise ValueError('truncated mesh') from exc
    def expect(value):
        if take() != value:
            raise ValueError('expected ' + value)
    def count():
        value = int(take())
        if not 0 <= value <= 10000:
            raise ValueError('count outside reviewer fixture bound')
        return value
    expect('$MeshFormat'); expect('2.2 0 8'); expect('$EndMeshFormat')
    expect('$Nodes'); nodes = {}
    for _ in range(count()):
        row = take().split()
        if len(row) != 4:
            raise ValueError('node record requires ID and three coordinates')
        tag = int(row[0]); xyz = tuple(map(float, row[1:]))
        if tag <= 0 or tag in nodes or not all(map(isfinite, xyz)):
            raise ValueError('invalid node ID or coordinate')
        nodes[tag] = xyz
    expect('$EndNodes'); expect('$Elements'); elements = {}
    for _ in range(count()):
        row = list(map(int, take().split()))
        if len(row) != 7 or row[1:4] != [1, 2, 0] or row[4] <= 0:
            raise ValueError('expected two-tag, two-node line with positive entity')
        tag, _, _, physical, entity, first, last = row
        if tag <= 0 or tag in elements or first == last or first not in nodes or last not in nodes:
            raise ValueError('invalid element ID or connectivity')
        elements[tag] = {'physical': physical, 'entity': entity, 'nodes': (first, last)}
    expect('$EndElements')
    if next(lines, None) is not None:
        raise ValueError('unexpected trailing section in reviewer subset')
    return {'nodes': nodes, 'elements': elements}


def network(points=None, *, radius=.125, material='default'):
    if points is None:
        points = [[0., 0., 0.], [.5, 0., 0.], [1., 0., 0.]]
    mat = Material() if material == 'default' else material
    return FiberNetwork(fibers=[Fiber(np.array(points, dtype=float), radius=radius, material=mat)])


class FEAExportContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def rejected_without_write(self, fn, net, exc=ValueError, **kwargs):
        for exists in (False, True):
            path = self.root / ('existing' if exists else 'absent')
            if exists:
                path.write_bytes(b'KEEP EXISTING BYTES\x00')
            elif path.exists():
                path.unlink()
            with self.assertRaises(exc):
                fn(net, path, **kwargs)
            if exists:
                self.assertEqual(path.read_bytes(), b'KEEP EXISTING BYTES\x00')
            else:
                self.assertFalse(path.exists())

    def test_public_aliases(self):
        self.assertIs(io.export_to_abaqus, fea.export_to_abaqus)
        self.assertIs(io.export_to_ansys, fea.export_to_ansys)
        self.assertIs(io.export_to_gmsh_fea, fea.export_to_gmsh)
        self.assertIs(io.export_to_lammps_fea, fea.export_to_lammps)

    def test_curved_backtracking_and_degenerate(self):
        examples = [ [[0,0,0],[.5,1,0],[1,0,0]],
                     [[0,0,0],[.8,0,0],[.2,0,0],[1,0,0]],
                     [[0,0,0],[1,0,0],[0,0,0]],
                     [[0,0,0],[2,0,0],[1,0,0]] ]
        for fn in (fea.export_to_abaqus, fea.export_to_ansys, fea.export_to_gmsh):
            for points in examples:
                with self.subTest(exporter=fn.__name__, points=points):
                    self.rejected_without_write(fn, network(points))

    def test_crosslinks(self):
        net = network(); net.fibers.append(Fiber([[.5,-1,0],[.5,1,0]], radius=.125))
        net.crosslinks.append(Crosslink(0, 1, .5, .5, np.array([.5,0,0])))
        for fn in (fea.export_to_abaqus, fea.export_to_ansys, fea.export_to_gmsh):
            self.rejected_without_write(fn, net)

    def test_nonfinite_and_overflow_geometry(self):
        for middle in ([np.nan,0,0], [np.inf,0,0]):
            for fn in (fea.export_to_abaqus, fea.export_to_ansys, fea.export_to_gmsh):
                self.rejected_without_write(fn, network([[0,0,0],middle,[1,0,0]]))
        self.rejected_without_write(fea.export_to_abaqus, network([[-1e308,0,0],[1e308,0,0]]))

    def test_large_offset_does_not_admit_curve(self):
        self.rejected_without_write(fea.export_to_abaqus,
            network([[1e15,0,0],[1e15+.5,.5,0],[1e15+1,0,0]]))

    def test_invalid_segments(self):
        for value in (0, -1, 1.5, True, np.bool_(True)):
            for fn in (fea.export_to_abaqus, fea.export_to_ansys, fea.export_to_gmsh):
                self.rejected_without_write(fn, network(), segments_per_fiber=value)

    def test_supported_straight_sampled_text_and_separate_endpoints(self):
        net = network(); net.fibers.append(Fiber([[1,0,0],[2,0,0]], radius=.125))
        a=self.root/'a'; b=self.root/'b'; g=self.root/'g'
        fea.export_to_abaqus(net,a,segments_per_fiber=2)
        fea.export_to_ansys(net,b,segments_per_fiber=np.int64(2))
        fea.export_to_gmsh(net,g,segments_per_fiber=2)
        text=a.read_text()
        self.assertIn('3, 1, 0, 0', text)
        self.assertIn('4, 1, 0, 0', text)
        self.assertIn('1, 1, 2\n2, 2, 3\n3, 4, 5\n4, 5, 6', text)
        self.assertIn('SECDATA, 0.125', b.read_text())
        self.assertIn('$Nodes', g.read_text())

    def test_ansys_radius_independent_of_missing_material(self):
        path=self.root/'a'; fea.export_to_ansys(network(material=None),path)
        self.assertIn('SECDATA, 0.125',path.read_text())
        self.assertIn('MP, EX, 1, 1000000000',path.read_text())
        self.assertNotIn('SECDATA, 0.5',path.read_text())

    def test_beam_heterogeneity_and_geometry_only_gmsh(self):
        for change in ('radius', 'youngs_modulus', 'poissons_ratio', 'density'):
            net=network(); second=network().fibers[0]; net.fibers.append(second)
            if change == 'radius': second.radius=.25
            else: setattr(second.material,change,getattr(second.material,change)*1.1)
            for fn in (fea.export_to_abaqus,fea.export_to_ansys):
                self.rejected_without_write(fn, net)
            fea.export_to_gmsh(net,self.root/'g')

    def test_unused_material_names_do_not_reject(self):
        net=network(); second=network().fibers[0];second.material.name='other';net.fibers.append(second)
        fea.export_to_abaqus(net,self.root/'a')
        fea.export_to_ansys(net,self.root/'b')

    def test_invalid_beam_properties(self):
        for attr,value in [('radius',0),('radius',np.nan),('youngs_modulus',-1),('poissons_ratio',.5),('density',np.inf)]:
            net=network()
            setattr(net.fibers[0] if attr=='radius' else net.fibers[0].material,attr,value)
            for fn in (fea.export_to_abaqus,fea.export_to_ansys):
                self.rejected_without_write(fn,net)
        net=network();net.fibers[0].cross_section=CrossSection.RECTANGULAR
        self.rejected_without_write(fea.export_to_abaqus,net)
        net=network();net.fibers[0].material.stiffness_tensor=np.eye(6)
        self.rejected_without_write(fea.export_to_ansys,net)

    def test_unsupported_controls(self):
        for kind in ('B32','B33'):
            self.rejected_without_write(fea.export_to_abaqus,network(),beam_type=kind)
        self.rejected_without_write(fea.export_to_ansys,network(),element_type='BEAM189')
        for size in (2.,np.nan,True):
            self.rejected_without_write(fea.export_to_abaqus,network(),element_size=size)

    def test_lammps_always_fails_before_open(self):
        for net in (network(), FiberNetwork()):
            self.rejected_without_write(fea.export_to_lammps,net,NotImplementedError)

    def test_wrong_representation(self):
        for fn in (fea.export_to_abaqus,fea.export_to_ansys,fea.export_to_gmsh):
            self.rejected_without_write(fn,object(),TypeError)

    def test_empty_network_rejected_before_open(self):
        for fn in (fea.export_to_abaqus,fea.export_to_ansys,fea.export_to_gmsh):
            self.rejected_without_write(fn,FiberNetwork())

    def test_periodic_network_rejected_before_open(self):
        net=network();net.periodic=True
        for fn in (fea.export_to_abaqus,fea.export_to_ansys,fea.export_to_gmsh):
            self.rejected_without_write(fn,net)

    def test_abaqus_microgeometry_and_property_roundtrip(self):
        mat=Material(youngs_modulus=123456789.12345679, poissons_ratio=.23456789123456789,
                     density=987.6543219876543)
        points=np.array([[1.234567891234567e-9,2.345678912345678e-9,0.],
                         [1.234567891234567e-8,2.345678912345678e-9,0.]])
        net=network(points,radius=1.234567891234567e-10,material=mat)
        path=self.root/'micro.inp';fea.export_to_abaqus(net,path,segments_per_fiber=1)
        text=path.read_text()
        rows=text.split('*NODE\n',1)[1].split('*ELEMENT',1)[0].strip().splitlines()
        actual=np.array([[float(x) for x in row.split(',')[1:]] for row in rows])
        np.testing.assert_array_equal(actual,points)
        radius=float(text.split('*BEAM SECTION',1)[1].splitlines()[1])
        self.assertEqual(radius,net.fibers[0].radius)
        elastic=text.split('*ELASTIC\n',1)[1].splitlines()[0]
        self.assertEqual(tuple(map(float,elastic.split(','))),(mat.youngs_modulus,mat.poissons_ratio))
        self.assertEqual(float(text.split('*DENSITY\n',1)[1].splitlines()[0]),mat.density)

    def test_ansys_microgeometry_and_property_roundtrip(self):
        mat=Material(youngs_modulus=123456789.12345679,poissons_ratio=.23456789123456789,
                     density=987.6543219876543)
        points=np.array([[1.234567891234567e-9,0.,0.],[1.234567891234567e-8,0.,0.]])
        net=network(points,radius=1.234567891234567e-10,material=mat)
        path=self.root/'micro.dat';fea.export_to_ansys(net,path,segments_per_fiber=1)
        lines=path.read_text().splitlines()
        actual=np.array([[float(x) for x in line.split(',')[2:]] for line in lines if line.startswith('N, ')])
        np.testing.assert_array_equal(actual,points)
        for prefix,value in [('SECDATA, ',net.fibers[0].radius),('MP, EX, 1, ',mat.youngs_modulus),
                             ('MP, PRXY, 1, ',mat.poissons_ratio),('MP, DENS, 1, ',mat.density)]:
            self.assertEqual(float(next(line[len(prefix):] for line in lines if line.startswith(prefix))),value)

    def test_gmsh_microgeometry_roundtrip_text(self):
        points=np.array([[1.234567891234567e-9,0.,0.],[1.234567891234567e-8,0.,0.]])
        path=self.root/'micro.msh';fea.export_to_gmsh(network(points),path,segments_per_fiber=1)
        # MSH2.2 node payload; mesh grammar is checked independently as well.
        rows=path.read_text().split('$Nodes\n',1)[1].split('$EndNodes',1)[0].strip().splitlines()[1:]
        actual=np.array([[float(x) for x in row.split()[1:]] for row in rows])
        np.testing.assert_array_equal(actual,points)

    def test_abaqus_fixed_orientation_preflight(self):
        for points in ([[0,0,0],[0,0,1]], [[0,0,1],[0,0,0]],
                       [[0,0,0],[1e-16,0,1]]):
            self.rejected_without_write(fea.export_to_abaqus,network(points))
        fea.export_to_abaqus(network([[0,0,0],[1,0,1]]),self.root/'diagonal.inp')
        # The restriction belongs only to the fixed Abaqus section orientation.
        fea.export_to_ansys(network([[0,0,0],[0,0,1]]),self.root/'vertical.dat')
        fea.export_to_gmsh(network([[0,0,0],[0,0,1]]),self.root/'vertical.msh')

    def test_gmsh22_independent_reader_geometry_and_connectivity(self):
        net=network([[0,0,0],[1.234567891234567e-9,0,0]],radius=1e-10)
        net.fibers.append(Fiber([[1.234567891234567e-9,0,0],[2.345678912345678e-9,0,0]],radius=2e-10))
        path=self.root/'independent.msh';fea.export_to_gmsh(net,path,segments_per_fiber=2)
        mesh=read_msh22_lines(path)
        self.assertEqual(list(mesh['nodes']),[1,2,3,4,5,6])
        self.assertEqual(mesh['nodes'][3],mesh['nodes'][4])
        self.assertEqual(mesh['nodes'][3],tuple(net.fibers[0].end_point))
        self.assertEqual(mesh['nodes'][6],tuple(net.fibers[1].end_point))
        self.assertEqual([e['nodes'] for e in mesh['elements'].values()],[(1,2),(2,3),(4,5),(5,6)])
        self.assertEqual([e['entity'] for e in mesh['elements'].values()],[1,1,2,2])
        self.assertEqual([e['physical'] for e in mesh['elements'].values()],[0,0,0,0])

    def test_gmsh22_reader_rejects_old_header_and_bad_connectivity(self):
        path=self.root/'bad.msh';fea.export_to_gmsh(network(),path,segments_per_fiber=1)
        good=path.read_text()
        for bad in (good.replace('2.2 0 8','4.1 0 8'),good.replace('1 1 2 0 1 1 2','1 1 2 0 1 1 99')):
            path.write_text(bad)
            with self.assertRaises(ValueError): read_msh22_lines(path)

if __name__ == '__main__':
    unittest.main()
