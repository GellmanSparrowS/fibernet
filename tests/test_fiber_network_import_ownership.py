"""Independent material parameter edits after legacy endpoint import stay local."""
import copy

import numpy as np
import pytest

from fibernet import Material, StructureGraph
from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork


def source_network():
    material = Material(youngs_modulus=100., extra={'calibration': [1.]},
                        stiffness_tensor=np.eye(6))
    net = FiberNetwork(dimension=3, box_size=np.array([5., 5., 5.]))
    for start, end in [([0., 0., 0.], [1., 0., 0.]),
                       ([1., 0., 0.], [1., 1., 0.])]:
        net.add_fiber(Fiber.straight(np.array(start), np.array(end),
                                   radius=.2, material=material, segments=7))
    return net


def change(material, field):
    if field == 'modulus':
        material.youngs_modulus = 250.
    elif field == 'nested':
        material.extra['calibration'].append(9.)
    else:
        material.stiffness_tensor[0, 0] = 7.


@pytest.mark.parametrize('field', ['modulus', 'nested', 'tensor'])
@pytest.mark.parametrize('direction', ['source', 'import'])
def test_parameter_edits_do_not_change_source_or_other_imports(field, direction):
    net = source_network()
    first = StructureGraph.from_fiber_network(net)
    second = StructureGraph.from_fiber_network(net)
    source_before = copy.deepcopy(net.to_dict())
    first_before = copy.deepcopy(first.to_dict())
    second_before = copy.deepcopy(second.to_dict())
    if direction == 'source':
        change(net.fibers[0].material, field)
        assert first.to_dict() == first_before
    else:
        change(first.edges[0].material, field)
        assert net.to_dict() == source_before
        # Source fibers sharing one material do not link imported siblings.
        assert first.edges[1].material.to_dict() == second.edges[1].material.to_dict()
    assert second.to_dict() == second_before


def test_import_keeps_existing_endpoint_projection_and_default_material():
    net = source_network()
    net.fibers[0].material = None
    graph = StructureGraph.from_fiber_network(net)
    assert graph.dimension == 3
    np.testing.assert_array_equal(graph.box_size, [5., 5., 5.])
    np.testing.assert_array_equal(graph.node_positions(), [[0., 0., 0.], [1., 0., 0.], [1., 1., 0.]])
    np.testing.assert_array_equal(graph.edge_array(), [[0, 1], [1, 2]])
    assert graph.edges[0].material.to_dict() == Material().to_dict()
    assert net.fibers[0].material is None
    for edge in graph.edges.values():
        assert edge.radius == .2 and edge.segments == 4
        assert edge.internal_points is None
    assert graph.edges[1].material.to_dict() == net.fibers[1].material.to_dict()


@pytest.mark.parametrize('separation', [0., 5e-7])
def test_omitted_collapsed_fiber_does_not_copy_material(separation):
    class UncopyableMaterial(Material):
        def __deepcopy__(self, memo):
            raise AssertionError('omitted fiber must not copy material')

    net = source_network()
    collapsed = Fiber(centerline=np.array([[3., 0., 0.], [3. + separation, 0., 0.]]),
                      radius=.3, material=UncopyableMaterial())
    net.add_fiber(collapsed)
    graph = StructureGraph.from_fiber_network(net, tolerance=1e-6)
    # The endpoint node established by the omitted fiber remains, as before.
    np.testing.assert_array_equal(graph.node_positions(),
                                  [[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [3., 0., 0.]])
    np.testing.assert_array_equal(graph.edge_array(), [[0, 1], [1, 2]])
    assert graph.num_edges == 2
    assert graph.add_node([3. + separation, 0., 0.]) == 3


def test_custom_material_copy_protocol_runs_once_per_retained_fiber():
    calls = []

    class CustomMaterial(Material):
        def __deepcopy__(self, memo):
            calls.append(self.name)
            return CustomMaterial(name=self.name, youngs_modulus=self.youngs_modulus,
                                  extra=copy.deepcopy(self.extra, memo),
                                  stiffness_tensor=self.stiffness_tensor.copy())

    net = source_network()
    source = CustomMaterial(name='shared', youngs_modulus=150.,
                            extra={'nested': [1.]}, stiffness_tensor=np.eye(6))
    for fiber in net.fibers:
        fiber.material = source
    graph = StructureGraph.from_fiber_network(net)
    assert calls == ['shared', 'shared']
    first, second = [edge.material for edge in graph.edges.values()]
    assert isinstance(first, CustomMaterial) and isinstance(second, CustomMaterial)
    first.extra['nested'].append(2.)
    first.stiffness_tensor[0, 0] = 9.
    assert source.extra == second.extra == {'nested': [1.]}
    assert source.stiffness_tensor[0, 0] == second.stiffness_tensor[0, 0] == 1.
    source.youngs_modulus = 200.
    assert first.youngs_modulus == second.youngs_modulus == 150.
