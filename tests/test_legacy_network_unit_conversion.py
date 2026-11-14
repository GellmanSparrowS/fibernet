"""Legacy network unit conversions preserve physical relationships and ownership."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from fibernet.core.fiber import Fiber, CrossSection
from fibernet.core.material import Material
from fibernet.core.network import Crosslink, FiberNetwork
from fibernet.utils.units import UnitSystem, UnitConverter, convert_network_units


def fixture():
    mat = Material(density=1000.0, youngs_modulus=100.0, shear_modulus=40.0,
                   yield_strength=2.0, tensile_strength=3.0,
                   fracture_toughness=4.0, thermal_conductivity=5.0,
                   specific_heat=6.0, thermal_expansion=7.0,
                   stiffness_tensor=np.eye(6) * 8.0,
                   permittivity=2.0, permeability=3.0,
                   extra={'label': 'synthetic', 'tags': ['test'], 'enabled': True})
    first = Fiber(np.array([[0., 0., 0.], [1., 0., 0.]]), radius=.01,
                  material=mat, fiber_id=17)
    second = Fiber(np.array([[1., 0., 0.], [1., 1., 0.]]), radius=.02,
                   material=mat, fiber_id=93)
    crosslink = Crosslink(0, 1, 1., 0., np.array([1., 0., 0.]),
                          strength=9., stiffness=10.)
    return FiberNetwork([first, second], [crosslink], np.array([2., 3., 4.]),
                        True, 3, {'label': 'synthetic', 'samples': 2})


class LegacyNetworkUnitConversionTests(unittest.TestCase):
    def test_undeclared_attributes_fail_without_input_mutation(self):
        for target, attribute, value in [('network', 'loads', np.ones(3)),
                                         ('fiber', 'prestress', 2.),
                                         ('material', 'viscosity', 2.),
                                         ('crosslink', 'damping', 2.)]:
            net = fixture()
            obj = {'network': net, 'fiber': net.fibers[0],
                   'material': net.fibers[0].material,
                   'crosslink': net.crosslinks[0]}[target]
            setattr(obj, attribute, value)
            before = copy.deepcopy(net.to_dict())
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, 'Undeclared'):
                convert_network_units(net, 'SI', 'CGS')
            self.assertEqual(net.to_dict(), before)
            self.assertIs(getattr(obj, attribute), value)

    def test_subclasses_are_explicitly_rejected(self):
        class DerivedNetwork(FiberNetwork):
            pass
        class DerivedFiber(Fiber):
            pass
        class DerivedMaterial(Material):
            pass
        class DerivedCrosslink(Crosslink):
            pass
        with self.assertRaises(TypeError):
            convert_network_units(DerivedNetwork(), 'SI', 'CGS')
        for target in ('fiber', 'material', 'crosslink'):
            net = fixture()
            if target == 'fiber':
                net.fibers[0] = DerivedFiber(np.zeros((2, 3)))
            elif target == 'material':
                net.fibers[0].material = DerivedMaterial()
            else:
                net.crosslinks[0] = DerivedCrosslink(0, 1, 1., 0., np.zeros(3))
            with self.subTest(target=target), self.assertRaises(TypeError):
                convert_network_units(net, 'SI', 'CGS')

    def test_invalid_and_extreme_unit_factors_raise_valueerror(self):
        values = (0., -1., float('nan'), float('inf'), True, '1', 1j,
                  [1.], 10**400, 1e-300, 1e300)
        for value in values:
            system = UnitSystem('temporary', time_to_si=value)
            with self.subTest(value=repr(value)), patch.dict(UnitConverter._systems,
                                                            {'TEMPORARY': system}):
                net = fixture()
                before = copy.deepcopy(net.to_dict())
                with self.assertRaises(ValueError):
                    UnitConverter.convert(1., 'force', 'TEMPORARY', 'SI')
                with self.assertRaises(ValueError):
                    convert_network_units(net, 'TEMPORARY', 'SI')
                self.assertEqual(net.to_dict(), before)

    def test_area_moment_mass_and_spring_force_dimensional_invariants(self):
        net = fixture()
        net.fibers[1].cross_section = CrossSection.RECTANGULAR
        net.fibers[1].cross_section_params = {'width': .03, 'height': .02}
        out = convert_network_units(net, 'SI', 'CGS')
        for a, b in zip(net.fibers, out.fibers):
            self.assertAlmostEqual(b.cross_section_area / a.cross_section_area, 10000.)
            self.assertAlmostEqual(b.second_moment_area[0] / a.second_moment_area[0], 1e8)
            original_mass = a.material.density * a.cross_section_area * a.length
            converted_mass = b.material.density * b.cross_section_area * b.length
            self.assertAlmostEqual(converted_mass / original_mass, 1000.)
        original_force = net.crosslinks[0].stiffness * .001
        converted_force = out.crosslinks[0].stiffness * .1
        self.assertAlmostEqual(converted_force / original_force, 100000.)

    def test_numpy_integer_base_factors_do_not_wrap(self):
        integer = np.int64(4294967297)
        for field in ('length_to_si', 'mass_to_si', 'time_to_si', 'temperature_to_si'):
            system = UnitSystem('integer_base', **{field: integer})
            float_system = UnitSystem('float_base', **{field: float(integer)})
            with self.subTest(field=field), patch.dict(UnitConverter._systems,
                    {'INTEGER_BASE': system, 'FLOAT_BASE': float_system}):
                for quantity in ('length', 'mass', 'time', 'force', 'energy',
                                 'stress', 'velocity', 'viscosity'):
                    self.assertEqual(UnitConverter.convert(1., quantity, 'INTEGER_BASE', 'SI'),
                                     UnitConverter.convert(1., quantity, 'FLOAT_BASE', 'SI'))
                net = fixture()
                integer_out = convert_network_units(net, 'INTEGER_BASE', 'SI')
                float_out = convert_network_units(net, 'FLOAT_BASE', 'SI')
                self.assertEqual(integer_out.to_dict(), float_out.to_dict())
        system = UnitSystem('integer_time', time_to_si=integer)
        expected = 1.0 / (4294967297.0 * 4294967297.0)
        self.assertEqual(system.force_to_si, expected)
        with patch.dict(UnitConverter._systems, {'INTEGER_TIME': system}):
            self.assertEqual(UnitConverter.convert(1., 'force', 'INTEGER_TIME', 'SI'), expected)
            out = convert_network_units(fixture(), 'INTEGER_TIME', 'SI')
            self.assertEqual(out.crosslinks[0].strength, 9.0 * expected)

    def test_nontrivial_temperature_scale(self):
        # One unit of this synthetic temperature scale equals two kelvin.
        system = UnitSystem('two_kelvin', temperature_to_si=2.)
        with patch.dict(UnitConverter._systems, {'TWO_KELVIN': system}):
            net = fixture()
            out = convert_network_units(net, 'SI', 'TWO_KELVIN')
            mat = out.fibers[0].material
            self.assertEqual(mat.thermal_conductivity, 10.)
            self.assertEqual(mat.specific_heat, 12.)
            self.assertEqual(mat.thermal_expansion, 14.)
            self.assertEqual(mat.youngs_modulus, 100.)
            restored = convert_network_units(out, 'TWO_KELVIN', 'SI')
            self.assertEqual(restored.fibers[0].material.thermal_conductivity, 5.)
            self.assertEqual(net.fibers[0].material.thermal_conductivity, 5.)

    def test_unknown_systems_and_quantities_fail_closed(self):
        for bad in ('sii', '', None, 3, []):
            with self.subTest(system=bad):
                with self.assertRaises(ValueError):
                    UnitConverter.convert(2., 'length', bad, 'SI')
                with self.assertRaises(ValueError):
                    UnitConverter.convert(2., 'length', 'SI', bad)
        for quantity in ('lenght', '', None, []):
            with self.subTest(quantity=quantity), self.assertRaises(ValueError):
                UnitConverter.convert(2., quantity, 'SI', 'CGS')

    def test_existing_scalar_and_array_entrypoints(self):
        self.assertEqual(UnitConverter.convert(1., 'length', 'si', 'cgs'), 100.)
        self.assertEqual(UnitConverter.to_si(100., 'length', 'CGS'), 1.)
        # 1 N = 10^5 dyn. The decimal base factors and their product/ratio
        # round in binary64; allow four machine epsilons, with no absolute floor.
        np.testing.assert_allclose(UnitConverter.from_si(1., 'force', 'CGS'),
                                   100000., rtol=4 * np.finfo(float).eps, atol=0.)
        np.testing.assert_array_equal(UnitConverter.convert(np.array([1., 2.]),
                                                           'length', 'SI', 'CGS'),
                                      [100., 200.])

    def test_shared_material_converts_once_and_input_is_unchanged(self):
        net = fixture()
        before = copy.deepcopy(net.to_dict())
        out = convert_network_units(net, 'SI', 'CGS')
        self.assertIs(out.fibers[0].material, out.fibers[1].material)
        self.assertIsNot(out.fibers[0].material, net.fibers[0].material)
        self.assertEqual(out.fibers[0].material.youngs_modulus, 1000.)
        self.assertEqual(out.fibers[0].material.density, 1.)
        self.assertEqual(net.to_dict(), before)

    def test_all_supported_material_dimensions(self):
        out = convert_network_units(fixture(), 'SI', 'CGS')
        mat = out.fibers[0].material
        for field, expected in {'youngs_modulus': 1000., 'shear_modulus': 400.,
                                'density': 1., 'yield_strength': 20.,
                                'tensile_strength': 30., 'fracture_toughness': 400.,
                                'thermal_conductivity': 500000.,
                                'specific_heat': 60000., 'thermal_expansion': 7.,
                                'permittivity': 2., 'permeability': 3.}.items():
            with self.subTest(field=field):
                self.assertAlmostEqual(getattr(mat, field), expected)
        np.testing.assert_array_equal(mat.stiffness_tensor, np.eye(6) * 80.)
        self.assertEqual(mat.extra, fixture().fibers[0].material.extra)

    def test_crosslink_geometry_force_stiffness_and_ids(self):
        out = convert_network_units(fixture(), 'SI', 'CGS')
        cl = out.crosslinks[0]
        np.testing.assert_array_equal(cl.position, out.fibers[0].end_point)
        np.testing.assert_array_equal(cl.position, out.fibers[1].start_point)
        np.testing.assert_array_equal(out.box_size, [200., 300., 400.])
        self.assertEqual([f.radius for f in out.fibers], [1., 2.])
        self.assertEqual([f.fiber_id for f in out.fibers], [17, 93])
        self.assertEqual((cl.fiber_i, cl.fiber_j, cl.param_i, cl.param_j), (0, 1, 1., 0.))
        self.assertEqual(cl.strength, 900000.)
        self.assertEqual(cl.stiffness, 10000.)
        self.assertEqual(out.metadata, {'label': 'synthetic', 'samples': 2,
                                        'unit_system': 'CGS'})

    def test_repeated_fiber_and_crosslink_references_scale_once(self):
        net = fixture()
        net.fibers.append(net.fibers[0])
        net.crosslinks.append(net.crosslinks[0])
        out = convert_network_units(net, 'SI', 'CGS')
        self.assertIs(out.fibers[0], out.fibers[2])
        self.assertIs(out.crosslinks[0], out.crosslinks[1])
        self.assertEqual(out.fibers[0].radius, 1.)
        self.assertEqual(out.crosslinks[0].strength, 900000.)

    def test_supported_section_parameters_scale(self):
        for shape, params in [(CrossSection.RECTANGULAR, {'width': 2., 'height': 3.}),
                              (CrossSection.ELLIPTICAL, {'semi_minor': .2}),
                              (CrossSection.HOLLOW_CIRCULAR, {'inner_radius': .1})]:
            with self.subTest(shape=shape):
                net = fixture()
                net.fibers[0].cross_section = shape
                net.fibers[0].cross_section_params = params
                out = convert_network_units(net, 'SI', 'CGS')
                self.assertEqual(out.fibers[0].cross_section_params,
                                 {key: value * 100 for key, value in params.items()})
                self.assertEqual(net.fibers[0].cross_section_params, params)

    def test_round_trip_and_identity_are_independent_copies(self):
        net = fixture()
        for target in ('SI', 'CGS', 'MICRO', 'NANO', 'MOLECULAR'):
            with self.subTest(target=target):
                out = convert_network_units(convert_network_units(net, 'SI', target),
                                            target, 'SI')
                for a, b in zip(out.fibers, net.fibers):
                    np.testing.assert_allclose(a.centerline, b.centerline, rtol=1e-12)
                    self.assertAlmostEqual(a.radius, b.radius)
                    self.assertAlmostEqual(a.material.youngs_modulus,
                                           b.material.youngs_modulus)
                np.testing.assert_allclose(out.crosslinks[0].position,
                                           net.crosslinks[0].position, rtol=1e-12)
                self.assertIsNot(out, net)

    def test_empty_network_and_infinite_crosslinks(self):
        self.assertEqual(convert_network_units(FiberNetwork(), 'SI', 'CGS').fibers, [])
        net = fixture()
        net.crosslinks[0].strength = net.crosslinks[0].stiffness = float('inf')
        out = convert_network_units(net, 'SI', 'CGS')
        self.assertEqual(out.crosslinks[0].strength, float('inf'))
        self.assertEqual(out.crosslinks[0].stiffness, float('inf'))

    def test_unsupported_material_properties_reject_without_mutation(self):
        for field, value in [('electrical_conductivity', 0.), ('extra', {'x': 3.}),
                             ('extra', {'nested': [1.]})]:
            with self.subTest(field=field, value=value):
                net = fixture()
                setattr(net.fibers[0].material, field, value)
                before = copy.deepcopy(net.to_dict())
                with self.assertRaises(ValueError):
                    convert_network_units(net, 'SI', 'CGS')
                self.assertEqual(net.to_dict(), before)

    def test_missing_material_unknown_sections_and_nonfinite_reject(self):
        net = fixture()
        net.fibers[0].material = None
        with self.assertRaises(TypeError):
            convert_network_units(net, 'SI', 'CGS')
        self.assertIsNone(net.fibers[0].material)
        for shape, params in [(CrossSection.CUSTOM, {}), (CrossSection.TRIANGULAR, {}),
                              (CrossSection.CIRCULAR, {'unexpected': 1.})]:
            net = fixture()
            net.fibers[0].cross_section, net.fibers[0].cross_section_params = shape, params
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                convert_network_units(net, 'SI', 'CGS')
        for value in (float('nan'), float('-inf'), 1e308):
            net = fixture()
            net.fibers[0].radius = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                convert_network_units(net, 'SI', 'CGS')


if __name__ == '__main__':
    unittest.main()
