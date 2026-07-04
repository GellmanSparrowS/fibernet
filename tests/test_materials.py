"""Tests for materials database."""

import pytest
from fibernet.materials import (
    get_material, list_materials, carbon_fiber, glass_fiber,
    collagen_fiber, spider_silk, polymer_fiber, metal_fiber
)
from fibernet.core.material import Material


class TestMaterialDatabase:
    def test_list_materials(self):
        """Test listing available materials."""
        materials = list_materials()
        assert len(materials) == 10
        assert 'carbon' in materials
        assert 'glass' in materials
        assert 'collagen' in materials
    
    def test_get_material_valid(self):
        """Test getting a valid material."""
        mat = get_material('carbon')
        assert isinstance(mat, Material)
        assert mat.youngs_modulus > 0
        assert mat.density > 0
    
    def test_get_material_invalid(self):
        """Test getting an invalid material raises error."""
        with pytest.raises(ValueError, match="Unknown material"):
            get_material('nonexistent')
    
    def test_carbon_fiber_grades(self):
        """Test different carbon fiber grades."""
        standard = carbon_fiber('standard')
        high_mod = carbon_fiber('high_modulus')
        
        assert high_mod.youngs_modulus > standard.youngs_modulus
        assert high_mod.density > standard.density
    
    def test_glass_fiber_types(self):
        """Test different glass fiber types."""
        e_glass = glass_fiber('E-glass')
        s_glass = glass_fiber('S-glass')
        
        assert s_glass.youngs_modulus > e_glass.youngs_modulus
    
    def test_collagen_properties(self):
        """Test collagen fiber has biological properties."""
        collagen = collagen_fiber()
        
        assert collagen.youngs_modulus < 10e9  # Soft material
        assert 'persistence_length' in collagen.extra
        assert collagen.extra['persistence_length'] > 0
    
    def test_spider_silk_toughness(self):
        """Test spider silk has high toughness."""
        dragline = spider_silk('dragline')
        
        assert 'toughness' in dragline.extra
        assert dragline.extra['toughness'] > 100e6  # Very tough
    
    def test_polymer_fiber_types(self):
        """Test different polymer fibers."""
        nylon = polymer_fiber('nylon')
        uhmwpe = polymer_fiber('UHMWPE')
        
        assert uhmwpe.youngs_modulus > nylon.youngs_modulus
    
    def test_metal_fiber_types(self):
        """Test different metal fibers."""
        steel = metal_fiber('steel')
        aluminum = metal_fiber('aluminum')
        
        assert steel.youngs_modulus > aluminum.youngs_modulus
        assert steel.density > aluminum.density
    
    def test_material_with_generator(self):
        """Test using predefined material with generator."""
        from fibernet import gen
        
        cf = get_material('carbon', grade='high_strength')
        net = gen.random_straight_2d(num_fibers=5, fiber_length=10, box_size=(20, 20), material=cf, seed=42)
        
        assert net.fibers[0].material.name == 'Carbon Fiber (high_strength)'
        assert net.fibers[0].material.youngs_modulus == cf.youngs_modulus
    
    def test_material_thermal_properties(self):
        """Test thermal properties are set."""
        mat = get_material('glass')
        
        assert mat.thermal_conductivity is not None
        assert mat.specific_heat is not None
        assert mat.thermal_expansion is not None
    
    def test_material_electrical_properties(self):
        """Test electrical properties for conductive materials."""
        cf = get_material('carbon')
        gf = get_material('glass')
        
        # Carbon is conductive
        assert cf.electrical_conductivity > 0
        # Glass is insulator
        assert gf.electrical_conductivity < 1e-10
