"""Tests for materials database."""

import pytest
from fibernet.materials import get_material, list_materials, compare_materials


class TestMaterialsDatabase:
    """Test materials database functionality."""
    
    def test_list_materials(self):
        """Test listing available materials."""
        materials = list_materials()
        assert len(materials) > 0
        assert 'steel' in materials
        assert 'aluminum' in materials
        assert 'carbon_fiber' in materials
    
    def test_get_material_steel(self):
        """Test getting steel material."""
        steel = get_material('steel')
        assert steel.name == 'steel'
        assert steel.youngs_modulus == 210e9
        assert steel.density == 7850.0
        assert steel.poissons_ratio == 0.28
    
    def test_get_material_aluminum(self):
        """Test getting aluminum material."""
        al = get_material('aluminum')
        assert al.name == 'aluminum'
        assert al.youngs_modulus == 69e9
        assert al.density == 2700.0
    
    def test_get_material_carbon_fiber(self):
        """Test getting carbon fiber material."""
        cf = get_material('carbon_fiber')
        assert cf.name == 'carbon_fiber'
        assert cf.youngs_modulus == 230e9
        assert cf.density == 1800.0
    
    def test_get_material_not_found(self):
        """Test getting non-existent material."""
        with pytest.raises(KeyError):
            get_material('nonexistent_material')
    
    def test_compare_materials(self):
        """Test comparing materials."""
        comp = compare_materials(['steel', 'aluminum', 'carbon_fiber'])
        assert 'stiffest' in comp
        assert 'lightest' in comp
        assert comp['stiffest'] == 'carbon_fiber'
        assert comp['lightest'] == 'carbon_fiber'
    
    def test_all_materials_have_required_properties(self):
        """Test that all materials have required properties."""
        materials = list_materials()
        for mat_name in materials:
            mat = get_material(mat_name)
            assert hasattr(mat, 'name')
            assert hasattr(mat, 'youngs_modulus')
            assert hasattr(mat, 'density')
            assert mat.youngs_modulus > 0
            assert mat.density > 0
