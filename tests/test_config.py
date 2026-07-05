"""Tests for YAML-based configuration system."""

import pytest
import tempfile
from pathlib import Path
from fibernet.utils.config import ExperimentConfig, create_template_config, run_from_config


class TestExperimentConfig:
    def test_create_empty_config(self):
        """Test creating empty configuration"""
        config = ExperimentConfig()
        assert config.experiment == {}
        assert config.network == {}
        assert config.simulation == {}
    
    def test_create_config_with_data(self):
        """Test creating configuration with data"""
        config = ExperimentConfig(
            experiment={'name': 'test', 'version': '1.0'},
            network={'generator': 'random_straight_2d'},
            simulation={'type': 'mechanical'}
        )
        assert config.experiment['name'] == 'test'
        assert config.network['generator'] == 'random_straight_2d'
        assert config.simulation['type'] == 'mechanical'
    
    def test_save_load_yaml(self):
        pytest.importorskip("yaml")
        """Test saving and loading YAML configuration"""
        config = ExperimentConfig(
            experiment={'name': 'test', 'version': '1.0'},
            network={'generator': 'random_straight_2d', 'num_fibers': 100},
            simulation={'type': 'mechanical', 'strain': 0.01}
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.to_yaml(f.name)
            loaded = ExperimentConfig.from_yaml(f.name)
            
            assert loaded.experiment['name'] == 'test'
            assert loaded.network['num_fibers'] == 100
            
            Path(f.name).unlink()
    
    def test_save_load_json(self):
        """Test saving and loading JSON configuration"""
        config = ExperimentConfig(
            experiment={'name': 'test', 'version': '1.0'},
            network={'generator': 'random_straight_2d'}
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.to_json(f.name)
            loaded = ExperimentConfig.from_json(f.name)
            
            assert loaded.experiment['name'] == 'test'
            
            Path(f.name).unlink()
    
    def test_validate_valid_config(self):
        """Test validation of valid configuration"""
        config = ExperimentConfig(
            experiment={'name': 'test', 'version': '1.0'},
            network={'generator': 'random_straight_2d', 'num_fibers': 100},
            simulation={'type': 'mechanical', 'strain': 0.01}
        )
        errors = config.validate()
        assert len(errors) == 0
    
    def test_validate_invalid_config(self):
        """Test validation catches errors"""
        config = ExperimentConfig()
        errors = config.validate()
        assert len(errors) > 0
    
    def test_update_config(self):
        """Test updating configuration"""
        config = ExperimentConfig(
            experiment={'name': 'test', 'version': '1.0'},
            network={'num_fibers': 100}
        )
        config.update({'experiment': {'version': '2.0'}})
        assert config.experiment['version'] == '2.0'
    
    def test_copy_config(self):
        """Test copying configuration"""
        config = ExperimentConfig(
            experiment={'name': 'test'},
            network={'num_fibers': 100}
        )
        copied = config.copy()
        copied.network['num_fibers'] = 200
        
        assert config.network['num_fibers'] == 100
        assert copied.network['num_fibers'] == 200
    
    def test_compute_hash(self):
        """Test hash computation"""
        config = ExperimentConfig(
            experiment={'name': 'test', 'version': '1.0'},
            network={'num_fibers': 100}
        )
        hash1 = config.compute_hash()
        hash2 = config.compute_hash()
        
        assert hash1 == hash2
        assert len(hash1) == 16  # SHA256 truncated to 16 chars
    
    def test_hash_changes_with_content(self):
        """Test hash changes when content changes"""
        config1 = ExperimentConfig(experiment={'name': 'test1'}, network={'num_fibers': 100})
        config2 = ExperimentConfig(experiment={'name': 'test2'}, network={'num_fibers': 100})
        assert config1.compute_hash() != config2.compute_hash()
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        config = ExperimentConfig(
            experiment={'name': 'test'},
            network={'generator': 'random_straight_2d'}
        )
        d = config.to_dict()
        assert isinstance(d, dict)
        assert 'experiment' in d
        assert 'network' in d
    
    def test_from_dict(self):
        """Test creation from dictionary"""
        data = {
            'experiment': {'name': 'test'},
            'network': {'generator': 'random_straight_2d'},
            'simulation': {'type': 'mechanical'}
        }
        config = ExperimentConfig.from_dict(data)
        assert config.experiment['name'] == 'test'
        assert config.network['generator'] == 'random_straight_2d'


class TestTemplateConfigs:
    def test_create_mechanical_template(self):
        """Test creating mechanical simulation template"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = create_template_config(f.name, template_type='mechanical')
            assert 'name' in config.experiment
            assert 'generator' in config.network
            Path(f.name).unlink()
    
    def test_create_thermal_template(self):
        """Test creating thermal simulation template"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = create_template_config(f.name, template_type='thermal')
            assert 'name' in config.experiment
            Path(f.name).unlink()
    
    def test_create_dma_template(self):
        """Test creating DMA simulation template"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = create_template_config(f.name, template_type='dma')
            assert 'name' in config.experiment
            Path(f.name).unlink()
    
    def test_create_parametric_template(self):
        """Test creating parametric study template"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = create_template_config(f.name, template_type='parametric')
            assert 'name' in config.experiment
            Path(f.name).unlink()
    
    def test_invalid_template_type(self):
        """Test invalid template type raises error"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            with pytest.raises(ValueError, match="Unknown template type"):
                create_template_config(f.name, template_type='invalid')
            Path(f.name).unlink()


class TestConfigRepr:
    def test_repr(self):
        """Test string representation"""
        config = ExperimentConfig(
            experiment={'name': 'test', 'version': '1.0'}
        )
        repr_str = repr(config)
        assert 'test' in repr_str
        assert '1.0' in repr_str
    
    def test_is_valid(self):
        """Test is_valid method"""
        config = ExperimentConfig()
        assert not config.is_valid()
        
        config2 = ExperimentConfig(
            network={'generator': 'random_straight_2d'},
            simulation={'type': 'mechanical'}
        )
        assert config2.is_valid()
