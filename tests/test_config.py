"""Tests for YAML-based configuration system."""

import pytest
import tempfile
import os
from pathlib import Path
from fibernet.utils.config import ExperimentConfig, create_template_config, run_from_config


def _tmp_path(suffix):
    """Create a temp file, close it, return the path. Caller must clean up."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path


class TestExperimentConfig:
    def test_create_empty_config(self):
        config = ExperimentConfig()
        assert config.experiment == {}
        assert config.network == {}
        assert config.simulation == {}
    
    def test_create_config_with_data(self):
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
        config = ExperimentConfig(
            experiment={'name': 'test', 'version': '1.0'},
            network={'generator': 'random_straight_2d', 'num_fibers': 100},
            simulation={'type': 'mechanical', 'strain': 0.01}
        )
        
        path = _tmp_path('.yaml')
        try:
            config.to_yaml(path)
            loaded = ExperimentConfig.from_yaml(path)
            assert loaded.experiment['name'] == 'test'
            assert loaded.network['num_fibers'] == 100
        finally:
            Path(path).unlink(missing_ok=True)
    
    def test_save_load_json(self):
        config = ExperimentConfig(
            experiment={'name': 'test', 'version': '1.0'},
            network={'generator': 'random_straight_2d'}
        )
        
        path = _tmp_path('.json')
        try:
            config.to_json(path)
            loaded = ExperimentConfig.from_json(path)
            assert loaded.experiment['name'] == 'test'
        finally:
            Path(path).unlink(missing_ok=True)
    
    def test_validate_valid_config(self):
        config = ExperimentConfig(
            experiment={'name': 'test', 'version': '1.0'},
            network={'generator': 'random_straight_2d', 'num_fibers': 100},
            simulation={'type': 'mechanical', 'strain': 0.01}
        )
        errors = config.validate()
        assert len(errors) == 0
    
    def test_validate_invalid_config(self):
        config = ExperimentConfig()
        errors = config.validate()
        assert len(errors) > 0
    
    def test_update_config(self):
        config = ExperimentConfig(
            experiment={'name': 'test', 'version': '1.0'},
            network={'num_fibers': 100}
        )
        config.update({'experiment': {'version': '2.0'}})
        assert config.experiment['version'] == '2.0'
    
    def test_copy_config(self):
        config = ExperimentConfig(
            experiment={'name': 'test'},
            network={'num_fibers': 100}
        )
        copied = config.copy()
        copied.network['num_fibers'] = 200
        assert config.network['num_fibers'] == 100
        assert copied.network['num_fibers'] == 200
    
    def test_compute_hash(self):
        config = ExperimentConfig(
            experiment={'name': 'test', 'version': '1.0'},
            network={'num_fibers': 100}
        )
        hash1 = config.compute_hash()
        hash2 = config.compute_hash()
        assert hash1 == hash2
        assert len(hash1) == 16
    
    def test_hash_changes_with_content(self):
        config1 = ExperimentConfig(experiment={'name': 'test1'}, network={'num_fibers': 100})
        config2 = ExperimentConfig(experiment={'name': 'test2'}, network={'num_fibers': 100})
        assert config1.compute_hash() != config2.compute_hash()
    
    def test_to_dict(self):
        config = ExperimentConfig(
            experiment={'name': 'test'},
            network={'generator': 'random_straight_2d'}
        )
        d = config.to_dict()
        assert isinstance(d, dict)
        assert 'experiment' in d
        assert 'network' in d
    
    def test_from_dict(self):
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
        pytest.importorskip('yaml')
        path = _tmp_path('.yaml')
        try:
            config = create_template_config(path, template_type='mechanical')
            assert 'name' in config.experiment
            assert 'generator' in config.network
        finally:
            Path(path).unlink(missing_ok=True)
    
    def test_create_thermal_template(self):
        pytest.importorskip('yaml')
        path = _tmp_path('.yaml')
        try:
            config = create_template_config(path, template_type='thermal')
            assert 'name' in config.experiment
        finally:
            Path(path).unlink(missing_ok=True)
    
    def test_create_dma_template(self):
        pytest.importorskip('yaml')
        path = _tmp_path('.yaml')
        try:
            config = create_template_config(path, template_type='dma')
            assert 'name' in config.experiment
        finally:
            Path(path).unlink(missing_ok=True)
    
    def test_create_parametric_template(self):
        pytest.importorskip('yaml')
        path = _tmp_path('.yaml')
        try:
            config = create_template_config(path, template_type='parametric')
            assert 'name' in config.experiment
        finally:
            Path(path).unlink(missing_ok=True)
    
    def test_invalid_template_type(self):
        pytest.importorskip('yaml')
        path = _tmp_path('.yaml')
        try:
            with pytest.raises(ValueError, match="Unknown template type"):
                create_template_config(path, template_type='invalid')
        finally:
            Path(path).unlink(missing_ok=True)


class TestConfigRepr:
    def test_repr(self):
        config = ExperimentConfig(
            experiment={'name': 'test', 'version': '1.0'}
        )
        repr_str = repr(config)
        assert 'test' in repr_str
        assert '1.0' in repr_str
    
    def test_is_valid(self):
        config = ExperimentConfig()
        assert not config.is_valid()
        
        config2 = ExperimentConfig(
            network={'generator': 'random_straight_2d'},
            simulation={'type': 'mechanical'}
        )
        assert config2.is_valid()
