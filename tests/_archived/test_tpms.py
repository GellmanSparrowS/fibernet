"""Tests for TPMS generators."""

import numpy as np
import pytest

from fibernet.gen.tpms import tpms_sheet, tpms_lattice, tpms_gradient


class TestTPMSSheet:
    def test_gyroid_basic(self):
        net = tpms_sheet(kind='gyroid', box_size=(10, 10, 10), resolution=20, seed=42)
        assert net is not None
        assert len(net.fibers) > 0
        assert net.metadata.get('tpms_type') == 'gyroid'
    
    def test_diamond(self):
        net = tpms_sheet(kind='diamond', box_size=(8, 8, 8), resolution=20, seed=42)
        assert net is not None
        assert len(net.fibers) > 0
    
    def test_primitive(self):
        net = tpms_sheet(kind='primitive', box_size=(10, 10, 10), resolution=20, seed=42)
        assert net is not None
        assert len(net.fibers) > 0
    
    def test_invalid_type(self):
        with pytest.raises(ValueError):
            tpms_sheet(kind='invalid')


class TestTPMSLattice:
    def test_gyroid_lattice(self):
        net = tpms_lattice(kind='gyroid', box_size=(10, 10, 10), resolution=20, seed=42)
        assert net is not None
        assert len(net.fibers) >= 0
    
    def test_metadata(self):
        net = tpms_lattice(kind='diamond', box_size=(10, 10, 10), resolution=20, seed=42)
        assert 'tpms_type' in net.metadata
        assert net.metadata['tpms_type'] == 'diamond'


class TestTPMSGradient:
    def test_gradient_gyroid(self):
        net = tpms_gradient(
            kind='gyroid',
            box_size=(20, 10, 10),
            resolution=15,
            gradient_axis=0,
            gradient_range=(0.5, 1.5),
            seed=42,
        )
        assert net is not None
        assert len(net.fibers) >= 0
        assert net.metadata.get('gradient_axis') == 0
    
    def test_gradient_metadata(self):
        net = tpms_gradient(kind='primitive', box_size=(20, 10, 10), resolution=15, seed=42)
        assert 'gradient_range' in net.metadata
