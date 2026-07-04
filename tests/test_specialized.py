"""Tests for specialized generators."""

import numpy as np
import pytest
from fibernet.gen.specialized import (
    cnt_network_2d,
    cnt_network_3d,
    paper_network,
    textile_weave,
    electrospun_mat,
    fiber_reinforced_composite,
)


class TestCNTNetworks:
    def test_cnt_2d(self):
        net = cnt_network_2d(num_tubes=20, tube_length=5, box_size=(30, 30), seed=42)
        assert net.num_fibers > 0
        assert net.dimension == 2
    
    def test_cnt_2d_bundled(self):
        net = cnt_network_2d(num_tubes=10, tube_length=5, bundle_size=3, seed=42)
        assert net.num_fibers == 30
    
    def test_cnt_3d(self):
        net = cnt_network_3d(num_tubes=20, tube_length=5, box_size=(30, 30, 30), seed=42)
        assert net.num_fibers > 0
        assert net.dimension == 3


class TestPaperNetwork:
    def test_paper_basic(self):
        net = paper_network(num_fibers=50, fiber_length=10, box_size=(30, 30), seed=42)
        assert net.num_fibers == 50
        assert net.dimension == 2
    
    def test_paper_curly(self):
        net = paper_network(num_fibers=30, fiber_length=10, curliness=0.8, seed=42)
        assert net.num_fibers == 30


class TestTextileWeave:
    def test_plain_weave(self):
        net = textile_weave(warp_count=5, weft_count=5, weave_pattern="plain", seed=42)
        assert net.num_fibers == 10
    
    def test_twill_weave(self):
        net = textile_weave(warp_count=5, weft_count=5, weave_pattern="twill", seed=42)
        assert net.num_fibers == 10
    
    def test_satin_weave(self):
        net = textile_weave(warp_count=5, weft_count=5, weave_pattern="satin", seed=42)
        assert net.num_fibers == 10


class TestElectrospunMat:
    def test_electrospun_basic(self):
        net = electrospun_mat(num_fibers=50, box_size=(30, 30), seed=42)
        assert net.num_fibers == 50
    
    def test_electrospun_aligned(self):
        net = electrospun_mat(num_fibers=30, alignment=0.8, seed=42)
        assert net.num_fibers == 30


class TestFiberReinforcedComposite:
    def test_unidirectional(self):
        net = fiber_reinforced_composite(
            matrix_size=(30, 30, 10),
            fiber_volume_fraction=0.3,
            fiber_orientation="unidirectional",
            seed=42,
        )
        assert net.num_fibers > 0
        assert net.dimension == 3
    
    def test_random(self):
        net = fiber_reinforced_composite(
            matrix_size=(30, 30, 10),
            fiber_volume_fraction=0.2,
            fiber_orientation="random",
            fiber_length=10.0,
            seed=42,
        )
        assert net.num_fibers > 0
