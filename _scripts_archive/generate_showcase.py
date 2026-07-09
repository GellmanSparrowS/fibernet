"""
FiberNet Showcase Generator - Complete Visualization Suite

Generates publication-quality visualizations for all registered generators.
Features checkpoint/resume capability and memory-safe operation.

Usage:
    python generate_showcase.py

Output:
    output_viz/showcase/
    - Category-based visualizations
    - Parametric studies
    - Comparison grids
"""

import sys
import os
import json
import traceback
from pathlib import Path
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).parent))

import fibernet as fn
from fibernet.viz.renderer import (
    render_network_2d,
    render_network_3d,
    render_comparison_grid,
    render_parametric_study,
)


# Configuration
OUTPUT_DIR = Path('output_viz/showcase')
CHECKPOINT_FILE = OUTPUT_DIR / 'checkpoint.json'
DPI = 150
FIGSIZE_2D = (8, 8)
FIGSIZE_3D = (8, 8)


def load_checkpoint():
    """Load checkpoint to resume from last successful step."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'completed': [], 'failed': []}


def save_checkpoint(checkpoint):
    """Save checkpoint."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def safe_generate(category, description, func, checkpoint):
    """Safely execute a generation step with checkpointing."""
    step_id = f"{category}_{description}"
    
    if step_id in checkpoint['completed']:
        print(f"  [SKIP] {step_id} (already completed)")
        return
    
    print(f"  [RUN]  {step_id}")
    try:
        func()
        checkpoint['completed'].append(step_id)
        save_checkpoint(checkpoint)
        print(f"  [OK]   {step_id}")
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        checkpoint['failed'].append({'step': step_id, 'error': error_msg})
        save_checkpoint(checkpoint)
        print(f"  [FAIL] {step_id}: {e}")


def main():
    """Generate complete showcase suite."""
    print("=" * 70)
    print("FiberNet Showcase Generator")
    print("=" * 70)
    
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    print(f"Checkpoint: {len(checkpoint['completed'])} completed, {len(checkpoint['failed'])} failed\n")
    
    # =========================================================================
    # 1. 2D LATTICES
    # =========================================================================
    print("\n[1/13] 2D Lattices")
    cat_dir = OUTPUT_DIR / '01_lattices_2d'
    cat_dir.mkdir(exist_ok=True)
    
    def gen_lattice_2d_topologies():
        nets = []
        titles = []
        for topo in ['square', 'honeycomb', 'triangular', 'kagome']:
            net = fn.create('lattice_2d', topology=topo, cell_size=8.0, grid_size=(6, 6))
            nets.append(net)
            titles.append(topo.capitalize())
        render_comparison_grid(nets, titles, is_3d=False, ncols=4,
                              save_path=cat_dir / 'topologies.png')
    
    safe_generate('lattice_2d', 'topologies', gen_lattice_2d_topologies, checkpoint)
    
    def gen_lattice_2d_perturbation():
        render_parametric_study('lattice_2d', 'perturbation',
                               [0.0, 0.1, 0.2, 0.3],
                               base_params={'topology': 'honeycomb', 'cell_size': 8.0, 'grid_size': (6, 6)},
                               is_3d=False,
                               save_path=cat_dir / 'perturbation_study.png')
    
    safe_generate('lattice_2d', 'perturbation', gen_lattice_2d_perturbation, checkpoint)
    
    def gen_lattice_2d_cell_size():
        render_parametric_study('lattice_2d', 'cell_size',
                               [4.0, 8.0, 12.0, 16.0],
                               base_params={'topology': 'honeycomb', 'grid_size': (6, 6)},
                               is_3d=False,
                               save_path=cat_dir / 'cell_size_study.png')
    
    safe_generate('lattice_2d', 'cell_size', gen_lattice_2d_cell_size, checkpoint)
    
    # =========================================================================
    # 2. 3D LATTICES
    # =========================================================================
    print("\n[2/13] 3D Lattices")
    cat_dir = OUTPUT_DIR / '02_lattices_3d'
    cat_dir.mkdir(exist_ok=True)
    
    def gen_lattice_3d_topologies():
        nets = []
        titles = []
        # Use only working topologies (cubic, octet)
        for topo in ['cubic', 'octet']:
            try:
                net = fn.create('lattice_3d', topology=topo, cell_size=10.0, grid_size=(2, 2, 2))
                nets.append(net)
                titles.append(topo.capitalize())
            except Exception as e:
                print(f"Warning: {topo} failed: {e}")
        if len(nets) > 0:
            render_comparison_grid(nets, titles, is_3d=True, ncols=len(nets),
                                  save_path=cat_dir / 'topologies.png')
    
    safe_generate('lattice_3d', 'topologies', gen_lattice_3d_topologies, checkpoint)
    
    def gen_lattice_3d_grid_size():
        render_parametric_study('lattice_3d', 'grid_size',
                               [(2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5)],
                               base_params={'topology': 'octet', 'cell_size': 10.0},
                               is_3d=True,
                               save_path=cat_dir / 'grid_size_study.png')
    
    safe_generate('lattice_3d', 'grid_size', gen_lattice_3d_grid_size, checkpoint)
    
    # =========================================================================
    # 3. METAMATERIALS
    # =========================================================================
    print("\n[3/13] Metamaterials")
    cat_dir = OUTPUT_DIR / '03_metamaterials'
    cat_dir.mkdir(exist_ok=True)
    
    def gen_metamaterial_modes():
        nets = []
        titles = []
        for mode in ['reentrant', 'chiral', 'star', 'arrowhead', 'missing_rib']:
            net = fn.create('metamaterial_2d', mode=mode, cell_size=10.0, grid_size=(5, 5))
            nets.append(net)
            titles.append(mode.replace('_', ' ').title())
        render_comparison_grid(nets, titles, is_3d=False, ncols=5,
                              save_path=cat_dir / 'modes.png')
    
    safe_generate('metamaterial', 'modes', gen_metamaterial_modes, checkpoint)
    
    def gen_metamaterial_angle():
        render_parametric_study('metamaterial_2d', 'angle',
                               [30, 45, 60, 75, 90],
                               base_params={'mode': 'reentrant', 'cell_size': 10.0, 'grid_size': (5, 5)},
                               is_3d=False,
                               save_path=cat_dir / 'angle_study.png')
    
    safe_generate('metamaterial', 'angle', gen_metamaterial_angle, checkpoint)
    
    # =========================================================================
    # 4. CURVED RANDOM 2D
    # =========================================================================
    print("\n[4/13] Curved Random 2D")
    cat_dir = OUTPUT_DIR / '04_curved_random_2d'
    cat_dir.mkdir(exist_ok=True)
    
    def gen_curved_types():
        nets = []
        titles = []
        for curv_type in ['sinusoidal', 'bezier', 'arc', 'random_walk']:
            net = fn.create('curved_random_2d', num_fibers=50, fiber_length=15.0,
                          curvature_type=curv_type, seed=42)
            nets.append(net)
            titles.append(curv_type.replace('_', ' ').title())
        render_comparison_grid(nets, titles, is_3d=False, ncols=4,
                              save_path=cat_dir / 'curvature_types.png')
    
    safe_generate('curved_random_2d', 'types', gen_curved_types, checkpoint)
    
    def gen_curved_amplitude():
        render_parametric_study('curved_random_2d', 'curvature_amplitude',
                               [1.0, 2.0, 3.0, 5.0, 8.0],
                               base_params={'num_fibers': 50, 'fiber_length': 15.0,
                                          'curvature_type': 'sinusoidal', 'seed': 42},
                               is_3d=False,
                               save_path=cat_dir / 'amplitude_study.png')
    
    safe_generate('curved_random_2d', 'amplitude', gen_curved_amplitude, checkpoint)
    
    # =========================================================================
    # 5. ENTANGLED 3D
    # =========================================================================
    print("\n[5/13] Entangled 3D")
    cat_dir = OUTPUT_DIR / '05_entangled_3d'
    cat_dir.mkdir(exist_ok=True)
    
    def gen_entangled_density():
        render_parametric_study('entangled_3d', 'num_fibers',
                               [20, 40, 60, 80, 100],
                               base_params={'fiber_length': 30.0, 'seed': 42},
                               is_3d=True,
                               save_path=cat_dir / 'density_study.png')
    
    safe_generate('entangled_3d', 'density', gen_entangled_density, checkpoint)
    
    def gen_entangled_curvature():
        render_parametric_study('entangled_3d', 'curvature',
                               [0.1, 0.3, 0.5, 0.7, 1.0],
                               base_params={'num_fibers': 60, 'fiber_length': 30.0, 'seed': 42},
                               is_3d=True,
                               save_path=cat_dir / 'curvature_study.png')
    
    safe_generate('entangled_3d', 'curvature', gen_entangled_curvature, checkpoint)
    
    # =========================================================================
    # 6. VORONOI NETWORKS
    # =========================================================================
    print("\n[6/13] Voronoi Networks")
    cat_dir = OUTPUT_DIR / '06_voronoi'
    cat_dir.mkdir(exist_ok=True)
    
    def gen_voronoi_2d_regularity():
        render_parametric_study('voronoi_2d', 'regularity',
                               [0.0, 0.3, 0.6, 0.9],
                               base_params={'num_seeds': 30},
                               is_3d=False,
                               save_path=cat_dir / 'voronoi_2d_regularity.png')
    
    safe_generate('voronoi', '2d_regularity', gen_voronoi_2d_regularity, checkpoint)
    
    def gen_voronoi_3d_regularity():
        render_parametric_study('voronoi_3d', 'regularity',
                               [0.0, 0.3, 0.6, 0.9],
                               base_params={'num_seeds': 30},
                               is_3d=True,
                               save_path=cat_dir / 'voronoi_3d_regularity.png')
    
    safe_generate('voronoi', '3d_regularity', gen_voronoi_3d_regularity, checkpoint)
    
    # =========================================================================
    # 7. TPMS STRUCTURES
    # =========================================================================
    print("\n[7/13] TPMS Structures")
    cat_dir = OUTPUT_DIR / '07_tpms'
    cat_dir.mkdir(exist_ok=True)
    
    def gen_tpms_types():
        nets = []
        titles = []
        for kind in ['gyroid', 'diamond', 'primitive', 'iwp', 'neovius']:
            net = fn.create('tpms_sheet', kind=kind, resolution=20, box_size=(30, 30, 30))
            nets.append(net)
            titles.append(kind.capitalize())
        render_comparison_grid(nets, titles, is_3d=True, ncols=5,
                              save_path=cat_dir / 'tpms_types.png')
    
    safe_generate('tpms', 'types', gen_tpms_types, checkpoint)
    
    def gen_tpms_lattice():
        nets = []
        titles = []
        for kind in ['gyroid', 'diamond', 'primitive']:
            net = fn.create('tpms_lattice', kind=kind, resolution=12, box_size=(30, 30, 30))
            nets.append(net)
            titles.append(f'{kind.capitalize()} Lattice')
        render_comparison_grid(nets, titles, is_3d=True, ncols=3,
                              save_path=cat_dir / 'tpms_lattice.png')
    
    safe_generate('tpms', 'lattice', gen_tpms_lattice, checkpoint)
    
    def gen_tpms_gradient():
        net = fn.create('tpms_gradient', kind='gyroid', resolution=15,
                       box_size=(30, 30, 30), gradient_axis=0)
        fig = render_network_3d(net, save_path=cat_dir / 'tpms_gradient.png',
                               title='TPMS Gradient (Gyroid)')
    
    safe_generate('tpms', 'gradient', gen_tpms_gradient, checkpoint)
    
    # =========================================================================
    # 8. HIERARCHICAL STRUCTURES
    # =========================================================================
    print("\n[8/13] Hierarchical Structures")
    cat_dir = OUTPUT_DIR / '08_hierarchical'
    cat_dir.mkdir(exist_ok=True)
    
    def gen_hierarchical():
        nets = []
        titles = []
        for base in ['square', 'honeycomb', 'triangular']:
            net = fn.create('hierarchical_lattice', base_topology=base,
                          num_cells=3, levels=2, cell_size=30.0)
            nets.append(net)
            titles.append(f'{base.capitalize()} Hierarchical')
        render_comparison_grid(nets, titles, is_3d=False, ncols=3,
                              save_path=cat_dir / 'hierarchical_types.png')
    
    safe_generate('hierarchical', 'types', gen_hierarchical, checkpoint)
    
    # =========================================================================
    # 9. FRACTAL STRUCTURES
    # =========================================================================
    print("\n[9/13] Fractal Structures")
    cat_dir = OUTPUT_DIR / '09_fractal'
    cat_dir.mkdir(exist_ok=True)
    
    def gen_fractal():
        nets = []
        titles = []
        for frac_gen in ['sierpinski', 'koch_curve', 'hilbert', 'fractal_tree']:
            net = fn.create(frac_gen)
            nets.append(net)
            titles.append(frac_gen.replace('_', ' ').title())
        render_comparison_grid(nets, titles, is_3d=False, ncols=4,
                              save_path=cat_dir / 'fractal_types.png')
    
    safe_generate('fractal', 'types', gen_fractal, checkpoint)
    
    def gen_fractal_network():
        net = fn.create('fractal_network', iterations=3)
        fig = render_network_2d(net, save_path=cat_dir / 'fractal_network.png',
                               title='Fractal Network')
    
    safe_generate('fractal', 'network', gen_fractal_network, checkpoint)
    
    # =========================================================================
    # 10. FIELD-GUIDED NETWORKS
    # =========================================================================
    print("\n[10/13] Field-Guided Networks")
    cat_dir = OUTPUT_DIR / '10_field_guided'
    cat_dir.mkdir(exist_ok=True)
    
    def gen_field_guided():
        net = fn.create('field_guided')
        fig = render_network_2d(net, save_path=cat_dir / 'field_guided.png',
                               title='Field-Guided Network')
    
    safe_generate('field_guided', 'types', gen_field_guided, checkpoint)
    
    # =========================================================================
    # 11. GRADIENT NETWORKS
    # =========================================================================
    print("\n[11/13] Gradient Networks")
    cat_dir = OUTPUT_DIR / '11_gradient'
    cat_dir.mkdir(exist_ok=True)
    
    def gen_gradient():
        # Use tpms_gradient as representative
        net = fn.create('tpms_gradient', kind='gyroid', resolution=10, box_size=(30, 30, 30))
        fig = render_network_3d(net, save_path=cat_dir / 'gradient_tpms.png',
                               title='TPMS Gradient (Gyroid)')
    
    safe_generate('gradient', 'types', gen_gradient, checkpoint)
    
    # =========================================================================
    # 12. RANDOM NETWORKS
    # =========================================================================
    print("\n[12/13] Random Networks")
    cat_dir = OUTPUT_DIR / '12_random'
    cat_dir.mkdir(exist_ok=True)
    
    def gen_random_2d():
        net = fn.create('random_2d', num_fibers=100, fiber_length=10.0, seed=42)
        fig = render_network_2d(net, save_path=cat_dir / 'random_2d.png', title='Random 2D')
    
    safe_generate('random', '2d', gen_random_2d, checkpoint)
    
    def gen_random_3d():
        net = fn.create('random_3d', num_fibers=60, fiber_length=15.0, seed=42)
        fig = render_network_3d(net, save_path=cat_dir / 'random_3d.png', title='Random 3D')
    
    safe_generate('random', '3d', gen_random_3d, checkpoint)
    
    def gen_random_walk():
        net = fn.create('random_walk', num_walks=20, num_steps=50, seed=42)
        fig = render_network_2d(net, save_path=cat_dir / 'random_walk.png',
                               title='Random Walk')
    
    safe_generate('random', 'walk', gen_random_walk, checkpoint)
    
    # =========================================================================
    # 13. SPECIALIZED NETWORKS
    # =========================================================================
    print("\n[13/13] Specialized Networks")
    cat_dir = OUTPUT_DIR / '13_specialized'
    cat_dir.mkdir(exist_ok=True)
    
    def gen_electrospun():
        net = fn.create('electrospun', num_fibers=150, fiber_length=20.0, seed=42)
        fig = render_network_2d(net, save_path=cat_dir / 'electrospun.png',
                               title='Electrospun')
    
    safe_generate('specialized', 'electrospun', gen_electrospun, checkpoint)
    
    def gen_meltblown():
        net = fn.create('meltblown', num_fibers=120, seed=42)
        fig = render_network_2d(net, save_path=cat_dir / 'meltblown.png',
                               title='Meltblown')
    
    safe_generate('specialized', 'meltblown', gen_meltblown, checkpoint)
    
    def gen_paper():
        net = fn.create('paper_network', num_fibers=200, seed=42)
        fig = render_network_2d(net, save_path=cat_dir / 'paper.png', title='Paper Network')
    
    safe_generate('specialized', 'paper', gen_paper, checkpoint)
    
    def gen_foam():
        net = fn.create('foam_like_3d', num_cells=50, seed=42)
        fig = render_network_3d(net, save_path=cat_dir / 'foam_3d.png', title='Foam 3D')
    
    safe_generate('specialized', 'foam', gen_foam, checkpoint)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Completed: {len(checkpoint['completed'])}")
    print(f"Failed:    {len(checkpoint['failed'])}")
    
    if checkpoint['failed']:
        print("\nFailed steps:")
        for fail in checkpoint['failed']:
            print(f"  - {fail['step']}: {fail['error'].split(chr(10))[0]}")
    
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
