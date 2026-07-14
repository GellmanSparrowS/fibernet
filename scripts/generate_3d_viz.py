#!/usr/bin/env python3
"""
FiberNet v4.1.0 — 3D Visualization Generator

Generates two main visualizations:
1. Gallery of all 14 3D unit types (including undeformed TPMS)
2. Stretch simulation with large deformation (elastomer-like)

Features:
- OOM protection (max_nodes limit)
- Checkpoint/resume for long runs
- Dark theme with visible colorbars
- Re-runnable with fibernet API

Usage:
    cd fibernet && source .venv/bin/activate
    python fibernet/scripts/generate_3d_viz.py
"""

import os
import sys
import json
import gc
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import fibernet as fn

# OOM protection
MAX_NODES = 10000
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'output_data', '3d_validation_v2')
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, '_viz_checkpoint.json')

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"completed": []}


def save_checkpoint(data):
    tmp = CHECKPOINT_FILE + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, CHECKPOINT_FILE)


def generate_gallery():
    """Generate gallery of all 14 unit types (undeformed)."""
    print("Generating gallery...")
    
    # All 14 unit types
    all_units = [
        "cubic", "bcc", "fcc", "hcp", "octet", "diamond_3d",
        "gyroid", "schwarz_p", "schwarz_d", "iwp", "neovius", "lidinoid",
        "chiral_3d", "reentrant_3d"
    ]
    
    graphs = []
    titles = []
    
    for unit_name in all_units:
        try:
            # Special kwargs for some units
            kwargs = {}
            if unit_name == "chiral_3d":
                kwargs["chirality"] = 0.3
            elif unit_name == "reentrant_3d":
                kwargs["angle"] = 15.0
            
            # Generate single unit cell (grid=1x1x1 for gallery)
            g = fn.pattern_3d(unit=unit_name, box=(10, 10, 10), grid=(1, 1, 1), unit_kwargs=kwargs)
            
            # OOM check
            if g.num_nodes > MAX_NODES:
                print(f"  WARNING: {unit_name} has {g.num_nodes} nodes > {MAX_NODES}, skipping")
                continue
            
            graphs.append(g)
            titles.append(unit_name.replace("_", " ").title())
            print(f"  {unit_name}: {g.num_nodes} nodes, {g.num_edges} edges")
            
        except Exception as e:
            print(f"  ERROR generating {unit_name}: {e}")
            continue
        
        gc.collect()
    
    # Render gallery
    try:
        import matplotlib
        matplotlib.use('Agg')
        
        path = os.path.join(OUTPUT_DIR, "gallery_all_3d_structures.png")
        fig = fn.render_gallery_3d(graphs, titles=titles, ncols=4, theme="dark",
                                    save_path=path, dpi=150)
        import matplotlib.pyplot as plt
        plt.close(fig)
        
        print(f"  Gallery saved: {path}")
        return True
    except Exception as e:
        print(f"  ERROR rendering gallery: {e}")
        traceback.print_exc()
        return False


def generate_stretch_simulation():
    """Generate stretch simulation visualization with large deformation."""
    print("\nGenerating stretch simulation (gyroid 2x2x2, 2.0x stretch)...")
    
    try:
        # Generate gyroid 2x2x2
        g = fn.pattern_3d(unit="gyroid", box=(10, 10, 10), grid=(2, 2, 2))
        
        if g.num_nodes > MAX_NODES:
            print(f"  WARNING: {g.num_nodes} nodes > {MAX_NODES}")
            # Reduce to 1x1x1 if too large
            print("  Falling back to 1x1x1 grid")
            g = fn.pattern_3d(unit="gyroid", box=(10, 10, 10), grid=(1, 1, 1))
        
        print(f"  Structure: {g.num_nodes} nodes, {g.num_edges} edges")
        
        # Run stretch test with large deformation (2.0x = 100% strain)
        engine = fn.TaichiEngine()
        result = engine.stretch_test(g, target_stretch=2.0, auto_steps=True)
        
        if result is None:
            print("  ERROR: stretch_test returned None")
            return False
        
        # Compute detailed metrics
        result.compute_detailed(g, stiffness=1e5)
        
        print(f"  Stretch: {result.max_stretch:.3f} max, {result.mean_stretch:.3f} mean")
        print(f"  Energy: {result.energy:.1f}")
        print(f"  Time: {result.time_seconds:.1f}s")
        
        # Render comparison (original vs deformed vs stress)
        import matplotlib
        matplotlib.use('Agg')
        
        path = os.path.join(OUTPUT_DIR, "stretch_simulation_gyroid.png")
        fig = fn.render_comparison_3d(g, result, theme="dark", save_path=path, dpi=150)
        import matplotlib.pyplot as plt
        plt.close(fig)
        
        print(f"  Visualization saved: {path}")
        
        # Verify deformation quality
        nids = sorted(g.nodes.keys())
        orig_pos = np.array([g.nodes[nid].position for nid in nids])
        def_pos = result.deformed_positions
        
        orig_span = orig_pos.max(0) - orig_pos.min(0)
        def_span = def_pos.max(0) - def_pos.min(0)
        actual_stretch = def_span[0] / orig_span[0]
        
        print(f"  Original span: {orig_span.round(2)}")
        print(f"  Deformed span: {def_span.round(2)}")
        print(f"  Actual stretch: {actual_stretch:.3f} (target: 2.0)")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return False


def main():
    print(f"FiberNet v{fn.__version__} — 3D Visualization Generator")
    print(f"Max nodes: {MAX_NODES}")
    print(f"Output: {OUTPUT_DIR}\n")
    
    checkpoint = load_checkpoint()
    completed = set(checkpoint["completed"])
    
    tasks = ["gallery", "stretch"]
    pending = [t for t in tasks if t not in completed]
    
    print(f"Tasks: {len(tasks)} total, {len(completed)} done, {len(pending)} pending\n")
    
    success = True
    
    if "gallery" in pending:
        if generate_gallery():
            completed.add("gallery")
            save_checkpoint({"completed": list(completed)})
        else:
            success = False
    
    if "stretch" in pending:
        if generate_stretch_simulation():
            completed.add("stretch")
            save_checkpoint({"completed": list(completed)})
        else:
            success = False
    
    print(f"\n{'='*70}")
    if success:
        print("All visualizations generated successfully!")
    else:
        print("Some visualizations failed. Check errors above.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
