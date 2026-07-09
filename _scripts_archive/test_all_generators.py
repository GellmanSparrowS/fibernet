"""Comprehensive test of all generators - data analysis only, no visualization."""
import sys, os, time, traceback
sys.path.insert(0, '.')
import numpy as np
import fibernet as fn

# ============================================================
# Part 1: Test registered generators via fn.create()
# ============================================================
def test_registered_generators():
    """Test all 40 registered generators with default params."""
    gens = fn.list_generators()
    results = {}
    
    for name in gens:
        try:
            t0 = time.time()
            net = fn.create(name)
            dt = time.time() - t0
            
            n_fibers = net.num_fibers
            n_cl = net.num_crosslinks
            dim = net.dimension
            bb_min, bb_max = net.bounding_box()
            size = bb_max - bb_min
            
            # Graph analysis
            from fibernet.analysis.graph_features import GraphFeatureExtractor
            try:
                # Use networkx graph directly
                import networkx as nx
                if hasattr(net, 'to_networkx'):
                    G = net.to_networkx()
                elif hasattr(net, 'graph'):
                    G = net.graph
                else:
                    G = None
                
                if G is not None and isinstance(G, nx.Graph):
                    n_nodes = G.number_of_nodes()
                    n_edges = G.number_of_edges()
                    density = nx.density(G)
                    if n_nodes > 0:
                        degrees = [d for _, d in G.degree()]
                        avg_deg = np.mean(degrees)
                        max_deg = max(degrees)
                        min_deg = min(degrees)
                        components = nx.number_connected_components(G)
                    else:
                        avg_deg = max_deg = min_deg = 0
                        components = 0
                        density = 0
                else:
                    n_nodes = n_edges = 0
                    avg_deg = max_deg = min_deg = components = density = 0
            except Exception as e2:
                n_nodes = n_edges = 0
                avg_deg = max_deg = min_deg = components = density = 0
                
            results[name] = {
                'status': 'OK',
                'time': dt,
                'fibers': n_fibers,
                'crosslinks': n_cl,
                'dim': dim,
                'size': size,
                'nodes': n_nodes,
                'edges': n_edges,
                'avg_deg': avg_deg,
                'max_deg': max_deg,
                'components': components,
                'density': density,
            }
        except Exception as e:
            results[name] = {
                'status': 'FAIL',
                'error': str(e)[:200],
                'traceback': traceback.format_exc()[-500:],
            }
    
    return results


# ============================================================
# Part 2: Test unregistered generators from gen submodules
# ============================================================
def test_unregistered_generators():
    """Test generators available in gen module but not in registry."""
    from fibernet import gen
    results = {}
    
    # Collect all callable generators from gen module
    extra_gens = {}
    
    # bundles
    for name in ['parallel_bundle_2d', 'twisted_bundle_2d', 'random_bundle_3d', 
                 'braided_bundle_3d', 'tendon_like_bundle_3d']:
        if hasattr(gen, name):
            extra_gens[f'gen.{name}'] = getattr(gen, name)
    
    # curved
    for name in ['sinusoidal_fiber_2d', 'helical_fiber_3d', 'arc_fiber_2d',
                 'bezier_fiber_3d', 'random_curved_network_3d', 'crimped_network_2d']:
        if hasattr(gen, name):
            extra_gens[f'gen.{name}'] = getattr(gen, name)
    
    # laminates
    for name in ['unidirectional_laminate', 'crossply_laminate', 'angle_ply_laminate',
                 'quasi_isotropic_laminate', 'custom_laminate', 'sandwich_laminate']:
        if hasattr(gen, name):
            extra_gens[f'gen.{name}'] = getattr(gen, name)
    
    # gradient
    for name in ['density_gradient_2d', 'property_gradient_2d', 'multi_zone_2d']:
        if hasattr(gen, name):
            extra_gens[f'gen.{name}'] = getattr(gen, name)
    
    # fractal
    for name in ['sierpinski_triangle', 'koch_curve', 'fractal_tree', 'hilbert_curve']:
        if hasattr(gen, name):
            extra_gens[f'gen.{name}'] = getattr(gen, name)
    
    # disordered extras
    for name in ['poisson_line_network_2d', 'random_curved_fibers_3d']:
        if hasattr(gen, name):
            extra_gens[f'gen.{name}'] = getattr(gen, name)
    
    # specialized
    for name in ['cnt_network_2d', 'cnt_network_3d', 'paper_network', 
                 'textile_weave', 'electrospun_mat', 'fiber_reinforced_composite']:
        if hasattr(gen, name):
            extra_gens[f'gen.{name}'] = getattr(gen, name)
    
    # variants
    for name in ['lattice_2d_to_3d', 'curved_lattice', 'multi_radius_network',
                 'variable_stiffness_network', 'gyroid_infill', 'foam_like_3d']:
        if hasattr(gen, name):
            extra_gens[f'gen.{name}'] = getattr(gen, name)
    
    # woven extras
    for name in ['satin_weave_2d', 'woven_3d_orthogonal']:
        if hasattr(gen, name):
            extra_gens[f'gen.{name}'] = getattr(gen, name)
    
    # chiral extras
    for name in ['chiral_metamaterial']:
        if hasattr(gen, name):
            extra_gens[f'gen.{name}'] = getattr(gen, name)
    
    # advanced extras
    for name in ['voronoi_network_3d', 'meltblown_network', 'defected_lattice',
                 'composite_network', 'graded_network', 'auxetic_structure', 'kirigami_structure']:
        if hasattr(gen, name):
            extra_gens[f'gen.{name}'] = getattr(gen, name)
    
    # hierarchical extras
    for name in ['hierarchical_bundle', 'gradient_density_network', 'core_shell_fiber', 'fractal_network']:
        if hasattr(gen, name):
            extra_gens[f'gen.{name}'] = getattr(gen, name)
    
    # tpms
    for name in ['tpms_sheet', 'tpms_lattice', 'tpms_gradient']:
        if hasattr(gen, name):
            extra_gens[f'gen.{name}'] = getattr(gen, name)

    for full_name, func in extra_gens.items():
        try:
            t0 = time.time()
            net = func()
            dt = time.time() - t0
            
            n_fibers = net.num_fibers
            n_cl = net.num_crosslinks
            dim = net.dimension
            bb_min, bb_max = net.bounding_box()
            size = bb_max - bb_min
            
            results[full_name] = {
                'status': 'OK',
                'time': dt,
                'fibers': n_fibers,
                'crosslinks': n_cl,
                'dim': dim,
                'size': size,
            }
        except Exception as e:
            results[full_name] = {
                'status': 'FAIL',
                'error': str(e)[:200],
                'traceback': traceback.format_exc()[-500:],
            }
    
    return results


# ============================================================
# Part 3: Test metamaterial workflow
# ============================================================
def test_metamaterial_workflow():
    """Test create_metamaterial with different unit cells."""
    results = {}
    unit_cells = [
        ("reentrant_honeycomb_2d", {"reentrant_angle": 150, "cell_height": 10, "cell_width": 10}),
        ("chiral_honeycomb_2d", {}),
        ("star_honeycomb_2d", {}),
        ("arrowhead_auxetic_2d", {}),
        ("hierarchical_lattice_2d", {}),
        ("missing_rib_auxetic_2d", {}),
    ]
    
    for cell_name, params in unit_cells:
        try:
            t0 = time.time()
            meta = fn.create_metamaterial(
                unit_cell=cell_name,
                array_size=(3, 3),
                **params,
            )
            dt = time.time() - t0
            results[cell_name] = {
                'status': 'OK',
                'time': dt,
                'fibers': meta.num_fibers,
                'crosslinks': meta.num_crosslinks,
                'dim': meta.dimension,
            }
        except Exception as e:
            results[cell_name] = {
                'status': 'FAIL',
                'error': str(e)[:200],
                'traceback': traceback.format_exc()[-500:],
            }
    
    return results


# ============================================================
# Part 4: Test parametric study capability
# ============================================================
def test_parametric_study():
    """Test generating structures with varying parameters."""
    results = {}
    
    # Reentrant angle sweep
    angles = [30, 60, 90, 120, 150]
    angle_data = []
    for angle in angles:
        try:
            net = fn.create("reentrant_honeycomb_2d", reentrant_angle=angle)
            angle_data.append({
                'angle': angle,
                'fibers': net.num_fibers,
                'crosslinks': net.num_crosslinks,
            })
        except Exception as e:
            angle_data.append({'angle': angle, 'error': str(e)[:100]})
    results['reentrant_angle_sweep'] = angle_data
    
    return results


# ============================================================
# Run all tests
# ============================================================
if __name__ == '__main__':
    print("=" * 80)
    print("PART 1: Registered Generators (fn.create)")
    print("=" * 80)
    r1 = test_registered_generators()
    
    ok_count = sum(1 for v in r1.values() if v['status'] == 'OK')
    fail_count = sum(1 for v in r1.values() if v['status'] == 'FAIL')
    print(f"\nTotal: {len(r1)} | OK: {ok_count} | FAIL: {fail_count}")
    print()
    
    # Summary table
    print(f"{'Name':<30} {'Status':<6} {'Time':>6} {'Fibers':>8} {'CL':>6} {'Dim':>4} {'Nodes':>6} {'Edges':>6} {'AvgDeg':>7} {'Comp':>5}")
    print("-" * 100)
    for name, r in sorted(r1.items()):
        if r['status'] == 'OK':
            print(f"{name:<30} {'OK':<6} {r['time']:>5.2f}s {r['fibers']:>8} {r['crosslinks']:>6} {r['dim']:>4} {r['nodes']:>6} {r['edges']:>6} {r['avg_deg']:>7.2f} {r['components']:>5}")
        else:
            print(f"{name:<30} {'FAIL':<6} {r['error'][:60]}")
    
    # Print failures detail
    fails = {k: v for k, v in r1.items() if v['status'] == 'FAIL'}
    if fails:
        print("\n--- FAILURES DETAIL ---")
        for name, r in fails.items():
            print(f"\n{name}: {r['error']}")
            if 'traceback' in r:
                print(r['traceback'])
    
    print("\n" + "=" * 80)
    print("PART 2: Unregistered/Extra Generators")
    print("=" * 80)
    r2 = test_unregistered_generators()
    
    ok2 = sum(1 for v in r2.values() if v['status'] == 'OK')
    fail2 = sum(1 for v in r2.values() if v['status'] == 'FAIL')
    print(f"\nTotal: {len(r2)} | OK: {ok2} | FAIL: {fail2}")
    print()
    
    print(f"{'Name':<40} {'Status':<6} {'Time':>6} {'Fibers':>8} {'CL':>6} {'Dim':>4}")
    print("-" * 80)
    for name, r in sorted(r2.items()):
        if r['status'] == 'OK':
            print(f"{name:<40} {'OK':<6} {r['time']:>5.2f}s {r['fibers']:>8} {r['crosslinks']:>6} {r['dim']:>4}")
        else:
            print(f"{name:<40} {'FAIL':<6} {r['error'][:50]}")
    
    fails2 = {k: v for k, v in r2.items() if v['status'] == 'FAIL'}
    if fails2:
        print("\n--- FAILURES DETAIL ---")
        for name, r in fails2.items():
            print(f"\n{name}: {r['error']}")
            if 'traceback' in r:
                print(r['traceback'])
    
    print("\n" + "=" * 80)
    print("PART 3: Metamaterial Workflow")
    print("=" * 80)
    r3 = test_metamaterial_workflow()
    
    print(f"\n{'Cell':<30} {'Status':<6} {'Time':>6} {'Fibers':>8} {'CL':>6}")
    print("-" * 70)
    for name, r in r3.items():
        if r['status'] == 'OK':
            print(f"{name:<30} {'OK':<6} {r['time']:>5.2f}s {r['fibers']:>8} {r['crosslinks']:>6}")
        else:
            print(f"{name:<30} {'FAIL':<6} {r['error'][:50]}")
    
    fails3 = {k: v for k, v in r3.items() if v['status'] == 'FAIL'}
    if fails3:
        print("\n--- FAILURES DETAIL ---")
        for name, r in fails3.items():
            print(f"\n{name}: {r['error']}")
            if 'traceback' in r:
                print(r['traceback'])
    
    print("\n" + "=" * 80)
    print("PART 4: Parametric Study")
    print("=" * 80)
    r4 = test_parametric_study()
    for key, data in r4.items():
        print(f"\n{key}:")
        for d in data:
            print(f"  {d}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total_ok = ok_count + ok2 + sum(1 for v in r3.values() if v['status'] == 'OK')
    total_fail = fail_count + fail2 + sum(1 for v in r3.values() if v['status'] == 'FAIL')
    print(f"Total generators tested: {len(r1) + len(r2) + len(r3)}")
    print(f"  Registered (fn.create): {ok_count}/{len(r1)} OK")
    print(f"  Extra (gen.xxx):        {ok2}/{len(r2)} OK")
    print(f"  Metamaterial workflow:  {sum(1 for v in r3.values() if v['status']=='OK')}/{len(r3)} OK")
    print(f"Total failures: {total_fail}")
