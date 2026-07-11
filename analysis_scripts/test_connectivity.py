#!/usr/bin/env python3
"""
Connectivity test for all pattern_2d units and transforms.
Run: python3 analysis_scripts/test_connectivity.py
"""
import sys, os
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..')))

import fibernet as fn
import numpy as np

def find_components(g):
    """BFS to find connected components."""
    adj = {}
    for nid in g.nodes:
        adj[nid] = set()
    for e in g.edges.values():
        adj[e.node_i].add(e.node_j)
        adj[e.node_j].add(e.node_i)
    
    visited = set()
    components = []
    for nid in g.nodes:
        if nid in visited:
            continue
        comp = set()
        queue = [nid]
        visited.add(nid)
        while queue:
            n = queue.pop(0)
            comp.add(n)
            for nb in adj.get(n, set()):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        components.append(comp)
    return components

def test_all():
    passed = 0
    failed = 0
    warnings = 0
    
    print("=" * 60)
    print("Connectivity Test Suite")
    print("=" * 60)
    
    # Test 1: All built-in units (3x3 grid)
    print("\n--- Built-in Units (3x3 grid) ---")
    for unit in fn.list_units():
        try:
            g = fn.pattern_2d(unit=unit, box=(10, 10), grid=(3, 3))
            comps = find_components(g)
            connected = len(comps) == 1
            sym = '✅' if connected else '⚠️'
            if connected:
                passed += 1
            else:
                warnings += 1
            sizes = sorted([len(c) for c in comps], reverse=True)
            print(f"  {sym} {unit:15s}: {g} | components={len(comps)} sizes={sizes[:5]}")
        except Exception as e:
            print(f"  ❌ {unit:15s}: {e}")
            failed += 1
    
    # Test 2: Rotated units
    print("\n--- Rotated Units ---")
    for unit in ['honeycomb', 'hexagon', 'square', 'kagome', 'reentrant']:
        for rot in [15, 30, 45, 60]:
            try:
                g = fn.pattern_2d(unit=unit, box=(10, 10), grid=(2, 2), rotation=rot)
                comps = find_components(g)
                connected = len(comps) == 1
                sym = '✅' if connected else '⚠️'
                if connected:
                    passed += 1
                else:
                    warnings += 1
                print(f"  {sym} {unit:15s} rot={rot:2d}: {g} | components={len(comps)}")
            except Exception as e:
                print(f"  ❌ {unit:15s} rot={rot:2d}: {e}")
                failed += 1
    
    # Test 3: Mirror + rotation combos
    print("\n--- Mirror + Rotation Combos ---")
    for unit in ['honeycomb', 'hexagon', 'square']:
        for mx, my, rot in [(True, False, 30), (False, True, 45), (True, True, 60)]:
            try:
                g = fn.pattern_2d(unit=unit, box=(10, 10), grid=(2, 2),
                                 mirror_x=mx, mirror_y=my, rotation=rot)
                comps = find_components(g)
                connected = len(comps) == 1
                sym = '✅' if connected else '⚠️'
                if connected:
                    passed += 1
                else:
                    warnings += 1
                print(f"  {sym} {unit:15s} mx={mx},my={my},rot={rot}: {g} | components={len(comps)}")
            except Exception as e:
                print(f"  ❌ {unit:15s} mx={mx},my={my},rot={rot}: {e}")
                failed += 1
    
    # Test 4: n_pts_per_side on custom points
    print("\n--- Custom Points + n_pts_per_side ---")
    for n_pts in [0, 1, 2, 3, 5]:
        g = fn.pattern_2d(points=[(0, 5), (10, 5)], closed=False, 
                         box=(10, 10), grid=(1, 1), n_pts_per_side=n_pts)
        expected_nodes = 2 + n_pts  # 2 endpoints + n intermediate
        actual_nodes = len(g.nodes)
        ok = actual_nodes == expected_nodes
        sym = '✅' if ok else '❌'
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  {sym} n_pts={n_pts}: expected {expected_nodes} nodes, got {actual_nodes}")
    
    # Test 5: 3D units
    print("\n--- 3D Units ---")
    for unit in ['cubic', 'octet', 'diamond_3d']:
        try:
            g = fn.pattern_3d(unit=unit, box=(5, 5, 5), grid=(2, 2, 2))
            comps = find_components(g)
            connected = len(comps) == 1
            sym = '✅' if connected else '⚠️'
            if connected:
                passed += 1
            else:
                warnings += 1
            print(f"  {sym} {unit:15s}: {g} | components={len(comps)}")
        except Exception as e:
            print(f"  ❌ {unit:15s}: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {warnings} warnings, {failed} failed")
    print("=" * 60)
    return failed == 0

if __name__ == '__main__':
    success = test_all()
    sys.exit(0 if success else 1)
