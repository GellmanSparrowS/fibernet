"""Phase 6b: Visualization from saved results. Edges colored by stress."""
import sys, json
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

RESULTS_DIR = Path(__file__).parent / "results"
VIZ_FILE = RESULTS_DIR / "phase6b_visualization.png"

def draw_edges(ax, pos, ei, el, es, cmap_name='hot', lw=0.8, alpha=0.9, colorbar=True, label='σ (Pa)'):
    """Draw edges colored by stress value."""
    norm = mcolors.Normalize(vmin=es.min(), vmax=es.max())
    cmap = plt.get_cmap(cmap_name)
    
    for idx, e in enumerate(el):
        n1, n2 = int(ei[0, e]), int(ei[1, e])
        c = cmap(norm(es[idx]))
        ax.plot([pos[n1,0], pos[n2,0]], [pos[n1,1], pos[n2,1]],
               color=c, linewidth=lw, alpha=alpha)
    
    if colorbar:
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
                         ax=ax, label=label, shrink=0.7)
    return norm

def draw_edges_3d(ax, pos, ei, el, es, cmap_name='hot', lw=0.8, alpha=0.7, colorbar=True):
    """Draw 3D edges colored by stress."""
    norm = mcolors.Normalize(vmin=es.min(), vmax=es.max())
    cmap = plt.get_cmap(cmap_name)
    
    for idx, e in enumerate(el):
        n1, n2 = int(ei[0, e]), int(ei[1, e])
        c = cmap(norm(es[idx]))
        ax.plot([pos[n1,0], pos[n2,0]], [pos[n1,1], pos[n2,1]],
               [pos[n1,2], pos[n2,2]], color=c, linewidth=lw, alpha=alpha)
    
    if colorbar:
        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                    ax=ax, label='σ (Pa)', shrink=0.6)

def main():
    print("Loading results...")
    with open(RESULTS_DIR / "phase6b_results.json") as f:
        data = json.load(f)
    
    baseline = data.get('deformed_baseline', {})
    large_def = data.get('large_deformation', {})
    multi_r = data.get('multi_radius', {})
    results_3d = data.get('3d', {})
    
    units_2d = ['honeycomb', 'kagome', 'reentrant', 'triangle']
    
    fig = plt.figure(figsize=(32, 42))
    gs = GridSpec(7, 4, figure=fig, hspace=0.4, wspace=0.35)
    
    # ---- Row 1: Baseline 10% stretch ----
    print("Row 1: Baseline deformed structures...")
    for i, unit in enumerate(units_2d):
        ax = fig.add_subplot(gs[0, i])
        if unit in baseline:
            r = baseline[unit]
            pos = np.array(r['node_pos'])
            disp = np.array(r['displacement'])
            ei = np.array(r['edge_index'])
            el = np.array(r['edge_list'])
            es = np.array(r['edge_stress'])
            
            pos_def = pos + disp * 0.3
            draw_edges(ax, pos_def, ei, el, es, lw=0.6, label='σ_total (Pa)')
            
            ax.set_title(f"{unit.capitalize()}\nmax_σ={es.max():.2e} Pa | SCF={r['scf']:.1f}\nprop={r['propagation_20pct_to_80pct']:.0%}",
                        fontsize=9)
            ax.set_aspect('equal')
            ax.tick_params(labelsize=5)
    
    fig.text(0.5, 0.935, 'Row 1: Deformed Structures (10% stretch, 10% fixed each side)',
             ha='center', fontsize=14, fontweight='bold')
    
    # ---- Row 2: Stretch 100% ----
    print("Row 2: Large deformation stretch 100%...")
    for i, unit in enumerate(units_2d):
        ax = fig.add_subplot(gs[1, i])
        t_key = 'stretch_100pct'
        if unit in large_def and t_key in large_def[unit].get('tests', {}):
            t = large_def[unit]['tests'][t_key]
            if 'error' not in t:
                pos = np.array(t['node_pos'])
                disp = np.array(t['displacement'])
                ei = np.array(t['edge_index'])
                el = np.array(t['edge_list'])
                es = np.array(t['edge_stress'])
                
                pos_def = pos + disp * 0.08  # small scale for 100%
                draw_edges(ax, pos_def, ei, el, es, lw=0.4, alpha=0.7)
                
                ax.set_title(f"{unit.capitalize()} Stretch 100%\nmax_σ={es.max():.2e}",
                            fontsize=9)
                ax.set_aspect('equal')
                ax.tick_params(labelsize=5)
    
    fig.text(0.5, 0.82, 'Row 2: Large Deformation — Stretch 100% (10cm on 10cm)',
             ha='center', fontsize=14, fontweight='bold')
    
    # ---- Row 3: Compress 50% ----
    print("Row 3: Large deformation compress 50%...")
    for i, unit in enumerate(units_2d):
        ax = fig.add_subplot(gs[2, i])
        t_key = 'compress_50pct'
        if unit in large_def and t_key in large_def[unit].get('tests', {}):
            t = large_def[unit]['tests'][t_key]
            if 'error' not in t:
                pos = np.array(t['node_pos'])
                disp = np.array(t['displacement'])
                ei = np.array(t['edge_index'])
                el = np.array(t['edge_list'])
                es = np.array(t['edge_stress'])
                
                pos_def = pos + disp * 0.15
                draw_edges(ax, pos_def, ei, el, es, lw=0.4, alpha=0.7)
                
                ax.set_title(f"{unit.capitalize()} Compress 50%\nmax_σ={es.max():.2e}",
                            fontsize=9)
                ax.set_aspect('equal')
                ax.tick_params(labelsize=5)
    
    fig.text(0.5, 0.71, 'Row 3: Large Deformation — Compress 50% (5cm on 10cm)',
             ha='center', fontsize=14, fontweight='bold')
    
    # ---- Row 4: Propagation curves ----
    print("Row 4: Propagation curves...")
    # Baseline propagation
    ax1 = fig.add_subplot(gs[3, :2])
    for unit in units_2d:
        if unit in baseline:
            bins = baseline[unit]['bin_means']
            x_bins = np.linspace(0, 1, len(bins))
            ax1.plot(x_bins, bins, '-o', label=unit, linewidth=2, markersize=5)
    ax1.set_xlabel('Normalized x-distance (fixed→loaded)', fontsize=10)
    ax1.set_ylabel('Mean |u|', fontsize=10)
    ax1.set_title('Propagation: 10% stretch on deformed structures', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Large deformation propagation
    ax2 = fig.add_subplot(gs[3, 2:])
    for unit in units_2d:
        if unit in large_def:
            t = large_def[unit].get('tests', {}).get('stretch_100pct', {})
            if 'bin_means' in t:
                bins = t['bin_means']
                x_bins = np.linspace(0, 1, len(bins))
                ax2.plot(x_bins, bins, '-s', label=unit, linewidth=2, markersize=5)
    ax2.set_xlabel('Normalized x-distance (fixed→loaded)', fontsize=10)
    ax2.set_ylabel('Mean |u|', fontsize=10)
    ax2.set_title('Propagation: 100% stretch', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    fig.text(0.5, 0.59, 'Row 4: Deformation Propagation Analysis',
             ha='center', fontsize=14, fontweight='bold')
    
    # ---- Row 5: Multi-radius ----
    print("Row 5: Multi-radius analysis...")
    for i, unit in enumerate(units_2d):
        ax = fig.add_subplot(gs[4, i])
        if unit in multi_r:
            r_vals, sa_vals, sb_vals, st_vals = [], [], [], []
            for k, v in multi_r[unit].items():
                r_val = float(k.replace('r', ''))
                r_vals.append(r_val)
                sa_vals.append(v['sigma_axial'])
                sb_vals.append(v['sigma_bending'])
                st_vals.append(v['sigma_total'])
            
            ax.plot(r_vals, sa_vals, 'b-o', label='σ_axial', linewidth=1.5, markersize=4)
            ax.plot(r_vals, sb_vals, 'r-s', label='σ_bending', linewidth=1.5, markersize=4)
            ax.plot(r_vals, st_vals, 'k-^', label='σ_total', linewidth=1.5, markersize=4)
            ax.set_xlabel('Fiber radius (m)', fontsize=9)
            ax.set_ylabel('Stress (Pa)', fontsize=9)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title(f"{unit.capitalize()}", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
    
    fig.text(0.5, 0.48, 'Row 5: Multi-Radius Scaling (10% stretch on deformed structures)',
             ha='center', fontsize=14, fontweight='bold')
    
    # ---- Row 6: 3D structures ----
    print("Row 6: 3D structures...")
    names_3d = list(results_3d.keys())[:3]
    for i, name in enumerate(names_3d):
        d = results_3d[name]
        if 'error' not in d:
            ax = fig.add_subplot(gs[5, i], projection='3d')
            
            pos = np.array(d['node_pos'])
            disp = np.array(d['displacement'])
            ei = np.array(d['edge_index'])
            el = np.array(d['edge_list'])
            es = np.array(d['edge_stress'])
            
            pos_def = pos + disp * 0.3
            draw_edges_3d(ax, pos_def, ei, el, es, lw=0.6)
            
            u_max = np.max(np.linalg.norm(disp, axis=1))
            ax.set_title(f"{name} ({d['n_nodes']}n/{d['n_edges']}e)\nmax_σ={es.max():.2e} | max_u={u_max:.2e}",
                        fontsize=9)
            ax.set_xlabel('X', fontsize=7)
            ax.set_ylabel('Y', fontsize=7)
            ax.set_zlabel('Z', fontsize=7)
    
    # 3D propagation
    ax = fig.add_subplot(gs[5, 3])
    for name in names_3d:
        d = results_3d[name]
        if 'error' not in d and 'bin_means_z' in d:
            bins = d['bin_means_z']
            z_bins = np.linspace(0, 1, len(bins))
            ax.plot(z_bins, bins, '-o', label=name, linewidth=2, markersize=5)
    ax.set_xlabel('Normalized z (bottom→top)', fontsize=9)
    ax.set_ylabel('Mean |u|', fontsize=9)
    ax.set_title('3D z-Propagation', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    fig.text(0.5, 0.36, 'Row 6: 3D Cube Lattices (20% compression, 10% fixed each side)',
             ha='center', fontsize=14, fontweight='bold')
    
    # ---- Row 7: Data analysis summary ----
    print("Row 7: Summary...")
    ax = fig.add_subplot(gs[6, :])
    
    lines = [
        "COMPREHENSIVE FEM VALIDATION — Phase 6b (Deformed Structures, Proper BCs)",
        "=" * 100,
        "Solver: BeamFrameFEM_v6 | E=1GPa | nu=0.3 | Deformed: n_pts_per_side=5, disp=+-0.4",
        "BCs: 10% nodes fixed on each boundary side | Fibers: welded joints (moment-resisting)",
        "",
        f"{'Structure':>12s} | {'Nodes':>5s} {'Edges':>5s} | {'max_u':>10s} {'max_s':>10s} {'SCF':>5s} {'Prop':>5s} | {'σ_ax':>10s} {'σ_bend':>10s}",
    ]
    for unit in units_2d:
        if unit in baseline:
            r = baseline[unit]
            lines.append(
                f"{unit:>12s} | {r['n_nodes']:5d} {r['n_edges']:5d} | "
                f"{r['max_disp']:10.2e} {r['max_total_stress']:10.2e} {r['scf']:5.1f} "
                f"{r['propagation_20pct_to_80pct']:5.0%} | "
                f"{r['max_axial_stress']:10.2e} {r['max_bending_stress']:10.2e}"
            )
    
    lines.append("")
    lines.append("LARGE DEFORMATION:")
    for unit in units_2d:
        if unit in large_def:
            s = large_def[unit].get('tests', {}).get('stretch_100pct', {})
            c = large_def[unit].get('tests', {}).get('compress_50pct', {})
            sm = f"{s['max_stress']:.2e}" if 'max_stress' in s else 'ERR'
            cm = f"{c['max_stress']:.2e}" if 'max_stress' in c else 'ERR'
            lines.append(f"  {unit:>12s}: stretch_100%: max_s={sm:>12s} | compress_50%: max_s={cm:>12s}")
    
    lines.append("")
    lines.append("PHYSICS ANALYSIS:")
    lines.append("  1. Deformation propagation: honeycomb 19%, kagome 3%, reentrant 17%, triangle 6%")
    lines.append("     - Honeycomb/reentrant: bending-dominated, deformation concentrates near loaded edge")
    lines.append("     - Kagome/triangle: stretch-dominated, linear propagation (uniform strain field)")
    lines.append("  2. Bending stress dominance:")
    lines.append("     - Honeycomb: sigma_bending/sigma_axial = 7737 (pure bending)")
    lines.append("     - Kagome: sigma_bending/sigma_axial = 2.5 (mixed)")
    lines.append("     - Reentrant: sigma_bending/sigma_axial = 15777 (extreme bending)")
    lines.append("     - Triangle: sigma_bending/sigma_axial = 1.9 (nearly equal)")
    lines.append("  3. SCF (Stress Concentration Factor):")
    lines.append("     - Triangle highest (7.04), then honeycomb (6.60), kagome (6.06), reentrant (5.83)")
    lines.append("  4. Radius scaling: sigma_bending grows with r for all structures")
    lines.append("     - Honeycomb/reentrant: sigma_bending >> sigma_axial for all r (bending always dominates)")
    lines.append("     - Kagome: sigma_axial ~ constant (stretch-dominated baseline), sigma_bending grows")
    lines.append("  5. 3D: compression propagates linearly through cube lattices")
    
    ax.text(0.5, 0.5, '\n'.join(lines), ha='center', va='center',
           fontsize=7.5, family='monospace', transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax.axis('off')
    
    plt.suptitle("FiberNet Beam FEM v6b: Deformed Structure Validation (Edges View)",
                 fontsize=20, fontweight='bold', y=0.985)
    
    print(f"Saving {VIZ_FILE}...")
    plt.savefig(VIZ_FILE, dpi=120, bbox_inches='tight')
    print(f"  Saved: {VIZ_FILE} ({VIZ_FILE.stat().st_size / 1024:.0f} KB)")
    plt.close()

if __name__ == '__main__':
    main()
