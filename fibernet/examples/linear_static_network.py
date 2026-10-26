"""Fixed 81-node linear static network: generate, solve, save and render.

Run ``python -m fibernet.examples.linear_static_network --output-dir NEWDIR``.
For saved-only rendering add ``--plot-only --figure-dir NEWFIGURES``.
No computation occurs on import. Output directories must not already exist.
"""
import argparse
import csv
import json
import math
from pathlib import Path
__all__ = ['run', 'draw_from_saved', 'main']
CASE = {'geometry': {'generator': 'pattern_2d', 'unit': 'triangle', 'box_cell_m': [0.025, 0.025], 'grid': [8, 8], 'n_internal': 0, 'n_pts_per_side': 0, 'boundary_mode': 'none', 'rotation': 0, 'perturbation': 0, 'radius_m': 0.0005, 'expected_nodes': 81, 'expected_edges': 208, 'total_extent_m': [0.2, 0.2]}, 'material': {'synthetic_not_calibrated': True, 'E_Pa': 1000000000, 'density_kg_m3': 1000, 'poisson_ratio': 0.3, 'section': 'solid circular', 'temperature_K': 300, 'reference_temperature_K': 300, 'alpha_per_K': 0}, 'boundary': {'clamp': 'all 3 planar DOFs at x=0 (9 nodes)', 'loads': 'each x=0.2 node with j=round(y/.025): [0.001*(1+j/8), -0.0005*(1+(j%3)/2), 0] in N,N,Nm', 'load_factors': [1, 2]}, 'model': 'existing planar welded Euler-Bernoulli snapshot with zero thermal strain; no geometric stiffness, shear, buckling or stability certification', 'checks': {'counts': True, 'finite': True, 'force_balance': 'support reactions + applied loads; each component <=1e-10 N +1e-8*sum(abs(applied forces))', 'moment_balance': 'sum(Mz+x*Fy-y*Fx) including support couples <=1e-11 Nm+1e-8*sum(abs(Mz)+abs(x*Fy)+abs(y*Fx)) over applied+support', 'free_residual': '<=1e-10 N for translation,1e-11 Nm for rotation plus1e-8 times respective max applied magnitude', 'F2_scaling': 'u2-2u1 <=1e-12 m/rad+1e-8*maxabs(2u1); reactions similarly force/moment floors 1e-10 N/1e-11 Nm', 'energy': 'independent axial-linear and cubic-Hermite curvature analytical integral per member, no solver K reuse; compare 0.5 sum(F*u) within1e-16 J+1e-8*max(abs(energies))', 'small_motion': 'max nodal translation/0.025 <=0.01; maxabs(theta)<=0.01 rad; maxabs axial strain<=0.005; exact retained values; no claim buckling stability', 'energy_scaling': 'abs(U2-4*U1)<=1e-16 J+1e-8*max(abs(U2),abs(4*U1))'}, 'display': {'magnification': 10000, 'load_arrows': 'schematic at reference-state node positions in both panels'}}

def run(output_dir):
    """Run exactly F/2F; refuse existing output before generating or solving.

    A later failure leaves partial output for inspection. No automatic retry,
    deletion, rollback or replacement of existing output is performed.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    try:
        return _run(out)
    except BaseException as error:
        try:
            with (out / 'failure.json').open('x', encoding='utf-8') as stream:
                json.dump({'error_type': type(error).__name__, 'message': str(error)}, stream, indent=2)
        except BaseException:
            pass
        raise

def _run(O):
    import numpy as np
    from fibernet.gen.pattern import pattern_2d
    from fibernet.core.material import Material
    from fibernet.core.structure_graph import StructureGraph
    from fibernet.sim.thermoelastic import solve_thermoelastic_network_snapshot
    checks = []

    def save(name, obj):
        path = O / name
        path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        if json.loads(path.read_text(encoding='utf-8')) != obj:
            raise AssertionError('JSON field readback mismatch: ' + name)

    def check(name, ok, details=None):
        checks.append({'name': name, 'pass': bool(ok), 'details': details})
        if not ok:
            raise AssertionError((name, details))
    save('case.json', CASE)
    g = pattern_2d(unit='triangle', box=(0.025, 0.025), grid=(8, 8), n_internal=0, n_pts_per_side=0, radius=0.0005, boundary_mode='none', rotation=0, perturbation=0, material=Material(name='synthetic_static_workflow', youngs_modulus=1000000000.0, density=1000.0, poissons_ratio=0.3))
    check('counts', len(g.nodes) == 81 and len(g.edges) == 208)
    g.save_json(O / 'graph.json')
    restored = StructureGraph.load_json(O / 'graph.json')
    check('graph_field_readback', restored.to_dict() == g.to_dict())
    g = restored
    clamps = {n: [0.0, 0.0, 0.0] for n, v in g.nodes.items() if abs(float(v.position[0])) < 1e-14}
    loads = {n: [0.001 * (1 + round(float(v.position[1]) / 0.025) / 8), -0.0005 * (1 + round(float(v.position[1]) / 0.025) % 3 / 2), 0.0] for n, v in g.nodes.items() if abs(float(v.position[0]) - 0.2) < 1e-14}
    check('boundary_counts', len(clamps) == 9 and len(loads) == 9)
    results = []
    reports = []
    for factor in (1, 2):
        r = solve_thermoelastic_network_snapshot(g, length_unit='m', E_Pa=1000000000.0, expansion_alpha_per_K={e: 0.0 for e in g.edges}, stress_free_temperature_K=300.0, temperature_K={n: 300.0 for n in g.nodes}, constraints=clamps, loads={n: [factor * x for x in f] for n, f in loads.items()}).to_dict()
        save(f'snapshot_F{factor}.json', r)
        with (O / f'nodal_F{factor}.csv').open('x', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['node_id', 'x_m', 'y_m', 'ux_m', 'uy_m', 'theta_rad', 'Fx_N', 'Fy_N', 'Mz_Nm', 'reaction_x_N', 'reaction_y_N', 'reaction_moment_Nm'])
            writer.writerows(([node, *position, *displacement, *load, *reaction] for node, position, displacement, load, reaction in zip(r['node_ids'], r['positions_m'], r['displacement'], r['loads'], r['reactions'])))
        results.append(r)
        p = np.array(r['positions_m'])
        u = np.array(r['displacement'])
        f = np.array(r['loads'])
        re = np.array(r['reactions'])
        mask = np.array(r['constraint_mask'])
        support = np.where(mask, re, 0.0)
        check(f'finite_F{factor}', all((np.isfinite(a).all() for a in (p, u, f, re))))
        force_balance = np.sum(support[:, :2] + f[:, :2], axis=0)
        force_scale = float(np.abs(f[:, :2]).sum())
        check(f'force_balance_F{factor}', np.max(abs(force_balance)) <= 1e-10 + 1e-08 * force_scale, force_balance.tolist())
        allf = support + f
        moment = float(np.sum(allf[:, 2] + p[:, 0] * allf[:, 1] - p[:, 1] * allf[:, 0]))
        mscale = float(np.sum(abs(allf[:, 2]) + abs(p[:, 0] * allf[:, 1]) + abs(p[:, 1] * allf[:, 0])))
        check(f'moment_balance_F{factor}', abs(moment) <= 1e-11 + 1e-08 * mscale, moment)
        for k in range(3):
            residual = float(np.max(np.abs(re[:, k][~mask[:, k]]), initial=0))
            bound = (1e-11 if k == 2 else 1e-10) + 1e-08 * float(np.max(abs(f[:, k])))
            check(f'free_residual_F{factor}_dof{k}', residual <= bound, {'value': residual, 'bound': bound})
        check(f'small_motion_F{factor}', float(np.max(np.linalg.norm(u[:, :2], axis=1))) / 0.025 <= 0.01 and float(np.max(abs(u[:, 2]))) <= 0.01 and (max(map(abs, r['total_axial_strain'])) <= 0.005))
        energies = _edge_energies(r)
        U = math.fsum((e['axial_J'] + e['bending_J'] for e in energies))
        work = 0.5 * float(np.sum(f * u))
        eb = 1e-16 + 1e-08 * max(abs(U), abs(work))
        check(f'independent_energy_F{factor}', abs(U - work) <= eb, {'integrated_J': U, 'half_F_dot_u_J': work, 'bound_J': eb})
        save(f'edge_energy_F{factor}.json', energies)
        reports.append({'factor': factor, 'max_translation_m': float(np.max(np.linalg.norm(u[:, :2], axis=1))), 'max_rotation_rad': float(np.max(abs(u[:, 2]))), 'max_axial_strain': max(map(abs, r['total_axial_strain'])), 'energy_J': U, 'half_work_J': work, 'force_balance_N': force_balance.tolist(), 'moment_balance_Nm': moment, 'sum_support': support.sum(axis=0).tolist(), 'sum_load': f.sum(axis=0).tolist(), 'applicability': r['applicability']})
    for field in ('displacement', 'reactions'):
        a = 2 * np.array(results[0][field])
        b = np.array(results[1][field])
        floors = np.array([1e-12] * 3) if field == 'displacement' else np.array([1e-10, 1e-10, 1e-11])
        bound = floors + 1e-08 * np.max(abs(a), axis=0)
        check('linear_scaling_' + field, bool(np.all(np.max(abs(b - a), axis=0) <= bound)), {'error': np.max(abs(b - a), axis=0).tolist(), 'bounds': bound.tolist()})
    check('energy_scaling', abs(reports[1]['energy_J'] - 4 * reports[0]['energy_J']) <= 1e-16 + 1e-08 * max(abs(reports[1]['energy_J']), abs(4 * reports[0]['energy_J'])))
    summary = {'status': 'PASS', 'solve_count': 2, 'graph_generations': 1, 'reports': reports, 'checks': checks, 'persistence': 'Snapshot JSON field readback only; not native load or restart', 'model': 'Synthetic linear welded Euler-Bernoulli; zero thermal strain'}
    draw_from_saved(O, O / 'figures')
    save('summary.json', summary)
    return summary

def _edge_energies(r):
    """Analytical integral of linear axial strain and cubic Hermite curvature.

 No stiffness matrix or solver call is used. With v(x)=v0+theta0*x+a2*x²+a3*x³,
 integral curvature² dx = 4*a2²*L+12*a2*a3*L²+12*a3²*L³.
 """
    import numpy as np
    p = np.array(r['positions_m'])
    u = np.array(r['displacement'])
    idx = {n: i for i, n in enumerate(r['node_ids'])}
    energies = []
    for eid, ends, rad in zip(r['edge_ids'], r['endpoints_by_id'], r['radii_m']):
        a, b = map(idx.__getitem__, ends)
        d = p[b, :2] - p[a, :2]
        L = float(np.linalg.norm(d))
        c, s = d / L
        ua, va = (c * u[a, 0] + s * u[a, 1], -s * u[a, 0] + c * u[a, 1])
        ub, vb = (c * u[b, 0] + s * u[b, 1], -s * u[b, 0] + c * u[b, 1])
        ta, tb = (u[a, 2], u[b, 2])
        a2 = 3 * (vb - va) / L ** 2 - (2 * ta + tb) / L
        a3 = 2 * (va - vb) / L ** 3 + (ta + tb) / L ** 2
        integral = 4 * a2 * a2 * L + 12 * a2 * a3 * L ** 2 + 12 * a3 * a3 * L ** 3
        axial = 0.5 * 1000000000.0 * math.pi * rad ** 2 * (ub - ua) ** 2 / L
        bending = 0.5 * 1000000000.0 * (math.pi * rad ** 4 / 4) * integral
        energies.append({'edge_id': eid, 'axial_J': float(axial), 'bending_J': float(bending)})
    return energies

def draw_from_saved(output_dir, figure_dir):
    """Render saved F1 fields into a NEW figure directory; never generate/solve.

    Missing or inconsistent fields raise rather than triggering recomputation.
    Raw data are never changed. Failed rendering may leave partial new figures.
    """
    O, V = (Path(output_dir), Path(figure_dir))
    if V.exists():
        raise FileExistsError('figure directory already exists: ' + str(V))
    import numpy as np
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    r = json.loads((O / 'snapshot_F1.json').read_text(encoding='utf-8'))
    _validate_saved(r)
    V.mkdir(parents=True, exist_ok=False)
    p = np.array(r['positions_m'])
    u = np.array(r['displacement'])
    ids = {n: i for i, n in enumerate(r['node_ids'])}
    clamp = np.all(np.array(r['constraint_mask']), axis=1)
    force = np.array(r['loads'])
    li = np.flatnonzero(np.any(force != 0, axis=1))
    with matplotlib.rc_context({'svg.fonttype': 'none'}):
        fig = Figure(figsize=(8.2677, 11.6929))
        try:
            FigureCanvasAgg(fig)
            axs = fig.subplots(2, 1)
            fig.subplots_adjust(left=0.13, right=0.8, bottom=0.11, top=0.84, hspace=0.52)
            fig.suptitle('Static response of a welded network\n81 nodes · 208 beams · synthetic SI case', fontsize=16, y=0.96)
            segments = []
            values = []
            curves = []
            for a_id, b_id in r['endpoints_by_id']:
                a, b = (ids[a_id], ids[b_id])
                d = p[b, :2] - p[a, :2]
                L = float(np.linalg.norm(d))
                c, s = d / L
                t = np.linspace(0, 1, 25)
                ua, ub = (c * u[a, 0] + s * u[a, 1], c * u[b, 0] + s * u[b, 1])
                va, vb = (-s * u[a, 0] + c * u[a, 1], -s * u[b, 0] + c * u[b, 1])
                axial = (1 - t) * ua + t * ub
                trans = (1 - 3 * t * t + 2 * t ** 3) * va + L * (t - 2 * t * t + t ** 3) * u[a, 2] + (3 * t * t - 2 * t ** 3) * vb + L * (-t * t + t ** 3) * u[b, 2]
                delta = np.column_stack((c * axial - s * trans, s * axial + c * trans))
                base = p[a, :2] + t[:, None] * d
                q = base + delta
                segments.extend(np.stack([q[:-1], q[1:]], axis=1))
                values.extend(1000000.0 * np.linalg.norm((delta[:-1] + delta[1:]) / 2, axis=1))
                curves.append((base, delta))
            lc = LineCollection(segments, cmap='viridis', norm=matplotlib.colors.Normalize(0, max(values)), linewidths=1.25)
            lc.set_array(np.array(values))
            axs[0].add_collection(lc)
            cax = fig.add_axes([0.84, 0.6, 0.025, 0.24])
            cb = fig.colorbar(lc, cax=cax)
            cb.set_label('Displacement magnitude [µm]', labelpad=12)
            axs[0].set_title('Actual geometry · displacement scale ×1', fontsize=12, pad=32)
            for base, delta in curves:
                axs[1].plot(base[:, 0], base[:, 1], color='#b9c0c6', lw=0.65)
                axs[1].plot((base + 10000 * delta)[:, 0], (base + 10000 * delta)[:, 1], color='#176a90', lw=0.95)
            axs[1].set_title('Deformed shape ×10,000 — display magnification only', fontsize=11, pad=32)
            for ax in axs:
                ax.scatter(p[clamp, 0], p[clamp, 1], marker='s', s=14, color='#242424', zorder=4)
                ax.quiver(p[li, 0], p[li, 1], force[li, 0], force[li, 1], angles='xy', scale_units='xy', scale=0.075, color='#b64b28', width=0.004)
                ax.set_xlim(-0.008, 0.23)
                ax.set_ylim(-0.025, 0.21)
                ax.set_aspect('equal')
                ax.set_xlabel('x [m]')
                ax.set_ylabel('y [m]')
                ax.grid(alpha=0.13)
            axs[0].legend(handles=[Line2D([], [], color='#242424', marker='s', ls='', label='Clamped nodes'), Line2D([], [], color='#b64b28', label='Applied forces (schematic)')], loc='lower left', bbox_to_anchor=(0, 1.005), ncol=2, frameon=False, fontsize=8)
            axs[1].legend(handles=[Line2D([], [], color='#b9c0c6', label='Undeformed'), Line2D([], [], color='#176a90', label='Deformed ×10,000')], loc='lower left', bbox_to_anchor=(0, 1.005), ncol=2, frameon=False, fontsize=8)
            fig.text(0.13, 0.05, f'F case: maximum nodal displacement {1000000.0 * np.max(np.linalg.norm(u[:, :2], axis=1)):.3f} µm\nE = 1 GPa · circular radius = 0.5 mm · cubic Hermite beam centerlines', fontsize=10)
            fig.savefig(V / 'static_workflow_A4.svg')
            fig.savefig(V / 'static_workflow_A4.png', dpi=160)
        except BaseException:
            try:
                fig.clear()
            except BaseException:
                pass
            raise
        else:
            fig.clear()
    return {'png': str(V / 'static_workflow_A4.png'), 'svg': str(V / 'static_workflow_A4.svg')}

def _validate_saved(r):
    """Check this example's saved field layout, not a general restart contract."""
    import numpy as np
    ids = r['node_ids']
    eids = r['edge_ids']
    if len(ids) != 81 or len(set(ids)) != 81 or len(eids) != 208 or (len(set(eids)) != 208):
        raise ValueError('expected this example: 81 distinct nodes and 208 edges')
    for name, shape in [('positions_m', (81, 2)), ('displacement', (81, 3)), ('loads', (81, 3)), ('constraint_mask', (81, 3))]:
        a = np.asarray(r[name])
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError('invalid saved field ' + name)
    ends = r['endpoints_by_id']
    if len(ends) != 208 or any((len(e) != 2 or any((n not in ids for n in e)) for e in ends)):
        raise ValueError('invalid saved member incidence')
    if any((np.linalg.norm(np.asarray(r['positions_m'])[ids.index(e[1])] - np.asarray(r['positions_m'])[ids.index(e[0])]) <= 0 for e in ends)):
        raise ValueError('zero-length saved member')
    if r['E_Pa'] != 1000000000.0 or len(r['radii_m']) != 208 or any((radius != 0.0005 for radius in r['radii_m'])):
        raise ValueError('expected the fixed example modulus and radii')
    if r['profile'] != 'thermoelastic_2d_network/1' or r['numerical_regularization'] is not False:
        raise ValueError('expected the unregularized planar welded network profile')
    if r['stress_free_temperature_K'] != 300.0 or r['temperature_K'] != [300.0] * 81 or r['expansion_alpha_per_K'] != [0.0] * 208 or (r['thermal_axial_strain'] != [0.0] * 208):
        raise ValueError('expected the fixed zero-thermal-strain case')
    if r['units'] != {'length': 'm', 'rotation': 'rad', 'force': 'N', 'moment': 'N*m', 'stress': 'Pa', 'temperature': 'K', 'alpha': '1/K'}:
        raise ValueError('expected explicit SI snapshot units')

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--figure-dir', help='new figure directory, required with --plot-only')
    args = parser.parse_args(argv)
    if args.plot_only and (not args.figure_dir):
        parser.error('--plot-only requires --figure-dir NEWDIR')
    if not args.plot_only and args.figure_dir:
        parser.error('--figure-dir is only used with --plot-only')
    result = draw_from_saved(args.output_dir, args.figure_dir) if args.plot_only else run(args.output_dir)
    print(json.dumps(result, indent=2, allow_nan=False))
    return result
if __name__ == '__main__':
    main()
