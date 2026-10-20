"""History extension of BeamFrameFEM's existing 2-D sparse assembly.

No alternative graph, element solver, inertia or hidden regularization.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import warnings
import numpy as np
from scipy.sparse.linalg import spsolve, MatrixRankWarning
from fibernet.core.viscoelastic import LinearViscoelasticMaterial

_STATE_SCHEMA = 'fibernet.viscoelastic_state/1'
_RESULT_SCHEMA = 'fibernet.viscoelastic_result/1'
_UNITS = {'length': 'm', 'force': 'N', 'moment': 'N*m', 'time': 's',
          'rotation': 'rad', 'energy': 'J', 'axial_stress': 'Pa',
          'curvature': '1/m', 'curvature_stress': 'Pa/m'}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode('utf-8')).hexdigest()


def _finite(value, shape, name):
    obj = np.asarray(value, dtype=object)
    if any(isinstance(item, (bool, np.bool_)) or
           not isinstance(item, (int, float, np.integer, np.floating)) for item in obj.ravel()):
        raise ValueError(f'{name} requires real numbers, not bools or strings')
    try:
        x = np.array(value, dtype=float, copy=True)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{name} must be numeric') from exc
    if x.size == 0 and math.prod(shape) == 0:
        x = x.reshape(shape)  # JSON [] loses trailing dimensions of empty branches.
    if x.shape != shape or not np.isfinite(x).all():
        raise ValueError(f'{name} requires finite shape {shape}')
    return x


class ViscoelasticResult:
    """SI history result with detached data and a restartable final_state.

    displacements/reactions have node order node_ids and components
    ux/uy/rz and Fx/Fy/Mz. section_strain has edge order edge_ids and
    epsilon/kappa_at_Gauss1/kappa_at_Gauss2. Bending moments are at those
    Gauss points, not endpoint maxima; M=-I*sigma_kappa follows the existing
    BeamFrameFEM sign convention. A load result is not a material fit.
    """
    _arrays = ('times', 'displacements', 'reactions', 'section_strain',
               'axial_stress', 'axial_force', 'bending_moment', 'free_residual')

    def __init__(self, data):
        if not isinstance(data, dict) or data.get('schema') != _RESULT_SCHEMA:
            raise ValueError('unsupported viscoelastic result schema')
        required = set(self._arrays) | {'schema', 'node_ids', 'edge_ids', 'energies', 'metadata', 'final_state'}
        if set(data) != required:
            raise ValueError('invalid viscoelastic result fields')
        # JSON validation rejects NaN and arbitrary objects; detach all values.
        self._data = json.loads(_json(data))
        t = np.asarray(data['times'], dtype=float)
        n, e = len(data['node_ids']), len(data['edge_ids'])
        if t.ndim != 1 or len(t) < 1 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError('result times must be finite and increasing')
        shapes = {'times': (len(t),), 'displacements': (len(t), n, 3),
                  'reactions': (len(t), n, 3), 'section_strain': (len(t), e, 3),
                  'axial_stress': (len(t), e), 'axial_force': (len(t), e),
                  'bending_moment': (len(t), e, 2), 'free_residual': (len(t),)}
        for key, shape in shapes.items():
            _finite(data[key], shape, key)
        if set(data['energies']) != {'stored', 'physical_dissipation', 'numerical_dissipation', 'work', 'balance_residual'}:
            raise ValueError('invalid energy fields')
        for key, values in data['energies'].items():
            _finite(values, (len(t),), key)
        state = data['final_state']
        if not isinstance(state, dict) or state.get('schema') != _STATE_SCHEMA:
            raise ValueError('invalid final state')
        check = dict(state)
        digest = check.pop('checksum', None)
        if digest != _hash(check) or state['time'] != float(t[-1]):
            raise ValueError('final state checksum/time mismatch')
        if (state['node_ids'] != data['node_ids'] or state['edge_ids'] != data['edge_ids'] or
                state['units'] != _UNITS or data['metadata'].get('units') != _UNITS or
                data['metadata'].get('fingerprint') != state['fingerprint']):
            raise ValueError('result/state identity mismatch')
        if not np.array_equal(_finite(state['displacement'],(n*3,),'final displacement').reshape(n,3),data['displacements'][-1]):
            raise ValueError('result/final displacement mismatch')
        if not np.array_equal(_finite(state['section_strain'],(e,3),'final strain'),data['section_strain'][-1]):
            raise ValueError('result/final strain mismatch')
        if state['stored_energy'] != data['energies']['stored'][-1] or any(
                state['cumulative'][key] != data['energies'][key][-1] for key in ('physical_dissipation','numerical_dissipation','work')):
            raise ValueError('result/final ledger mismatch')

    def __getattr__(self, name):
        if name in self._arrays:
            return np.array(self._data[name], dtype=float, copy=True)
        if name in ('node_ids', 'edge_ids', 'energies', 'metadata', 'final_state'):
            return copy.deepcopy(self._data[name])
        raise AttributeError(name)

    def to_dict(self):
        return copy.deepcopy(self._data)

    def save(self, path):
        path = Path(path)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                    dir=path.parent, prefix='.'+path.name+'.', delete=False) as stream:
                temporary = stream.name
                stream.write(_json(self._data)+'\n'); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))


def _geometry(graph, length_unit):
    scales = {'m': 1.0, 'mm': 1e-3, 'um': 1e-6}
    if length_unit not in scales:
        raise ValueError('length_unit must explicitly be m, mm or um')
    if graph.dimension != 2:
        raise ValueError('viscoelastic frame history supports dimension=2 only')
    node_ids, edge_ids = sorted(graph.nodes), sorted(graph.edges)
    if len(node_ids) < 2 or not edge_ids:
        raise ValueError('a frame requires at least two nodes and an edge')
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) or i < 0
           for i in node_ids + edge_ids):
        raise ValueError('stable IDs must be nonnegative integers')
    node_ids, edge_ids = list(map(int, node_ids)), list(map(int, edge_ids))
    rows = {nid: i for i, nid in enumerate(node_ids)}
    xyz = _finite([graph.nodes[n].position for n in node_ids], (len(node_ids), 3), 'positions')
    if np.any(xyz[:, 2] != xyz[0, 2]):
        raise ValueError('frame geometry must lie in one XY plane')
    pos = xyz[:, :2] * scales[length_unit]
    edges, radii = [], []
    for eid in edge_ids:
        edge = graph.edges[eid]
        if edge.node_i not in rows or edge.node_j not in rows or edge.node_i == edge.node_j:
            raise ValueError('invalid edge endpoints')
        a, b = rows[edge.node_i], rows[edge.node_j]
        rad = float(edge.radius) * scales[length_unit]
        if isinstance(edge.radius, (bool, np.bool_)) or not math.isfinite(rad) or rad <= 0:
            raise ValueError('radius must be positive and finite')
        d = xyz[b] - xyz[a]
        length = float(np.linalg.norm(pos[b] - pos[a]))
        if not math.isfinite(length) or length < 1e-12:
            raise ValueError('SI member length must be at least 1e-12 m')
        if edge.internal_points is not None and np.size(edge.internal_points):
            points = np.array(edge.internal_points, dtype=float, copy=True)
            if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
                raise ValueError('invalid internal points')
            xi = (points - xyz[a]) @ d / np.dot(d, d)
            deviation = points - xyz[a] - xi[:, None] * d
            if (np.any(xi < 0) or np.any(xi > 1) or np.any(np.diff(xi) <= 0)
                    or np.any(np.linalg.norm(deviation, axis=1) > 1e-10 * np.linalg.norm(d))):
                raise ValueError('curved members require explicit straight-edge subdivision')
        edges.append((a, b)); radii.append(rad)
    edges, radii = np.array(edges, dtype=np.int64), np.array(radii)
    dofs = np.array([[3*a, 3*a+1, 3*a+2, 3*b, 3*b+1, 3*b+2] for a, b in edges])
    B, metric = [], []
    for (a, b), r in zip(edges, radii):
        dx, dy = pos[b] - pos[a]
        L = math.hypot(dx, dy); c, s = dx/L, dy/L
        T = np.array([[c,s,0,0,0,0],[-s,c,0,0,0,0],[0,0,1,0,0,0],
                      [0,0,0,c,s,0],[0,0,0,-s,c,0],[0,0,0,0,0,1]])
        local = [[-1/L, 0, 0, 1/L, 0, 0]]
        for xi in ((1-1/math.sqrt(3))/2, (1+1/math.sqrt(3))/2):
            local.append([0, (-6+12*xi)/L**2, (-4+6*xi)/L,
                          0, (6-12*xi)/L**2, (-2+6*xi)/L])
        B.append(np.array(local) @ T)
        A, I = math.pi*r*r, math.pi*r**4/4
        metric.append((A*L, I*L/2, I*L/2))
    B, metric = np.array(B), np.array(metric)
    if not np.isfinite(B).all() or not np.isfinite(metric).all() or np.any(metric <= 0):
        raise ValueError('section geometry underflow/overflow')
    return node_ids, edge_ids, pos, edges, radii, dofs, B, metric


def _histories(times, node_ids, loads, constraints):
    n = len(node_ids); nt = len(times); rows = {nid: i for i, nid in enumerate(node_ids)}
    force = np.zeros((nt, n, 3)); prescribed = np.zeros_like(force)
    mask = np.zeros((n, 3), dtype=bool)
    for mapping, kind in ((loads, 'loads'), (constraints, 'constraints')):
        if mapping is None:
            continue
        if not isinstance(mapping, dict):
            raise ValueError(f'{kind} must map stable node IDs to (times,3) samples')
        for nid, history in mapping.items():
            if isinstance(nid, (bool, np.bool_)) or not isinstance(nid, (int, np.integer)) or nid not in rows:
                raise ValueError(f'{kind} references an invalid stable node ID')
            i = rows[nid]
            if kind == 'loads':
                force[:, i] = _finite(history, (nt, 3), 'load history')
            else:
                h = np.asarray(history, dtype=object)
                if h.shape != (nt, 3):
                    raise ValueError('constraint history requires (times,3) samples')
                active = np.array([[x is not None for x in row] for row in h])
                if np.any(active != active[0]):
                    raise ValueError('constraint masks must be constant in time')
                mask[i] = active[0]
                for component in range(3):
                    if active[0, component]:
                        prescribed[:, i, component] = _finite(h[:, component], (nt,), 'prescribed history')
    return force.reshape(nt, -1), prescribed.reshape(nt, -1), mask.ravel()


def _check_supports(pos, edges, mask):
    """Positive welded beam elements have only component rigid null modes."""
    parent = list(range(len(pos))); degree = np.zeros(len(pos), dtype=int)
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for a, b in edges:
        parent[root(a)] = root(b); degree[a] += 1; degree[b] += 1
    if np.any(degree == 0):
        raise ValueError('isolated nodes are unsupported in a beam history')
    components = {}
    for i in range(len(pos)):
        components.setdefault(root(i), []).append(i)
    for nodes in components.values():
        xy = pos[nodes] - pos[nodes].mean(axis=0)
        scale = max(float(np.max(np.abs(xy))), 1e-12)
        restrictions = []
        for i, (x, y) in zip(nodes, xy/scale):
            modes = ((1,0,-y),(0,1,x),(0,0,1))
            restrictions.extend(modes[k] for k in range(3) if mask[3*i+k])
        if len(restrictions) < 3 or np.linalg.matrix_rank(np.array(restrictions), tol=1e-10) < 3:
            raise ValueError('each connected component requires three independent rigid-mode restraints')


def _assemble(dofs, B, metric, stress, ndof):
    local = np.einsum('eji,ej->ei', B, metric * stress)
    force = np.zeros(ndof)
    np.add.at(force, dofs.ravel(), local.ravel())
    return force


def _section(B, dofs, u):
    return np.einsum('eij,ej->ei', B, u[dofs])


def _residual(reaction, internal, applied, free, length_scale):
    """Normalize moment rows by specimen member length to compare forces."""
    factors = np.tile([1.,1.,length_scale], len(reaction)//3)
    scale = max(float(np.max(np.abs(internal/factors))),
                float(np.max(np.abs(applied/factors))), 1e-30)
    return float(np.max(np.abs((reaction/factors)[free]))/scale) if free.size else 0.


def _stored(law, s, v, metric):
    density = law.equilibrium_modulus * s**2
    for j, (e, _) in enumerate(law.branches):
        density = density + e*(s-v[j])**2
    return float(0.5*np.sum(metric*density))


def solve_history(solver, graph, *, material, times, constraints, loads=None,
                  length_unit, initial_state=None):
    """Implementation of BeamFrameFEM.history; see VISCOELASTIC_NETWORK.md."""
    if not isinstance(material, LinearViscoelasticMaterial):
        raise ValueError('material must be a LinearViscoelasticMaterial')
    t = np.asarray(times)
    t = _finite(times, t.shape, 'times')
    if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('times must contain at least two finite strictly increasing samples')
    node_ids, edge_ids, pos, edges, radii, dofs, B, metric = _geometry(graph, length_unit)
    force, prescribed, mask = _histories(t, node_ids, loads, constraints)
    _check_supports(pos, edges, mask)
    n, ne, ndof = len(node_ids), len(edge_ids), 3*len(node_ids)
    free = np.flatnonzero(~mask); fixed = np.flatnonzero(mask)
    length_scale=float(np.max(np.linalg.norm(pos[edges[:,1]]-pos[edges[:,0]],axis=1)))
    # Reuse the original sparse assembler. E=1 yields a geometry/section matrix.
    assembler = type(solver)(E=1.0, nu=solver.nu)
    K0, selected = assembler.build_stiffness_2d(edges.T, pos, radii, deduplicate=False)
    if list(selected) != list(range(ne)) or not np.isfinite(K0.data).all():
        raise ValueError('unsupported element assembly')
    identity = {'node_ids': node_ids, 'edge_ids': edge_ids, 'positions_m': pos.tolist(),
                'edges': edges.tolist(), 'radii_m': radii.tolist(),
                'material': material.to_dict(), 'constraint_mask': mask.tolist(),
                'kinematics': 'linear_2d_euler_bernoulli_two_gauss/1'}
    fingerprint = _hash(identity)
    u = np.zeros(ndof); s = np.zeros((ne,3)); v = np.zeros((len(material.branches),ne,3))
    cumulative = {'physical_dissipation': 0., 'numerical_dissipation': 0., 'work': 0.}
    step = 0; stored = 0.; origin_energy = 0.
    initial_stress = np.zeros((ne,3)); initial_reaction = np.zeros(ndof); initial_residual=0.
    last_rate = np.zeros((ne,3))
    if initial_state is None:
        if np.any(force[0] != 0) or np.any(prescribed[0] != 0):
            raise ValueError('fresh history starts at zero load and zero prescribed displacement')
    else:
        state = copy.deepcopy(initial_state)
        fields = {'schema','fingerprint','node_ids','edge_ids','units','time','step','displacement',
                  'section_strain','viscous_strain','last_load','prescribed_values','constraint_mask',
                  'cumulative','stored_energy','origin_energy','last_stress','last_strain_rate','checksum'}
        if not isinstance(state, dict) or set(state) != fields:
            raise ValueError('invalid restart fields')
        check = dict(state); checksum = check.pop('checksum')
        if checksum != _hash(check) or state['schema'] != _STATE_SCHEMA:
            raise ValueError('restart checksum/schema mismatch')
        if (state['fingerprint'] != fingerprint or state['node_ids'] != node_ids or
                state['edge_ids'] != edge_ids or state['units'] != _UNITS or state['constraint_mask'] != mask.tolist()):
            raise ValueError('restart material/geometry/identity/support mismatch')
        if state['time'] != float(t[0]) or isinstance(state['step'], bool) or not isinstance(state['step'], int) or state['step'] < 0:
            raise ValueError('restart time/step mismatch')
        u = _finite(state['displacement'], (ndof,), 'restart displacement')
        s = _finite(state['section_strain'], (ne,3), 'restart strain')
        v = _finite(state['viscous_strain'], (len(material.branches),ne,3), 'restart viscous strain')
        initial_stress = _finite(state['last_stress'], (ne,3), 'restart stress')
        last_rate = _finite(state['last_strain_rate'], (ne,3), 'restart strain rate')
        reconstructed = material.equilibrium_modulus*s + material.parallel_viscosity*last_rate
        for j,(e,_) in enumerate(material.branches):
            reconstructed = reconstructed + e*(s-v[j])
        if not np.allclose(reconstructed,initial_stress,rtol=1e-12,atol=1e-24):
            raise ValueError('restart stress is incompatible with material history')
        if (not np.array_equal(_finite(state['last_load'], (ndof,), 'restart load'),force[0]) or
                not np.array_equal(_finite(state['prescribed_values'], (ndof,), 'restart prescribed values'),prescribed[0]) or
                not np.array_equal(u[fixed], prescribed[0,fixed])):
            raise ValueError('restart initial controls differ from committed controls')
        if not np.allclose(s, _section(B,dofs,u), rtol=1e-12, atol=1e-14):
            raise ValueError('restart strain is incompatible with displacement')
        if not isinstance(state['cumulative'],dict) or set(state['cumulative']) != set(cumulative):
            raise ValueError('invalid cumulative restart ledger')
        cumulative = {key: float(value) for key,value in state['cumulative'].items()}
        stored, origin_energy = float(state['stored_energy']), float(state['origin_energy'])
        if not all(math.isfinite(x) for x in [stored,origin_energy,*cumulative.values()]):
            raise ValueError('nonfinite restart ledger')
        if stored < 0 or origin_energy < 0 or min(cumulative['physical_dissipation'],cumulative['numerical_dissipation']) < 0:
            raise ValueError('negative restart energy')
        if not math.isclose(stored,_stored(material,s,v,metric),rel_tol=1e-10,abs_tol=1e-24):
            raise ValueError('restart stored energy mismatch')
        balance=cumulative['work']-(stored-origin_energy)-cumulative['physical_dissipation']-cumulative['numerical_dissipation']
        escale=max(abs(cumulative['work']),stored,origin_energy,cumulative['physical_dissipation'],cumulative['numerical_dissipation'],1e-30)
        if abs(balance)>1e-8*escale:
            raise ValueError('restart cumulative ledger is inconsistent')
        step = state['step']
        initial_internal = _assemble(dofs,B,metric,initial_stress,ndof)
        initial_reaction = initial_internal-force[0]
        initial_residual = _residual(initial_reaction,initial_internal,force[0],free,length_scale)
        if initial_residual > 1e-8:
            raise ValueError('restart is not in equilibrium for its committed loads')
    history_u=[u.reshape(n,3).copy()]; history_r=[initial_reaction.reshape(n,3).copy()]
    history_s=[s.copy()]; history_stress=[initial_stress.copy()]; residuals=[initial_residual]
    energy = {'stored':[stored], 'physical_dissipation':[cumulative['physical_dissipation']],
              'numerical_dissipation':[cumulative['numerical_dissipation']],
              'work':[cumulative['work']],
              'balance_residual':[cumulative['work']-(stored-origin_energy)-cumulative['physical_dissipation']-cumulative['numerical_dissipation']]}
    for index in range(1,len(t)):
        dt=float(t[index]-t[index-1]); a, tangent=material.coefficients(dt)
        memory = material.parallel_viscosity/dt*s
        for j,(e,_) in enumerate(material.branches):
            memory = memory + e*a[j]*v[j]
        rhs = force[index] + _assemble(dofs,B,metric,memory,ndof)
        K = tangent*K0
        if not np.isfinite(K.data).all() or not np.isfinite(rhs).all():
            raise FloatingPointError('nonfinite history system')
        candidate = prescribed[index].copy()
        if free.size:
            right = rhs[free] - K[free][:,fixed] @ candidate[fixed]
            with warnings.catch_warnings():
                warnings.simplefilter('error', MatrixRankWarning)
                candidate[free] = spsolve(K[free][:,free],right)
        if not np.isfinite(candidate).all():
            raise FloatingPointError('nonfinite history solution')
        next_s = _section(B,dofs,candidate)
        trial = material.trial(next_s,s,v,dt)
        internal = _assemble(dofs,B,metric,trial['stress'],ndof)
        reaction = internal-force[index]
        residual=_residual(reaction,internal,force[index],free,length_scale)
        if residual > 1e-8:
            raise FloatingPointError('free-DOF equilibrium residual exceeds 1e-8')
        du=candidate-u
        work=float(np.dot(force[index],du)+np.dot(reaction[fixed],du[fixed]))
        new_stored=float(np.sum(metric*trial['stored_energy_density']))
        physical=float(np.sum(metric*trial['physical_dissipation_density']))
        numerical=float(np.sum(metric*trial['numerical_dissipation_density']))
        balance=work-(new_stored-stored)-physical-numerical
        escale=max(abs(work),abs(new_stored),abs(stored),physical,numerical,1e-30)
        if not all(math.isfinite(x) for x in (work,new_stored,physical,numerical,balance)) or abs(balance)>1e-8*escale:
            raise FloatingPointError('discrete history energy identity failed')
        # Transaction boundary: no input state or graph has been changed.
        last_rate=(next_s-s)/dt
        u,s,v,stored=candidate,next_s,trial['viscous_strain'],new_stored
        cumulative['work']+=work; cumulative['physical_dissipation']+=physical
        cumulative['numerical_dissipation']+=numerical; step+=1
        history_u.append(u.reshape(n,3).copy()); history_r.append(reaction.reshape(n,3).copy())
        history_s.append(s.copy()); history_stress.append(trial['stress'].copy()); residuals.append(residual)
        energy['stored'].append(stored)
        for key in cumulative: energy[key].append(cumulative[key])
        energy['balance_residual'].append(cumulative['work']-(stored-origin_energy)-cumulative['physical_dissipation']-cumulative['numerical_dissipation'])
    state={'schema':_STATE_SCHEMA,'fingerprint':fingerprint,'node_ids':node_ids,'edge_ids':edge_ids,
           'units':_UNITS,'time':float(t[-1]),'step':step,'displacement':u.tolist(),
           'section_strain':s.tolist(),'viscous_strain':v.tolist(),'last_load':force[-1].tolist(),
           'prescribed_values':prescribed[-1].tolist(),'constraint_mask':mask.tolist(),
           'cumulative':cumulative,'stored_energy':stored,'origin_energy':origin_energy,
           'last_stress':history_stress[-1].tolist(),'last_strain_rate':last_rate.tolist()}
    state['checksum']=_hash(state)
    stress=np.array(history_stress); A=math.pi*radii**2; I=math.pi*radii**4/4
    return ViscoelasticResult({'schema':_RESULT_SCHEMA,'times':t.tolist(),'node_ids':node_ids,
        'edge_ids':edge_ids,'displacements':np.array(history_u).tolist(),
        'reactions':np.array(history_r).tolist(),'section_strain':np.array(history_s).tolist(),
        'axial_stress':stress[:,:,0].tolist(),'axial_force':(stress[:,:,0]*A).tolist(),
        'bending_moment':(-stress[:,:,1:]*I[None,:,None]).tolist(),
        'free_residual':residuals,'energies':energy,'final_state':state,
        'metadata':{'units':_UNITS,'material':material.to_dict(),'backend':'fem',
            'solver':'BeamFrameFEM.history','kinematics':'linear_2d_euler_bernoulli',
            'quadrature':'two_point_Gauss_bending','integration':'backward_euler',
            'bending_moment_sign':'M=-I*generalized_curvature_stress',
            'residual_moment_normalization_length_m':length_scale,
            'work_convention':'right_endpoint_load_plus_constraint_work',
            'graph_length_unit':length_unit,'material_assignment':'explicit_uniform',
            'graph_static_moduli_used':False,'numerical_regularization':False,
            'assumptions':['small_strain','small_rotation','straight_members','welded_joints'],
            'initialization':'restart' if initial_state is not None else 'zero_unloaded',
            'fingerprint':fingerprint}})
