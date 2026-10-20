"""Conservative, explicit coarse beads; deterministic input, never a solver.

Cosine bending has its minimum at a straight triplet. Curved initial strands
therefore store bending energy. Beads can fold and cross; this is neither a
continuous fiber contact law nor calibrated silk or adhesion hysteresis.
"""
from dataclasses import dataclass
import json
import math
from typing import Mapping, Optional

from .lammps_backend import (HarmonicBond, RunSpec, PreparedRun, LAMMPSContractError,
                            StructureGraph, LENGTH_FACTORS, _number, _integer,
                            _vector, _json, _digest)

MODEL = 'conservative-ball-cosine-strand-cg/1'
GROUP_ENCODING = 'static-groups-zero-cutoff/1'


def _group_encoding_contract():
    return {'name': GROUP_ENCODING, 'native_equivalence': 'NOT_YET_VALIDATED',
            'disabled_coefficients': [0.0, 1.0, 0.0, 0.0],
            'exclusions': ['strand-strand', 'projectile-or-unselected self-group'],
            'diagnostics': 'generated run/dump/thermo only; single() and custom commands unsupported',
            'source_reference_commit': '611ca3b7f8525ba802373b04f9e3d632d515f7e3'}

from . import _cg_network_profile as network

SCHEMA = 'lammps_cg_state/1'
FORCE_ROLES = {'edge_forces':'harmonic_bond_axial','max_force':'maximum_absolute_harmonic_bond_axial','dynamics_state.nodal_forces':'native_net_nodal_after_fixes'}


@dataclass(frozen=True)
class CGAngle:
    node_i: int
    node_j: int
    node_k: int
    edge_ij: int
    edge_jk: int
    K_J: float


@dataclass(frozen=True)
class MorsePair:
    D_J: float
    alpha_per_m: float
    r0_m: float
    cutoff_m: float
    role: str = 'attractive_site'


@dataclass(frozen=True)
class CGRequest:
    """Explicit bead model; a null projectile admits only pair-free small pilots.

    Pass ``projectile_node_id=None``, ``projectile_pairs={}``, and leave
    ``pair_encoding`` and ``profile`` unset for a fiber-only network. All
    existing positional arguments retain their order.
    """
    length_unit: str
    maximum_bond_length_m: float
    masses_kg: Mapping[int, float]
    bonds: Mapping[int, HarmonicBond]
    angles: Mapping[int, CGAngle]
    projectile_node_id: Optional[int]
    projectile_pairs: Mapping[int, MorsePair]
    run: RunSpec
    fixed_node_ids: tuple = ()
    velocities_m_s: Mapping = None
    pair_encoding: str = None
    profile: str = None
    thermo_every: int = None
    neighbor_policy: Mapping = None

    def __post_init__(self):
        # Preserve legacy construction semantics; validate the new opt-in only.
        if self.projectile_node_id is None:
            if not isinstance(self.projectile_pairs, Mapping):
                raise LAMMPSContractError('null projectile requires an empty pair mapping')
            _projectile_identity(None, self.projectile_pairs, self.pair_encoding, self.profile)


def _projectile_identity(value, pairs, pair_encoding=None, profile=None):
    """Validate the null branch before any projectile-dependent work."""
    if value is None:
        if pairs or pair_encoding is not None or profile is not None:
            raise LAMMPSContractError('null projectile requires no pairs, pair encoding or network profile')
        return None
    return _integer(value, 'projectile node ID')


def _fiber_only_declarations():
    return dict(pair_model='none',
        pair_policy='no nonbonded interactions; all type pairs excluded',
        required_packages=['MOLECULE'],
        required_styles={'atom':['angle'],'bond':['harmonic'],'angle':['cosine'],'pair':['zero'],'fix':['nve']},
        energy='sum of harmonic bonds and cosine angles; J')



NEIGHBOR_POLICY = 'cached-nsq-si/1'


def _neighbor_policy(value, box, cutoffs, serialized=False):
    """Strict opt-in only; native neighbor completeness is not trajectory accuracy."""
    if value is None:
        return None
    if type(value) is not dict or len(value) != (10 if serialized else 2) or set(value) != ({'name', 'skin_m', 'style', 'every', 'delay', 'check', 'once', 'displacement_group', 'boundary', 'qualification'} if serialized else {'name', 'skin_m'}):
        raise LAMMPSContractError('explicit bounded cached NSQ policy required')
    if type(value['name']) is not str or value['name'] != NEIGHBOR_POLICY:
        raise LAMMPSContractError('unknown neighbor policy')
    skin = _number(value['skin_m'], 'neighbor SI skin', True)
    half = skin/2
    _number(half*half, 'native neighbor displacement threshold squared', True)
    if skin > min(hi-lo for lo,hi in box)/2:
        raise LAMMPSContractError('neighbor skin exceeds half smallest fixed box width')
    for lo,hi in box:
        if any(abs(x)+half == abs(x) for x in (lo,hi)):
            raise LAMMPSContractError('neighbor skin lost at box coordinate precision')
    for cutoff in cutoffs:
        expanded = _number(cutoff+skin, 'expanded neighbor cutoff', True)
        if expanded <= cutoff:
            raise LAMMPSContractError('neighbor skin lost at cutoff precision')
        _number(expanded*expanded, 'native expanded neighbor cutoff squared', True)
    policy = dict(name=NEIGHBOR_POLICY, skin_m=skin, style='nsq', every=1,
                  delay=0, check=True, once=False, displacement_group='all',
                  boundary='fixed', qualification='source-only; moving equivalence pending')
    if serialized and _json(value) != _json(policy):
        raise LAMMPSContractError('neighbor policy declarations mismatch')
    return policy


def morse_energy(r, coefficients):
    """Shifted-force Morse energy (J), not a native execution qualification."""
    D, a, r0, rc = coefficients
    if r >= rc: return 0.0
    try:
        e = math.exp(-a*(r-r0)); ec = math.exp(-a*(rc-r0))
        return _number(D*(e*e-2*e)-D*(ec*ec-2*ec)-(r-rc)*2*a*D*(ec-ec*ec), 'Morse energy')
    except OverflowError as exc:
        raise LAMMPSContractError('Morse parameter arithmetic overflows') from exc



def check_morse_coefficients(c):
    if not isinstance(c,(tuple,list)) or len(c) != 4:
        raise LAMMPSContractError("Morse coefficients require four SI scalars")
    c = tuple(_number(v,"Morse coefficient",True) for v in c)
    # Bound exponent and shifted-force arithmetic even at maximum compression.
    morse_energy(0.,c)
    D,alpha,r0,rc = c
    gap = _number(alpha*(rc-r0),'Morse cutoff gap',True)
    # A 64-ulp absolute guard avoids a binary64 borderline attractive branch.
    if gap <= math.log(2.) + 64*math.ulp(math.log(2.)):
        raise LAMMPSContractError('Morse cutoff must support a resolved attractive branch')
    zc = math.exp(-gap)
    minimum = r0-math.log1p(-zc)/alpha
    if not (0 < zc < .5 and math.isfinite(minimum) and r0 < minimum < rc and morse_energy(minimum,c) < 0):
        raise LAMMPSContractError('shifted Morse minimum must be representable with negative energy')
    return c



def _pair_roles_contract():
    return dict(name='explicit-attractive-repulsive-morse/1',
        attractive_site='unchanged resolved attractive shifted-force Morse',
        repulsive_site='cutoff equals r0; D*(exp(-alpha*(r-r0))-1)^2 below cutoff; zero outside',
        limitations='finite penetrable core; selected coincidence refused; nodal sites, not continuous fiber contact',
        native_validation='not established by source compilation')


def check_pair_coefficients(c, role):
    if type(role) is not str or role not in ('attractive_site','repulsive_site'):
        raise LAMMPSContractError('unknown explicit pair role')
    if role == 'attractive_site':return check_morse_coefficients(c)
    if not isinstance(c,(tuple,list)) or len(c)!=4:
        raise LAMMPSContractError('Morse coefficients require four SI scalars')
    c=tuple(_number(v,'repulsive Morse coefficient',True) for v in c)
    if c[3] != c[2]:raise LAMMPSContractError('repulsive cutoff must exactly equal r0')
    _number(morse_energy(0.,c),'resolved repulsive core energy',True)
    # Native force prefactor must remain finite at maximum compression.
    try:
        z=math.exp(c[1]*c[2]);_number(2*c[0]*c[1]*z*(z-1),'repulsive core force',True)
    except OverflowError as exc:raise LAMMPSContractError('repulsive coefficient arithmetic overflows') from exc
    return c


def _contract_roles(c):
    if not isinstance(c.get('projectile_pairs'),list):raise LAMMPSContractError('missing selected pair table')
    for row in c['projectile_pairs']:
        if not isinstance(row,(list,tuple)) or len(row)!=2:raise LAMMPSContractError('invalid selected pair row')
        _integer(row[0],'selected pair stable ID')
    if 'pair_roles' not in c and 'projectile_pair_roles' not in c:
        return {n:'attractive_site' for n,_ in c['projectile_pairs']}
    if _json(c.get('pair_roles')) != _json(_pair_roles_contract()):
        raise LAMMPSContractError('pair role declaration mismatch')
    rows=c.get('projectile_pair_roles')
    if not isinstance(rows,list) or len(rows)!=len(c['projectile_pairs']):
        raise LAMMPSContractError('pair roles must exactly cover selected IDs')
    ids=[];roles={}
    for row in rows:
        if not isinstance(row,list) or len(row)!=2:raise LAMMPSContractError('invalid pair role row')
        n,role=row;_integer(n,'pair role stable ID')
        if type(role)is not str or role not in ('attractive_site','repulsive_site'):raise LAMMPSContractError('unknown pair role')
        ids.append(n);roles[n]=role
    if ids != [n for n,_ in c['projectile_pairs']] or ids != sorted(set(ids)) or 'repulsive_site' not in roles.values():
        raise LAMMPSContractError('noncanonical pair role IDs or redundant declaration')
    return roles


def prepare_cg(graph, request):
    """Compile bounded explicit beads, optionally pair-free with no projectile.

    The null-projectile branch is limited to 64 beads and 128 bonds/angles;
    existing explicit integer-projectile network profiles retain their bounds.
    """
    if type(request) is not CGRequest or not isinstance(graph, StructureGraph) or graph.dimension != 3:
        raise LAMMPSContractError('expected CGRequest and authoritative 3-D StructureGraph')
    q = request
    projectile = _projectile_identity(q.projectile_node_id, q.projectile_pairs,
                                      q.pair_encoding, q.profile)
    if q.pair_encoding is not None and (type(q.pair_encoding) is not str or q.pair_encoding != GROUP_ENCODING):
        raise LAMMPSContractError('unknown CG pair encoding')
    grouped = q.pair_encoding == GROUP_ENCODING
    profile = network.select(q.profile)
    if profile and not grouped:raise LAMMPSContractError('network profile requires reviewed group encoding')
    if not profile and q.thermo_every is not None:raise LAMMPSContractError('thermo_every requires network profile')
    limits = profile or dict(max_nodes=64,max_edges=128,max_angles=128,max_pairs=63,prepared_bytes=1024**2,max_steps=1000)

    if q.length_unit not in LENGTH_FACTORS or type(q.run) is not RunSpec:
        raise LAMMPSContractError('explicit supported length unit and RunSpec required')
    dt, steps, every, box = q.run.checked(max_steps=limits['max_steps'])
    if not profile and (steps > 1000 or every != 1): raise LAMMPSContractError('CG requires 0..1000 steps and every frame')
    if profile:network.schedule(steps,every,q.thermo_every)
    nodes, edges = tuple(sorted(graph.nodes)), tuple(sorted(graph.edges))
    if not 2 <= len(nodes) <= limits['max_nodes'] or len(edges) > limits['max_edges']:
        raise LAMMPSContractError('CG graph count exceeds bounded pilot')
    for mapping,cap in ((q.masses_kg,limits['max_nodes']), (q.bonds,limits['max_edges']), (q.angles,limits['max_angles']), (q.projectile_pairs,limits['max_pairs']), (q.velocities_m_s or {},limits['max_nodes'])):
        if not isinstance(mapping, Mapping) or len(mapping) > (cap if profile else 128):
            raise LAMMPSContractError('bounded explicit mappings required')
        for key in mapping: _integer(key, 'mapping stable ID')
    for key in (*nodes, *edges): _integer(key, 'graph stable ID')
    if set(q.masses_kg) != set(nodes) or set(q.bonds) != set(edges):
        raise LAMMPSContractError('masses and bonds must exactly cover graph IDs')
    if (projectile is not None and projectile not in nodes) or len(q.projectile_pairs) > limits['max_pairs']:
        raise LAMMPSContractError('invalid projectile identity or pair count')
    scale = LENGTH_FACTORS[q.length_unit]
    reach = _number(q.maximum_bond_length_m, 'maximum_bond_length_m', True)
    if reach > math.hypot(*(hi-lo for lo,hi in box)):
        raise LAMMPSContractError('maximum bond length exceeds box diagonal')
    points, masses = {}, {}
    for n in nodes:
        if _integer(graph.nodes[n].node_id,'node record ID') != n: raise LAMMPSContractError('node ID mismatch')
        points[n] = tuple(_number(x*scale, 'SI position') for x in _vector(graph.nodes[n].position, 'position'))
        if any(not box[a][0] < points[n][a] < box[a][1] for a in range(3)):
            raise LAMMPSContractError('positions must be strictly inside box')
        masses[n] = _number(q.masses_kg[n], 'nodal mass', True)
    if not isinstance(q.fixed_node_ids,(tuple,list)) or len(q.fixed_node_ids) > limits['max_nodes']:
        raise LAMMPSContractError('fixed IDs require a bounded tuple/list')
    fixed = tuple(sorted(_integer(n,'fixed node ID') for n in q.fixed_node_ids))
    if len(set(fixed)) != len(fixed) or not set(fixed) <= set(nodes) or len(fixed) == len(nodes):
        raise LAMMPSContractError('invalid fixed node IDs')
    velocities = {n:_vector(v,'velocity') for n,v in (q.velocities_m_s or {}).items()}
    if not set(velocities) <= set(nodes) or any(any(velocities.get(n,(0,0,0))) for n in fixed):
        raise LAMMPSContractError('invalid velocity node or moving fixed bead')
    bonds, seen = {}, set()
    for e in edges:
        edge = graph.edges[e]
        i,j = _integer(edge.node_i,'edge endpoint'),_integer(edge.node_j,'edge endpoint')
        pair = tuple(sorted((i,j)))
        if _integer(edge.edge_id,'edge record ID') != e or i == j or not set(pair) <= set(nodes) or projectile in pair or pair in seen:
            raise LAMMPSContractError('invalid/multiple edge or nonisolated projectile')
        seen.add(pair)
        if edge.internal_points is not None and len(edge.internal_points):
            raise LAMMPSContractError('curved edges cannot be flattened')
        if type(q.bonds[e]) is not HarmonicBond: raise LAMMPSContractError('explicit HarmonicBond required')
        k,l0 = q.bonds[e].checked(); _number(k/2,'native harmonic K',True)
        r = math.dist(points[i],points[j])
        if not 0 < r <= reach or l0 > reach: raise LAMMPSContractError('bond outside declared domain')
        _number(.5*k*(r-l0)**2,'initial bond energy')
        bonds[e] = (k,l0)
    angles, seen_angles, initial_bending = {}, set(), 0.0
    for aid,a in sorted(q.angles.items()):
        if type(a) is not CGAngle: raise LAMMPSContractError('explicit CGAngle required')
        i,j,k,e,f = (_integer(x,'angle stable ID') for x in (a.node_i,a.node_j,a.node_k,a.edge_ij,a.edge_jk))
        if len({i,j,k}) != 3 or projectile in (i,j,k) or not {i,j,k} <= set(nodes) or e == f or not {e,f} <= set(edges):
            raise LAMMPSContractError('invalid angle node/edge IDs')
        if {graph.edges[e].node_i,graph.edges[e].node_j} != {i,j} or {graph.edges[f].node_i,graph.edges[f].node_j} != {j,k}:
            raise LAMMPSContractError('angle must reference its two graph edges')
        if i > k: i,k,e,f = k,i,f,e
        if (i,j,k) in seen_angles: raise LAMMPSContractError('duplicate/reversed angle')
        seen_angles.add((i,j,k)); K = _number(a.K_J,'cosine K_J',True)
        u = tuple((x-y)/math.dist(points[i],points[j]) for x,y in zip(points[i],points[j]))
        v = tuple((x-y)/math.dist(points[k],points[j]) for x,y in zip(points[k],points[j]))
        initial_bending += K*(1+max(-1.,min(1.,sum(x*y for x,y in zip(u,v)))))
        angles[aid] = (i,j,k,e,f,K)
    _number(initial_bending,'initial bending energy')
    pairs = {}
    pair_roles = {}
    for n,p in sorted(q.projectile_pairs.items()):
        if n not in nodes or n == projectile or type(p) is not MorsePair:
            raise LAMMPSContractError('pair must select projectile and one known strand bead')
        c = tuple(_number(x,label,True) for x,label in zip((p.D_J,p.alpha_per_m,p.r0_m,p.cutoff_m),('D_J','alpha_per_m','r0_m','cutoff_m')))
        check_pair_coefficients(c,p.role)
        pair_roles[n] = p.role
        if grouped or p.role == 'repulsive_site':
            _number(c[3]*c[3], 'native squared selected cutoff', True)
            dx,dy,dz = (points[n][i]-points[projectile][i] for i in range(3))
            _number(dx*dx+dy*dy+dz*dz, 'native squared selected separation', True)
        if math.dist(points[n],points[projectile]) == 0: raise LAMMPSContractError('coincident selected pair')
        if p.role == 'repulsive_site':
            r=math.sqrt(dx*dx+dy*dy+dz*dz)
            if r<c[3]:
                z=math.exp(-c[1]*(r-c[2]))
                _number((2*c[0]*c[1])*(z*z-z)/r,'initial native repulsive force multiplier')
        pairs[n] = c
    mixed = 'repulsive_site' in pair_roles.values()
    if profile and mixed != (q.profile == network.MIXED_NAME):
        raise LAMMPSContractError('mixed pairs require the explicit mixed network profile and vice versa')
    atom = {n:i+1 for i,n in enumerate(nodes)}
    cutoff = max([reach]+[p[3] for p in pairs.values()])
    neighbor = _neighbor_policy(q.neighbor_policy,box,[cutoff]+[p[3] for p in pairs.values()])
    contract = dict(model=MODEL, length_unit=q.length_unit,length_to_m=scale,units='si',
        maximum_bond_length_m=reach,masses_kg=sorted(masses.items()),bonds=sorted(bonds.items()),
        angles=sorted(angles.items()),angle_model='cosine-straight/1',initial_bending_energy_J=initial_bending,
        projectile_node_id=projectile,projectile_pairs=sorted(pairs.items()),
        pair_policy='explicit selected projectile-strand pairs; all other type pairs excluded',
        pair_model='morse/smooth/linear',atom_type_key_schema='stable-node-id/1',
        atom_to_type_id=[(atom[n],atom[n]) for n in nodes],type_to_mass_kg=[(atom[n],masses[n]) for n in nodes],
        type_pair_cutoff_payload_lower_bound_bytes=16*(len(nodes)+1)**2,
        fixed_node_ids=fixed,forces_N=[],velocities_m_s=sorted(velocities.items()),
        timestep_s=dt,steps=steps,sample_every=every,box_m=box,boundary=['f']*3,
        required_packages=['MOLECULE','EXTRA-PAIR'],
        required_styles={'atom':['angle'],'bond':['harmonic'],'angle':['cosine'],'pair':['zero','hybrid/overlay','morse/smooth/linear'],'fix':['nve']},
        limitations='coarse beads; no continuous fiber contact, impenetrability, calibrated silk, adhesion hysteresis or inferred EI',
        energy='sum of harmonic bonds, cosine angles and shifted-force Morse pairs; J',
        stress='global pressure tensor Pa, not material stress',native_validation='not established by source compilation')
    if projectile is None:
        contract.update(_fiber_only_declarations())
    if neighbor is not None:
        contract['neighbor_policy'] = neighbor
    if mixed:
        contract['pair_roles'] = _pair_roles_contract()
        contract['projectile_pair_roles'] = sorted(pair_roles.items())
    if grouped:
        contract['pair_encoding'] = _group_encoding_contract()
        contract['pair_policy'] = 'explicit selected projectile-strand pairs; all other physical pairs statically group-excluded'
        contract['required_styles']['pair'] = ['morse/smooth/linear']
    if profile:
        out,thermo=network.schedule(steps,every,q.thermo_every)
        contract.update(profile=profile,thermo_every=q.thermo_every,output_frames=out,
            thermo_frames=thermo,integration_steps=steps,observation_scope=network.SCOPE)
        network.contract(contract)
    from . import _harmonic_profile as bounds
    for edge in graph.edges.values():
        tensor = getattr(edge.material,'stiffness_tensor',None)
        if tensor is not None and getattr(tensor,'size',37) > 36:
            raise LAMMPSContractError('CG material tensor exceeds bound')
    bounds.json_bytes([graph.metadata]+[n.metadata for n in graph.nodes.values()]+[e.metadata for e in graph.edges.values()],1024*1024)
    bounds.json_bytes([e.material.to_dict() for e in graph.edges.values()],1024*1024)
    graph_json = _json(graph.to_dict())
    lines = ['FiberNet conservative coarse bead model (SI)','',f'{len(nodes)} atoms',f'{len(edges)} bonds',f'{len(angles)} angles','',f'{len(nodes)} atom types',f'{len(edges)} bond types',f'{len(angles)} angle types','']
    lines += [f'{lo:.17g} {hi:.17g} {axis}lo {axis}hi' for (lo,hi),axis in zip(box,'xyz')]
    lines += ['', 'Masses','']+[f'{atom[n]} {masses[n]:.17g}' for n in nodes]
    lines += ['', 'Atoms # angle','']+[f'{atom[n]} 1 {atom[n]} '+' '.join(f'{x:.17g}' for x in points[n]) for n in nodes]
    if edges: lines += ['', 'Bonds','']+[f'{i} {i} {atom[graph.edges[e].node_i]} {atom[graph.edges[e].node_j]}' for i,e in enumerate(edges,1)]
    if angles: lines += ['', 'Angles','']+[f'{i} {i} {atom[a[0]]} {atom[a[1]]} {atom[a[2]]}' for i,a in enumerate(angles.values(),1)]
    lines += ['', 'Velocities','']+[f'{atom[n]} '+' '.join(f'{x:.17g}' for x in velocities.get(n,(0,0,0))) for n in nodes]
    script = ['units si','dimension 3','boundary f f f','atom_style angle','atom_modify map array','bond_style harmonic','angle_style cosine','read_data model.data']
    script += [f'bond_coeff {i} {bonds[e][0]/2:.17g} {bonds[e][1]:.17g}' for i,e in enumerate(edges,1)]
    script += [f'angle_coeff {i} {a[5]:.17g}' for i,a in enumerate(angles.values(),1)]
    if grouped:
        # Safe finite disabled initialization; exact cutoff zero also gates compute().
        # Group exclusions are essential protection, not a claim about single().
        script += [f'pair_style morse/smooth/linear {cutoff:.17g}',
                   'pair_coeff * * 0 1 0 0']
        for n,p in sorted(pairs.items()):
            i,j = sorted((atom[projectile],atom[n]))
            script += [f'pair_coeff {i} {j} '+' '.join(f'{x:.17g}' for x in p)]
        strand = [atom[n] for n in nodes if n != projectile]
        disabled = [atom[n] for n in nodes if n == projectile or n not in pairs]
        script += ['group cg_strand id '+' '.join(map(str,strand)),
                   'group cg_disabled id '+' '.join(map(str,disabled)),
                   'neigh_modify exclude group cg_strand cg_strand',
                   'neigh_modify exclude group cg_disabled cg_disabled']
    else:
        script += [f'pair_style hybrid/overlay zero {cutoff:.17g} morse/smooth/linear {cutoff:.17g}'] if pairs else [f'pair_style zero {cutoff:.17g}']
        selected = {tuple(sorted((atom[projectile],atom[n]))):p for n,p in pairs.items()}
        for i in range(1,len(nodes)+1):
            for j in range(i,len(nodes)+1):
                script += [f'pair_coeff {i} {j}'+(' zero' if pairs else '')]
                if (i,j) in selected: script += [f'pair_coeff {i} {j} morse/smooth/linear '+' '.join(f'{x:.17g}' for x in selected[i,j])]
                else: script += [f'neigh_modify exclude type {i} {j}']
    if neighbor is None:
        script += ['special_bonds lj/coul 1 1 1', f'comm_modify cutoff {max(reach*2,cutoff):.17g}','neighbor 0.0 nsq','neigh_modify every 1 delay 0 check no']
    else:
        communication = _number(max(reach*2,cutoff+neighbor['skin_m']), 'communication cutoff', True)
        script += ['special_bonds lj/coul 1 1 1', f'comm_modify cutoff {communication:.17g}',
                   f'neighbor {neighbor["skin_m"]:.17g} nsq', 'neigh_modify every 1 delay 0 check yes once no']
    _number(reach*2,'angle communication reach',True)
    script += ['group held id '+' '.join(str(atom[n]) for n in fixed),'group mobile subtract all held'] if fixed else ['group mobile union all']
    script += ['fix integrate mobile nve',f'timestep {dt:.17g}',f'thermo {q.thermo_every if profile else 1}','thermo_style custom step atoms pe ke etotal pxx pyy pzz pxy pxz pyz','thermo_modify norm no flush yes format float %.17g lost error',f'dump trajectory all custom {every if profile else 1} trajectory.dump id type x y z vx vy vz fx fy fz','dump_modify trajectory sort id units yes time yes format float %.17g',f'run {steps}']
    data = '\n'.join(lines)+'\n'; cjson = _json(contract)
    marker = 'FIBERNET_LAMMPS_COMPLETE_'+_digest(_json([graph_json,cjson,data]))
    script += [f'print "{marker}"']
    if sum(len(s.encode()) for s in (graph_json,cjson,data,'\n'.join(script))) > limits['prepared_bytes']:
        raise LAMMPSContractError('CG prepared provenance exceeds 1 MiB')
    return PreparedRun(graph_json,cjson,data,'\n'.join(script)+'\n',nodes,edges,dt,steps,every,profile['name'] if profile else None)


def regenerate(prepared, c):
    _projectile_identity(c['projectile_node_id'], c['projectile_pairs'],
                         c.get('pair_encoding'), c.get('profile'))
    profile = network.contract(c)
    if prepared.profile != (profile['name'] if profile else None):raise LAMMPSContractError('CG profile mismatch')
    limits=profile or dict(max_nodes=64,max_edges=128)
    record = json.loads(prepared.graph_json)
    if not 2 <= len(record['nodes']) <= limits['max_nodes'] or len(record['edges']) > limits['max_edges']:
        raise LAMMPSContractError('CG serialized graph exceeds count bounds')
    def tensor_size(value):
        if isinstance(value,(list,tuple)):
            if len(value)>36: raise LAMMPSContractError('serialized material tensor exceeds bound')
            return sum(tensor_size(v) for v in value)
        return 1
    for edge in record['edges']:
        tensor = edge.get('material',{}).get('stiffness_tensor')
        if tensor is not None and tensor_size(tensor)>36:
            raise LAMMPSContractError('serialized material tensor exceeds bound')
    roles = _contract_roles(c)
    neighbor = _neighbor_policy(c.get('neighbor_policy'),c['box_m'],[c['maximum_bond_length_m']]+[v[3] for _,v in c['projectile_pairs']],serialized=True)
    if 'neighbor_policy' in c and neighbor is None:raise LAMMPSContractError('null serialized neighbor policy')
    return prepare_cg(StructureGraph.from_dict(record),CGRequest(
        c['length_unit'],c['maximum_bond_length_m'],dict(c['masses_kg']),
        {e:HarmonicBond(*v) for e,v in c['bonds']},{a:CGAngle(*v) for a,v in c['angles']},
        c['projectile_node_id'],{n:MorsePair(*v,role=roles[n]) for n,v in c['projectile_pairs']},
        RunSpec(c['timestep_s'],c['steps'],c['sample_every'],c['box_m']),tuple(c['fixed_node_ids']),dict(c['velocities_m_s']),
        c['pair_encoding']['name'] if 'pair_encoding' in c else None, profile['name'] if profile else None,c.get('thermo_every'),
        {k:neighbor[k] for k in ('name','skin_m')} if neighbor else None))


def guard_result(metadata, n, e, state, arrays=None):
    """Narrow result admission; CG never borrows harmonic profile/count rules."""
    c = metadata['model_contract']
    projectile = _projectile_identity(c.get('projectile_node_id'), c.get('projectile_pairs'),
                                      c.get('pair_encoding'), c.get('profile'))
    from ._harmonic_profile import json_bytes
    profile=network.contract(c)
    limits=profile or dict(max_nodes=64,max_edges=128,max_angles=128,max_pairs=63,max_steps=1000,prepared_bytes=1024**2)
    json_bytes(c,limits['prepared_bytes'])
    if metadata.get('initial_bending_energy_J') != c.get('initial_bending_energy_J'):
        raise ValueError('CG initial bending energy metadata mismatch')
    expected = {'model','length_unit','length_to_m','units','maximum_bond_length_m','masses_kg','bonds','angles','angle_model','initial_bending_energy_J','projectile_node_id','projectile_pairs','pair_policy','pair_model','atom_type_key_schema','atom_to_type_id','type_to_mass_kg','type_pair_cutoff_payload_lower_bound_bytes','fixed_node_ids','forces_N','velocities_m_s','timestep_s','steps','sample_every','box_m','boundary','required_packages','required_styles','limitations','energy','stress','native_validation'}
    if 'neighbor_policy' in c:
        expected.add('neighbor_policy')
        if c['neighbor_policy'] is None:raise ValueError('null serialized neighbor policy')
        _neighbor_policy(c['neighbor_policy'],c['box_m'],[c['maximum_bond_length_m']]+[v[3] for _,v in c['projectile_pairs']],serialized=True)
    if profile:expected.update(('profile','thermo_every','output_frames','thermo_frames','integration_steps','observation_scope'))
    roles = _contract_roles(c)
    if 'pair_roles' in c:expected.update(('pair_roles','projectile_pair_roles'))
    if profile and ('repulsive_site' in roles.values()) != (profile['name']==network.MIXED_NAME):raise ValueError('pair/profile scope mismatch')
    grouped = 'pair_encoding' in c
    if grouped:
        expected.add('pair_encoding')
        # Canonical JSON is type-sensitive: booleans cannot impersonate sentinel zeros.
        if _json(c['pair_encoding']) != _json(_group_encoding_contract()):
            raise ValueError('CG pair encoding declaration mismatch')
    if metadata.get('model_contract_sha256') != _digest(_json(c)):
        raise ValueError('CG result contract digest mismatch')
    if set(c) != expected or c['model'] != MODEL:
        raise ValueError('CG result contract fields mismatch')
    declarations = dict(units='si',boundary=['f']*3,angle_model='cosine-straight/1',pair_model='morse/smooth/linear',atom_type_key_schema='stable-node-id/1',
        pair_policy='explicit selected projectile-strand pairs; all other type pairs excluded',
        required_packages=['MOLECULE','EXTRA-PAIR'],
        required_styles={'atom':['angle'],'bond':['harmonic'],'angle':['cosine'],'pair':['zero','hybrid/overlay','morse/smooth/linear'],'fix':['nve']},
        limitations='coarse beads; no continuous fiber contact, impenetrability, calibrated silk, adhesion hysteresis or inferred EI',
        energy='sum of harmonic bonds, cosine angles and shifted-force Morse pairs; J',
        stress='global pressure tensor Pa, not material stress',native_validation='not established by source compilation')
    if projectile is None:
        declarations.update(_fiber_only_declarations())
    if grouped:
        declarations['pair_policy'] = 'explicit selected projectile-strand pairs; all other physical pairs statically group-excluded'
        declarations['required_styles']['pair'] = ['morse/smooth/linear']
    if any(c.get(k)!=v for k,v in declarations.items()):raise ValueError('CG fixed declarations mismatch')
    if c['length_unit'] not in LENGTH_FACTORS or c['length_to_m'] != LENGTH_FACTORS[c['length_unit']]:
        raise ValueError('CG graph units mismatch')
    _number(c['length_to_m'],'CG length scale',True)
    reach = _number(c['maximum_bond_length_m'],'CG reach',True)
    _,_,_,box=RunSpec(c['timestep_s'],c['steps'],c['sample_every'],c['box_m']).checked(max_steps=limits['max_steps'])
    if reach > math.hypot(*(hi-lo for lo,hi in box)):raise ValueError('CG reach exceeds box')
    if type(n) is not int or type(e) is not int or not 2 <= n <= limits['max_nodes'] or not 0 <= e <= limits['max_edges']:
        raise ValueError('CG result count bound exceeded')
    if ((not profile and ('profile' in metadata or 'profile' in c or (state is not None and 'profile' in state))) or (state is not None and state.get('schema') != SCHEMA)):
        raise ValueError('CG result schema/profile mismatch')
    if c.get('atom_type_key_schema') != 'stable-node-id/1' or c.get('angle_model') != 'cosine-straight/1' or c.get('pair_model') != declarations['pair_model']:
        raise ValueError('CG contract model declarations mismatch')
    from ._harmonic_profile import pairs
    for key,count,contract_key in (('node_ids',n,'masses_kg'),('edge_ids',e,'bonds')):
        ids = metadata.get(key)
        if not isinstance(ids,(list,tuple)) or len(ids) != count or any(type(x) is not int or not 0 <= x < 2**31 for x in ids) or list(ids) != sorted(set(ids)):
            raise ValueError('CG result stable IDs invalid')
        if pairs(c.get(contract_key),contract_key,count) != set(ids) or [row[0] for row in c[contract_key]] != list(ids):
            raise ValueError('CG result contract IDs mismatch')
    if any(type(a)is not int or type(t)is not int for a,t in c['atom_to_type_id']) or any(type(t)is not int or isinstance(m,bool) for t,m in c['type_to_mass_kg']):
        raise ValueError('CG atom/type IDs must be integers')
    if c.get('atom_to_type_id') != [[i,i] for i in range(1,n+1)] or c.get('type_to_mass_kg') != [[i+1,m] for i,(_,m) in enumerate(c['masses_kg'])]:
        raise ValueError('CG type/mass identity mismatch')
    for _,mass in c['masses_kg']: _number(mass,'CG mass',True)
    if type(c['type_pair_cutoff_payload_lower_bound_bytes'])is not int or c['type_pair_cutoff_payload_lower_bound_bytes']!=16*(n+1)**2:
        raise ValueError('CG type table size mismatch')
    if profile:
        if network.record(metadata.get('profile'))!=profile:raise ValueError('network metadata profile mismatch')
        if state is not None:
            if network.record(state.get('profile'))!=profile:raise ValueError('network state profile mismatch')
            for key in ('thermo_every','sample_every','output_frames','thermo_frames','integration_steps','observation_scope'):
                if type(state.get(key))is not type(c[key]) or state[key]!=c[key]:raise ValueError('network state schedule mismatch')
    _integer(c.get('steps'),'CG steps',0,limits['max_steps'])
    if not profile and (c.get('sample_every') != 1 or type(c.get('sample_every')) is not int):
        raise ValueError('CG requires every frame')
    _number(c.get('timestep_s'),'CG timestep',True)
    if len(c['angles']) > limits['max_angles'] or len(c['projectile_pairs']) > limits['max_pairs']:
        raise ValueError('CG interaction count exceeded')
    pairs(c['angles'],'angles'); pairs(c['projectile_pairs'],'projectile pairs')
    node_set,edge_set = set(metadata['node_ids']),set(metadata['edge_ids'])
    for _,bond in c['bonds']:
        if not isinstance(bond,(list,tuple)) or len(bond)!=2: raise ValueError('invalid CG bond shape')
        HarmonicBond(*bond).checked()
    angle_keys=set()
    for _,angle in c['angles']:
        if not isinstance(angle,(list,tuple)) or len(angle)!=6: raise ValueError('invalid CG angle shape')
        i,j,k,e1,e2 = (_integer(x,'angle stable ID') for x in angle[:5])
        if len({i,j,k})!=3 or not {i,j,k}<=node_set or projectile in (i,j,k) or i>=k or e1==e2 or not {e1,e2}<=edge_set or (i,j,k) in angle_keys:
            raise ValueError('invalid CG angle identity/roles')
        angle_keys.add((i,j,k));_number(angle[5],'CG angle K',True)
    for nid,coeff in c['projectile_pairs']:
        check_pair_coefficients(coeff,roles[nid])
        if grouped: _number(coeff[3]*coeff[3], 'native squared selected cutoff', True)
    fixed=c['fixed_node_ids']
    if not isinstance(fixed,(tuple,list)) or len(fixed)>=n or any(type(x)is not int for x in fixed) or len(set(fixed))!=len(fixed) or not set(fixed)<=node_set:
        raise ValueError('CG fixed IDs invalid')
    if c['forces_N']!=[]:raise ValueError('CG does not admit constant forces')
    pairs(c['velocities_m_s'],'velocities')
    for nid,v in c['velocities_m_s']:
        if nid not in node_set or (nid in fixed and any(v)):raise ValueError('CG velocity role invalid')
        _vector(v,'CG velocity')
    _number(c['initial_bending_energy_J'],'initial bending energy')
    if c['initial_bending_energy_J']<0:raise ValueError('CG initial bending energy must be nonnegative')
    if (projectile is not None and projectile not in node_set) or any(n == projectile or n not in node_set for n,_ in c['projectile_pairs']):
        raise ValueError('CG result projectile role mismatch')
    if metadata.get('force_roles') != FORCE_ROLES:
        raise ValueError('CG force role declarations mismatch')
    if _number(metadata.get('final_max_net_nodal_force_N'),'net nodal force norm') < 0:
        raise ValueError('negative net nodal force norm')
    if state is not None and state.get('final',{}).get('step') != c['steps']:
        raise ValueError('CG result time contract mismatch')
    if profile:
        from ._harmonic_profile import shape
        large={'dynamics_state','positions_trajectory','displacements','deformed_positions','edge_forces','edge_stretches','history'}
        json_bytes({k:v for k,v in metadata.items() if k not in large},profile['prepared_bytes'],indent=2)
        if state is not None:json_bytes(state,profile['export_bytes'])
        if arrays:
            for key,dims in [('displacements',(n,3)),('deformed_positions',(n,3)),('edge_forces',(e,)),('edge_stretches',(e,))]:
                if arrays.get(key) is not None:shape(arrays[key],dims,key)
            trajectory=arrays.get('positions_trajectory')
            if trajectory is not None and len(trajectory):shape(trajectory,(c['output_frames'],n,3),'positions_trajectory')
            if len(arrays.get('history',[]))>c['output_frames']:raise ValueError('network history frame bound exceeded')
    return profile


def source_capabilities():
    """Source declarations only; no installed binary or physics claim."""
    return dict(model=MODEL,state_schema=SCHEMA,dimension=3,units='si',
        max_nodes=64,max_edges=128,max_angles=128,max_projectile_pairs=63,max_steps=1000,
        static_run_zero=True,runtime_readiness='unknown',physics_qualification='not_established',
        native_execution='existing LAMMPSTask owned CLI child',
        contact='selected conservative bead pairs; no continuous fiber contact or impenetrability',
        neighbor_policies=[NEIGHBOR_POLICY], pair_roles=_pair_roles_contract(), network_profiles=[network.NAME,network.MIXED_NAME])
