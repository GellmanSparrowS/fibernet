"""Experimental bounded scalar P1 Tet4 steady conduction."""
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from fractions import Fraction
import math
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim import tet4 as core
from fibernet.sim.tet4_loads import _prepare_exterior_face_areas, _iter_exterior_face_areas
from fibernet.sim._tet4_robin import _prepare_robin_terms, _add_robin_operator

__all__ = [
    "Tet4ConvectionConductionResult", "solve_steady_conduction_tet4_with_convection",
    "Tet4MixedConductionResult", "solve_steady_conduction_tet4_with_mixed_loads",
    "Tet4SurfaceConductionResult", "solve_steady_conduction_tet4_with_surface_flux",
    "Tet4ConductionResult", "Tet4SourceConductionResult",
    "solve_steady_conduction_tet4", "solve_steady_conduction_tet4_with_source",
]

_MAX_NODES = 1024
_MAX_CELLS = 8000
_MAX_K = 1e6  # W/(m K); bounded prototype envelope
_MIN_K = 1e-6
_ABS_T_MIN = 1.
_ABS_T_MAX = 2000.
_COORD_MAX_M = 1.
_VOLUME_MIN_M3 = 1e-24
_GRAD_MAX_PER_M = 1e12
_TRANSFER_REL = 2e-10
_RHS_CANCELLATION_REL = 1e-10
_NO_SOURCE = object()
# Centred temperatures can retain variation lost when adding a Kelvin offset.
# The 2e-10 relative gate explicitly limits that representation loss on the
# complete physical field. This is a consistency gate, not a PDE error bound.


@dataclass(frozen=True)
class Tet4ConductionResult:
    """Owned SI fields in original stable-ID order, with per-node component references."""
    node_ids: np.ndarray
    cell_ids: np.ndarray
    temperature_K: np.ndarray
    temperature_difference_K: np.ndarray
    reference_K_by_node: np.ndarray
    prescribed_mask: np.ndarray
    heat_into_solid_W: np.ndarray
    gradient_K_per_m: np.ndarray
    flux_W_per_m2: np.ndarray
    dirichlet_quadratic_W_K: float
    terminal_pairing_W_K: float
    free_residual_pairing_W_K: float
    quadratic_closure_W_K: float
    max_absolute_transfer_scaled_error: float
    free_residual_max_W: float
    normalized_backward: float
    factor_entries: int


@dataclass(frozen=True)
class Tet4SourceConductionResult(Tet4ConductionResult):
    """Source-aware ledger; pairings use the per-component centered field."""

    source_by_node_W: np.ndarray
    integrated_source_W_by_cell: np.ndarray
    total_source_W: float
    source_pairing_W_K: float
    balance_W: float


@dataclass(frozen=True)
class Tet4SurfaceConductionResult(Tet4ConductionResult):
    """Surface heat is separate from the prescribed-node heat response."""

    surface_by_node_W: np.ndarray
    total_surface_W: float
    surface_pairing_W_K: float
    balance_W: float


@dataclass(frozen=True)
class Tet4MixedConductionResult(Tet4ConductionResult):
    """Signed volume, surface and prescribed-node heat ledgers.

    `applied_nodal_heat_W` rounds the combined physical load once, without
    prescribed-temperature elimination. The split ledgers round separately.
    The quadratic closure uses `applied_pairing_W_K` and centered temperatures.
    """
    source_by_node_W: np.ndarray
    integrated_source_W_by_cell: np.ndarray
    total_source_W: float
    source_pairing_W_K: float
    surface_by_node_W: np.ndarray
    total_surface_W: float
    surface_pairing_W_K: float
    applied_nodal_heat_W: np.ndarray
    applied_pairing_W_K: float
    balance_W: float


@dataclass(frozen=True)
class Tet4ConvectionConductionResult(Tet4SourceConductionResult):
    """Convection heat is signed positive into the solid, in watts."""
    convective_heat_into_solid_W_by_node: np.ndarray
    total_convective_heat_into_solid_W: float
    convection_pairing_W_K: float


def _lock(a):
    return np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape)


def _mapped(mapping,keys,label,low,high,complete=False):
    if not isinstance(mapping,Mapping) or len(mapping)>len(keys):
        raise ValueError('invalid bounded '+label+' mapping')
    out={}
    for j,(k,v) in enumerate(mapping.items()):
        k=core._integer(k,label+' stable ID')
        v=core._float64(v,label)
        if j>=len(keys) or k in out or k not in keys or not low<=v<=high:
            raise ValueError('invalid '+label+' ID or value')
        out[k]=v
    if complete and set(out)!=keys:
        raise ValueError('missing '+label)
    return out


def _combine_rhs(boundary, source):
    """Combine bounded binary64 free-row terms; refuse unresolved small differences."""
    effective=boundary+source
    if not np.isfinite(effective).all():
        raise ValueError('nonfinite combined thermal RHS')
    both=(boundary!=0.)&(source!=0.)
    exact=both&(boundary==-source)
    if np.any(exact&(effective!=0.)):
        raise ValueError('exact opposing thermal RHS was not zero')
    if np.any(both&((effective==boundary)|(effective==source))):
        raise ValueError('thermal RHS lost a nonzero operand')
    opposed=both&(np.signbit(boundary)!=np.signbit(source))
    uncertified=abs(effective)<=_RHS_CANCELLATION_REL*np.maximum(abs(boundary),abs(source))
    if np.any(opposed&~exact&uncertified):
        raise ValueError('unresolved near-opposing thermal RHS')
    core._check_normal_or_zero(effective,'combined thermal RHS')
    return effective


def solve_steady_conduction_tet4(
    bundle: Tet4Bundle, *, conductivity_W_per_m_K_by_material: Mapping,
    prescribed_K_by_id: Mapping,
) -> Tet4ConductionResult:
    """Solve a steady source-free Tet4 snapshot with prescribed absolute Kelvin.

    Conductivity is scalar W/(m K) keyed by stable material ID. Prescribe
    at least one node in each conductive component. Unspecified exterior faces
    are weakly insulated. Positive discrete cap heat enters the solid.
    The Dirichlet quadratic and terminal pairing are measured in W K,
    not stored energy. This is not a heat source, transient, or reverse
    thermoelastic coupling API.
    """
    return _solve_conduction(bundle, conductivity_W_per_m_K_by_material,
                             prescribed_K_by_id, _NO_SOURCE)


def solve_steady_conduction_tet4_with_source(
    bundle: Tet4Bundle, *, conductivity_W_per_m_K_by_material: Mapping,
    prescribed_K_by_id: Mapping, source_W_per_m3_by_cell: Mapping,
) -> Tet4SourceConductionResult:
    """As source-free conduction, with signed constant source density per cell.

    Supply one value per stable cell ID, including explicit zero where off.
    Positive values heat and negative values cool. Admitted nonzero
    magnitudes are 1e-8 to 1e8 W/m³, a numerical pilot envelope. The heat
    entering on prescribed nodes plus net generation balances to numerical
    tolerance. Quadratic, terminal, source and residual pairings are W·K
    using each component's centered temperatures; they are not stored
    energy. No state-dependent source, transient, or reverse coupling.
    """
    return _solve_conduction(bundle, conductivity_W_per_m_K_by_material,
                             prescribed_K_by_id, source_W_per_m3_by_cell)


def solve_steady_conduction_tet4_with_surface_flux(
    bundle: Tet4Bundle, *, conductivity_W_per_m_K_by_material: Mapping,
    prescribed_K_by_id: Mapping, faces_by_node_id, inward_flux_W_per_m2,
) -> Tet4SurfaceConductionResult:
    """Private pilot: uniform signed inward flux on explicit exterior triangles.

    All selected face loads, including fully prescribed faces, enter the load
    ledger. The prescribed reaction is the remaining heat; a fully prescribed
    face adds no free-row forcing. Use the mixed-load entry point for volume heating.
    """
    return _solve_conduction(bundle, conductivity_W_per_m_K_by_material,
                             prescribed_K_by_id, _NO_SOURCE,
                             surface=(faces_by_node_id, inward_flux_W_per_m2))


def solve_steady_conduction_tet4_with_mixed_loads(
    bundle: Tet4Bundle, *, conductivity_W_per_m_K_by_material: Mapping,
    prescribed_K_by_id: Mapping, source_W_per_m3_by_cell: Mapping,
    faces_by_node_id, inward_flux_W_per_m2,
) -> Tet4MixedConductionResult:
    """Apply both signed cell heat density and uniform signed exterior heat flux.

    One stable cell ID entry per cell, including zero, plus distinct exterior
    triangles identified by their three stable node IDs. Positive loads enter
    the solid. Separate heat and temperature-pairing ledgers include loads on
    fully prescribed faces. Nonzero local cancellation outside the numerical
    envelope is rejected; net global balance does not imply an isothermal body.
    Geometry and rates are finite-precision inputs; no transient or feedback.
    """
    if source_W_per_m3_by_cell is _NO_SOURCE:
        raise TypeError('expected a complete source density mapping')
    return _solve_conduction(bundle, conductivity_W_per_m_K_by_material,
                             prescribed_K_by_id, source_W_per_m3_by_cell,
                             surface=(faces_by_node_id, inward_flux_W_per_m2))


def solve_steady_conduction_tet4_with_convection(
    bundle: Tet4Bundle, *, conductivity_W_per_m_K_by_material: Mapping,
    prescribed_K_by_id: Mapping, source_W_per_m3_by_cell: Mapping,
    faces_by_node_id, convection_W_per_m2_K, ambient_K,
) -> Tet4ConvectionConductionResult:
    """Solve with constant linear convection and cellwise signed volume heat.

    Each connected component still requires a prescribed-temperature node.
    The uniform coefficient and ambient temperature apply only to explicitly
    selected topological exterior triangles. The remaining unheated exterior
    is insulated. Supply a complete stable-cell-ID map, including zeros.
    """
    if source_W_per_m3_by_cell is _NO_SOURCE:
        raise TypeError('expected a complete source density mapping')
    return _solve_conduction(bundle, conductivity_W_per_m_K_by_material,
                             prescribed_K_by_id, source_W_per_m3_by_cell,
                             convection=(faces_by_node_id,convection_W_per_m2_K,ambient_K))


def _local_mixed_watts(terms):
    """Round once after exact products of the represented load factors."""
    acc, shift = 0, 0
    magnitudes = []
    for geometry, rate, divisor in terms:
        if not rate:
            continue
        rn, rd = (rate if isinstance(rate,Fraction) else float(rate)).as_integer_ratio()
        n = rn * (12 // divisor)
        bits = rd.bit_length()-1
        for factor in (geometry if isinstance(geometry,tuple) else (geometry,)):
            gn, gd = float(factor).as_integer_ratio()
            n *= gn
            bits += gd.bit_length()-1
        if bits > shift:
            acc <<= bits - shift
            shift = bits
        acc += n << (shift - bits)
        magnitudes.append((n,bits))
    den = 12 << shift
    vn,vp=0,0
    subtotals=[0,0]
    smallest=[None,None]
    for n,bits in magnitudes:
        a=abs(n) << (shift-bits); sign=int(n>0)
        if n>0: vp=max(vp,a)
        else: vn=max(vn,a)
        subtotals[sign]+=a
        smallest[sign]=a if smallest[sign] is None else min(smallest[sign],a)
    for subtotal,a in zip(subtotals,smallest):
        if a is not None and subtotal/den == (subtotal-a)/den:
            raise ValueError('mixed nodal heat lost a nonzero same-sign load contribution')
    if not acc:
        return 0.
    r = acc / den
    core._check_normal_or_zero(r,'mixed applied nodal heat in W')
    if vn and vp and abs(acc)*10**10 <= max(vn,vp):
        raise ValueError('unresolved near-opposing mixed nodal heat')
    for sign,a in enumerate(smallest):
        if a is not None and (acc+(-a if sign else a))/den == r:
            raise ValueError('mixed nodal heat lost a nonzero load contribution')
    return r


def _resolved_surface_lump(terms):
    """Every represented face term must affect the rounded nodal value.

    A single signed uniform flux gives same-sign terms. Removing a smallest
    magnitude term gives the closest leave-one-out exact sum, so by monotonic
    rounding it suffices to test that one term. This is a representability
    guard, not a claim that the aggregate or the underlying area is exact.
    """
    total = math.fsum(terms)
    if len(terms) > 1 and total:
        i = min(range(len(terms)), key=lambda j: abs(terms[j]))
        without = math.fsum(t for j,t in enumerate(terms) if j != i)
        if without == total:
            raise ValueError('surface nodal heat lost a nonzero face contribution')
    return total


def _surface_terms(mesh, selection, key, *, factors=None):
    faces, flux = selection
    rate = core._float64(flux, 'inward flux in W/m²')
    core._check_normal_or_zero(rate, 'inward flux in W/m²')
    if rate and not 1e-8 <= abs(rate) <= 1e8:
        raise ValueError('inward flux outside pilot envelope')
    prepared = _prepare_exterior_face_areas(mesh, faces)
    nm, ne = math.frexp(rate)
    groups = [[] for _ in key]
    face_powers, first_rows = [], []
    try:
        for face, area in _iter_exterior_face_areas(prepared):
            am, ae = math.frexp(area)
            lump = math.ldexp(am * nm / 3., ae + ne)
            total = math.ldexp(am * nm, ae + ne)
            core._check_normal_or_zero((lump,total), 'integrated surface heat in W')
            if rate and (not lump or not total or not math.isfinite(total)):
                raise ValueError('unresolved integrated surface heat')
            first_rows.append(key[face[0]])
            face_powers.append(total)
            for node in face:
                groups[key[node]].append(lump)
                if factors is not None and rate:
                    factors[key[node]].append((area,rate,3))
    except OverflowError as exc:
        raise ValueError('surface area or heat outside supported binary64 range') from exc
    return (np.array([_resolved_surface_lump(g) for g in groups]), np.array(face_powers),
            np.asarray(first_rows,dtype=np.intp), bool(rate))


def _solve_conduction(bundle, conductivity_W_per_m_K_by_material,
                      prescribed_K_by_id, source_W_per_m3_by_cell, *, surface=None, convection=None):
    if not isinstance(bundle, Tet4Bundle):
        raise TypeError('expected a Tet4Bundle')
    if core._row_count(bundle.points,'points')>_MAX_NODES or core._row_count(bundle.tetrahedra,'tetrahedra')>_MAX_CELLS:
        raise ValueError('scalar conduction limited to 1024 nodes and 8000 Tet4')
    mesh=core._validate_static_mesh(bundle)
    x=mesh.points*core._LENGTH_TO_M[mesh.length_unit]; t=mesh.tetrahedra
    n=len(x)
    if np.max(abs(x),initial=0)>_COORD_MAX_M:
        raise ValueError('conduction coordinate limit')
    key={int(k):i for i,k in enumerate(mesh.node_ids)}
    kk=_mapped(conductivity_W_per_m_K_by_material,set(mesh.materials),'conductivity',_MIN_K,_MAX_K,True)
    temp=_mapped(prescribed_K_by_id,set(key),'temperature',_ABS_T_MIN,_ABS_T_MAX)
    if not temp: raise ValueError('each conductive component requires a prescribed node')
    fixed=np.zeros(n,dtype=np.bool_); known=np.zeros(n)
    for k,vt in temp.items():j=key[k];fixed[j]=True;known[j]=vt
    graph=core._free_node_topology(t,n,np.arange(n))
    _,components=connected_components(graph,directed=False)
    if set(components[fixed])!=set(components):
        raise ValueError('each conductive component requires a prescribed node')
    ref=np.empty(n)
    for lab in np.unique(components):
        ix=components==lab
        ref[ix]=np.min(known[ix&fixed])
    theta=np.zeros(n);theta[fixed]=known[fixed]-ref[fixed]
    ge=[]; V=[]; axes=[]
    for i,c in enumerate(t):
        g, v = core._shape_gradients_and_volume_m(x, c, i)
        if not (_VOLUME_MIN_M3<=v<=1) or not np.isfinite(g).all() or np.max(abs(g))>_GRAD_MAX_PER_M:
            raise ValueError('cell outside conduction volume/gradient envelope')
        ge.append(g); V.append(v)
        axes.append(kk[int(mesh.material_ids[i])])
    ge=np.array(ge);V=np.array(V);k=np.array(axes)
    aware = source_W_per_m3_by_cell is not _NO_SOURCE
    active = False
    mixed = aware and surface is not None
    use_robin = convection is not None
    factors = ([[] for _ in range(n)] if mixed or use_robin else None)
    if aware:
        table=_mapped(source_W_per_m3_by_cell,set(map(int,mesh.cell_ids)),'source density',-1e8,1e8,True)
        density=np.array([table[int(cid)] for cid in mesh.cell_ids])
        if np.any((density!=0.)&(abs(density)<1e-8)):
            raise ValueError('nonzero source density below 1e-8 W/m³')
        generated=V*density
        groups=[[] for _ in range(n)]
        for c,cellw in zip(t,generated):
            for i in c: groups[i].append(cellw*.25)
        source=np.array([math.fsum(g) for g in groups])
        gross_source=np.array([math.fsum(map(abs,g)) for g in groups])
        active = bool(np.any(density))
        if mixed:
            volume_integrated=generated
            volume_gross=gross_source
        if factors is not None:
            for row,v,r in zip(t,V,density):
                if r:
                    for i in row:
                        factors[i].append((v,r,4))
    if surface is not None:
        surface_source, face_integrated, face_rows, face_active = _surface_terms(
            mesh, surface, key, factors=factors)
        if mixed:
            active=active or face_active
            source=np.array([_local_mixed_watts(group) for group in factors])
            volume_source=np.array([_local_mixed_watts(t for t in group if t[2]==4)
                                    for group in factors])
            surface_source=np.array([_local_mixed_watts(t for t in group if t[2]==3)
                                     for group in factors])
            gross_source=volume_gross+abs(surface_source)
        else:
            source,generated,active=surface_source,face_integrated,face_active
            gross_source=abs(source)
    local=(k*V)[:,None,None]*(ge@ge.transpose(0,2,1))
    rows=np.broadcast_to(t[:,:,None],local.shape); cols=np.broadcast_to(t[:,None,:],local.shape)
    A=sparse.coo_matrix((local.ravel(),(rows.ravel(),cols.ravel())),shape=(n,n)).tocsr()
    A=.5*A+.5*A.T;A.eliminate_zeros()
    if not np.isfinite(A.data).all():raise ValueError('nonfinite conduction matrix')
    K=A
    robin=None
    fixed_offsets=None
    if use_robin:
        robin=_prepare_robin_terms(mesh,key,*convection)
        if len(robin.rows):
            A=_add_robin_operator(K,robin.matrix,fixed)
            fixed_offsets={i:Fraction.from_float(float(known[i]))-Fraction.from_float(float(ref[i]))
                           for i in np.flatnonzero(fixed)}
    ids=np.flatnonzero(~fixed);cap=np.flatnonzero(fixed)
    tol=64*np.finfo(np.float64).eps*max(1,len(ids))
    backward=0.; entries=0
    if len(ids):
        graph=core._free_node_topology(t,n,ids)
        order=core._canonical_node_order(graph,x[ids],mesh.node_ids[ids])
        ranks=np.empty_like(order);ranks[order]=np.arange(len(order))
        i,j=graph.nonzero()
        h=int(np.max(abs(ranks[i]-ranks[j]),initial=0))
        entries=(2*len(ids)-h)*(h+1);pad=len(ids)*(h+1)
        mat=A[ids][:,ids]; bound=float(abs(mat).sum(axis=1).max())
        if not np.isfinite(bound) or bound<=0:raise ValueError('unresolved scalar operator scale')
        boundary=-np.asarray(K[ids][:,cap]@theta[cap])
        if fixed_offsets is not None:
            robin_terms=robin.represented_drive_terms(theta,ref,(fixed,known))
            cross=K[ids][:,cap].tocsr()
            conduct=[]
            for start,end in zip(cross.indptr[:-1],cross.indptr[1:]):
                conduct.append(-sum(Fraction.from_float(float(k))*fixed_offsets[cap[j]]
                                    for k,j in zip(cross.data[start:end],cross.indices[start:end])))
            rhs_raw=np.array([
                _local_mixed_watts((*factors[i],(1.,b,1),*robin_terms[i]))
                for i,b in zip(ids,conduct)
            ])
            rhs=rhs_raw/bound
            if np.any((rhs_raw!=0.)&(rhs==0.)):
                raise ValueError('convection forcing lost during normalization')
            core._check_normal_or_zero(rhs,'normalized convection RHS')
        elif mixed and active:
            rhs=np.array([
                _local_mixed_watts((*factors[i],(1.,b,1))) if b and source[i]
                else (b if b else source[i])
                for i,b in zip(ids,boundary)
            ])/bound
        elif active and np.any(source[ids]):
            rhs=_combine_rhs(boundary,source[ids])/bound
        else:
            rhs=boundary/bound
        m=mat/bound
        if robin is not None and len(robin.rows):
            if np.any((mat.data!=0.)&(m.data==0.)):
                raise ValueError('convection operator coefficient lost during normalization')
            core._check_normal_or_zero(m.data,'normalized convection matrix')
        plan=core._BandPlan(ids,order,h,entries,pad)
        theta[ids]=core._factor_band(m,tol,plan)(rhs)
        den=float(abs(m).sum(axis=1).max())*float(np.max(abs(theta[ids])))+float(np.max(abs(rhs)))
        defect=float(np.max(abs(m@theta[ids]-rhs)))
        backward=defect/den if den>0 else defect
        if not 0<=backward<=tol:raise ValueError('scalar backward residual rejected')
    T=ref+theta;T[fixed]=known[fixed]  # publish input absolute K exactly
    if not np.all((_ABS_T_MIN<=T)&(T<=_ABS_T_MAX)):
        raise ValueError('absolute conduction result outside [1,2000] K; no clipping')
    recon=T-ref
    max_transfer=0.
    for lab in np.unique(components):
        ix=components==lab
        span=float(np.ptp(theta[ix]))
        err=float(np.max(abs(recon[ix]-theta[ix])))
        if span==0.:
            if err!=0.:raise ValueError('unresolved constant absolute temperature reconstruction')
        elif err>_TRANSFER_REL*span:
            raise ValueError('unresolved absolute-to-centered temperature transfer')
        if span>0.:max_transfer=max(max_transfer,err/span)
    # Rebuild fluxes from differences, independently of A@theta, avoiding constant modes.
    deltas=theta[t[:,1:]]-theta[t[:,:1]]
    if fixed_offsets is not None:
        represented=[fixed_offsets[i] if fixed[i] else Fraction.from_float(float(x))
                     for i,x in enumerate(theta)]
        deltas=np.array([[float(represented[i]-represented[row[0]]) for i in row[1:]] for row in t])
    grad=np.einsum('ni,nik->nk',deltas,ge[:,1:,:])
    flux=-k[:,None]*grad
    el=-V[:,None]*np.einsum('nki,ni->nk',ge,flux)
    power=np.zeros(n)
    np.add.at(power,t,el)
    has_forcing=active or (robin is not None and len(robin.rows)>0)
    conv_by_node=np.zeros(n)
    conv_by_face=np.zeros(0)
    if robin is not None and len(robin.rows):
        exact_out_nodes,exact_out_faces=robin.outward_exact(theta,ref,(fixed,known))
        conv_by_node=np.array([-float(v) for v in exact_out_nodes])
        conv_by_face=np.array([-float(v) for v in exact_out_faces])
        core._check_normal_or_zero(conv_by_node,'nodal convection heat in W')
        core._check_normal_or_zero(conv_by_face,'integrated face convection heat in W')
        physical=np.array([math.fsum((a,b)) for a,b in zip(source,conv_by_node)])
        gross_source=gross_source+abs(conv_by_node)
    elif has_forcing:
        physical=source
    else:
        physical=np.zeros(n)
    if robin is not None and len(robin.rows):
        # Do not let an exact represented residual become zero merely
        # because two separately rounded public load ledgers oppose.
        def balance_row(p, c, group):
            v=sum(Fraction.from_float(float(a))*Fraction.from_float(float(r))/k
                  for a,r,k in group)
            return float(Fraction.from_float(float(p))+c-v)
        residual=np.array([balance_row(p,c,row) for p,c,row in zip(power,exact_out_nodes,factors)])
    else:
        residual=power-physical if has_forcing else power
    heat=np.where(fixed,residual,0.)
    qcell=(k*V)*np.einsum('ni,ni->n',grad,grad)  # K W, not heat flow/power
    if np.any(~np.isfinite(qcell)) or np.any(qcell<0.):
        raise ValueError('invalid conductive quadratic')
    q=math.fsum(qcell)
    work=math.fsum(heat*theta)
    free_work=math.fsum(residual[~fixed]*theta[~fixed])
    source_work=math.fsum(source*theta) if active else 0.
    conv_work=math.fsum(conv_by_node*theta)
    effective_work=math.fsum((source_work,conv_work)) if use_robin else source_work
    audited_A=K@theta
    gross_local=(abs(local)*abs(theta[t])[:,None,:]).sum(axis=(1,2))
    labels=components[t[:,0]]
    for lab in np.unique(components):
        ix=components==lab;cells=labels==lab; free=ix&(~fixed)
        flo=256*np.finfo(np.float64).eps*math.fsum(gross_local[cells])
        if has_forcing:
            eps_W=1e-9*(math.fsum(abs(el[cells]).ravel())+math.fsum(gross_source[ix]))+flo
            if max(float(abs(residual[free]).max(initial=0)),abs(math.fsum(power[ix])),
                   float(abs(power[ix]-audited_A[ix]).max(initial=0)))>eps_W:
                raise ValueError('conduction flux or free residual rejected')
            # This is the PUBLIC cap+generation conservation statement, not
            # merely the sum of every algebraic row. Free residuals are measured
            # separately here and may account only for their own net signed heat.
            cap_heat=math.fsum(heat[ix])
            if mixed:
                integrated=np.concatenate((volume_integrated[cells],
                                           face_integrated[ix[face_rows]]))
            elif surface is None:
                integrated=generated[cells]
            else:
                integrated=generated[ix[face_rows]]
            if robin is not None and len(robin.rows):
                face_values=conv_by_face[ix[robin.rows[:,0]]]
                scale=math.fsum(abs(face_values))+math.fsum(abs(conv_by_node[ix]))
                gap=math.fsum((*conv_by_node[ix],*(-face_values)))
                if abs(gap)>128*np.finfo(np.float64).eps*scale:
                    raise ValueError('component convection nodal and face heat disagree')
                integrated=np.concatenate((integrated,face_values))
            gen_heat=math.fsum(integrated)
            free_net=math.fsum(residual[free])
            direct=math.fsum((cap_heat,gen_heat))
            roundoff_W=flo+256*np.finfo(np.float64).eps*(
                math.fsum(abs(el[cells]).ravel())+math.fsum(gross_source[ix])+
                math.fsum(abs(integrated))+math.fsum(abs(residual[ix])))
            if (abs(math.fsum((direct,free_net)))>roundoff_W or
                abs(direct)>abs(free_net)+roundoff_W):
                raise ValueError('direct prescribed heat and generation do not balance' if surface is None
                                 else 'direct prescribed heat and applied surface heat do not balance')
            U=math.fsum(qcell[cells]); W=math.fsum(heat[ix]*theta[ix]); F=math.fsum(residual[free]*theta[free]); S=math.fsum(physical[ix]*theta[ix])
            if use_robin:
                S=math.fsum((*tuple(source[ix]*theta[ix]),*tuple(conv_by_node[ix]*theta[ix])))
            if mixed:
                for loads,active_integrated in (
                    (volume_source[ix],volume_integrated[cells]),
                    (surface_source[ix],face_integrated[ix[face_rows]]),
                ):
                    gross=math.fsum(map(abs,active_integrated))
                    gap=math.fsum((*loads,*(-active_integrated)))
                    if not math.isfinite(gap) or abs(gap)>128*np.finfo(np.float64).eps*gross:
                        raise ValueError('mixed nodal and integrated heat ledgers do not match')
                wv=volume_source[ix]*theta[ix]
                we=surface_source[ix]*theta[ix]
                scale=math.fsum(map(abs,wv))+math.fsum(map(abs,we))
                if abs(math.fsum((math.fsum(wv),math.fsum(we),-S)))>128*np.finfo(np.float64).eps*scale:
                    raise ValueError('mixed thermal ledger pairings do not match applied heat')
            eps_WK=1e-9*(math.fsum(abs(residual[ix]*theta[ix]))+(math.fsum(abs(source[ix]*theta[ix]))+math.fsum(abs(conv_by_node[ix]*theta[ix]))
                        if use_robin else math.fsum(abs(physical[ix]*theta[ix])))+abs(U))+flo*float(np.max(abs(theta[ix])))
            if not np.isfinite([U,W,F,S]).all() or abs(math.fsum((U,-W,-F,-S)))>eps_WK:
                raise ValueError('conduction Dirichlet quadratic failed')
            # Independently bound the *direct* terminal pairing against the
            # geometric scalar quadratic by measured free-row error and roundoff.
            residual_virtual_bound=math.fsum(abs(residual[free]*theta[free]))
            if abs(math.fsum((U,-W,-S)))>residual_virtual_bound+eps_WK:
                raise ValueError('conduction terminal pairing unresolved')
            continue
        eps_W=1e-9*math.fsum(abs(el[cells]).ravel())+flo
        if max(float(abs(power[free]).max(initial=0)),abs(math.fsum(power[ix])),
               float(abs(power[ix]-audited_A[ix]).max(initial=0)))>eps_W:
            raise ValueError('conduction flux or free residual rejected')
        U=math.fsum(qcell[cells]); W=math.fsum(heat[ix]*theta[ix]); F=math.fsum(power[free]*theta[free])
        eps_WK=1e-9*(math.fsum(abs(power[ix]*theta[ix]))+abs(U))+flo*float(np.max(abs(theta[ix])))
        if not np.isfinite([U,W,F]).all() or abs(math.fsum((U,-W,-F)))>eps_WK:
            raise ValueError('conduction Dirichlet quadratic failed')
        # Independently bound the *direct* terminal pairing against the
        # geometric scalar quadratic by measured free-row error and roundoff.
        residual_virtual_bound=math.fsum(abs(power[free]*theta[free]))
        if abs(math.fsum((U,-W)))>residual_virtual_bound+eps_WK:
            raise ValueError('conduction terminal pairing unresolved')
    fields=(*map(_lock,(mesh.node_ids,mesh.cell_ids,T,theta,ref)),
            *map(_lock,(fixed,heat,grad,flux)))
    closure=math.fsum((q,-work,-free_work,-effective_work)) if has_forcing else math.fsum((q,-work,-free_work))
    shared=(*fields,q,work,free_work,closure,max_transfer,
            float(np.max(abs(residual[~fixed]),initial=0)),backward,entries)
    if use_robin:
        heat_balance=math.fsum((*heat,*generated,*conv_by_face))
        return Tet4ConvectionConductionResult(
            *shared,_lock(source),_lock(generated),math.fsum(generated),source_work,heat_balance,
            _lock(conv_by_node),math.fsum(conv_by_face),conv_work,
        )
    if mixed:
        V=math.fsum(volume_integrated)
        E=math.fsum(face_integrated)
        Wv=math.fsum(volume_source*theta)
        We=math.fsum(surface_source*theta)
        return Tet4MixedConductionResult(
            *shared,_lock(volume_source),_lock(volume_integrated),V,Wv,
            _lock(surface_source),E,We,_lock(source),source_work,
            math.fsum((*heat,*volume_integrated,*face_integrated)),
        )
    if surface is not None:
        return Tet4SurfaceConductionResult(
            *shared,_lock(source),math.fsum(generated),source_work,
            math.fsum((*heat,*generated)),
        )
    if not aware:
        return Tet4ConductionResult(*shared)
    return Tet4SourceConductionResult(
        *shared,_lock(source),_lock(generated),math.fsum(generated),source_work,
        math.fsum((*heat,*generated)),
    )
