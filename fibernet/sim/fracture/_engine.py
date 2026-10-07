"""Frozen linear-static stress-removal equations and near-tie event rule."""
from __future__ import annotations
import math
from . import _sparse as base
MIB = 1024**2

def require(ok,message):
    if not ok:raise ValueError(message)

def same(a,b,message):require(base.canonical(a)==base.canonical(b),message)

def topology(g,mask):
    require(type(mask) is list and len(mask)==len(g['original_edges']) and all(type(v) is bool for v in mask),'Typed original-edge active mask')
    require(not any(a and not b for a,b in zip(mask,g['initial_mask'])),'Notch cannot resurrect')
    edges=[e for e,a in zip(g['original_edges'],mask) if a];groups,isolates=base.components(g['order'],edges)
    fixed=set(g['fixed']);loaded=set(g['loaded']);free=[x for x in groups if not set(x)&(fixed|loaded)]
    degree={i:0 for i in g['order']}
    for _,a,b in edges:degree[a]+=1;degree[b]+=1
    return dict(edges=edges,finite_components=groups,isolates=isolates,free_finite_components=free,
        path_present=any(set(x)&fixed and set(x)&loaded for x in groups),
        fixed_incident_rods={str(i):degree[i] for i in g['fixed']},loaded_incident_rods={str(i):degree[i] for i in g['loaded']},
        fixed_attachment_lost_ids=[i for i in g['fixed'] if not degree[i]],loaded_attachment_lost_ids=[i for i in g['loaded'] if not degree[i]],
        both_grips_retain_finite_material=bool(any(degree[i] for i in fixed) and any(degree[i] for i in loaded)))

def stress_envelope(local_end,parameters):
    require(type(local_end) is list and len(local_end)==12,'Actual twelve end resultants')
    radius=base.number(parameters['diameter_m'],'diameter',positive=True)/2
    sigma=base.number(parameters['sigma_Pa'],'sigma',positive=True);tau=base.number(parameters['tau_Pa'],'tau',positive=True)
    v=[base.number(x,'local resultant') for x in local_end];A=math.pi*radius**2;I=math.pi*radius**4/4;J=2*I
    values=[]
    for off,sign in ((0,-1),(6,1)):
        N=sign*v[off];V=math.hypot(v[off+1],v[off+2]);T=abs(v[off+3]);M=math.hypot(v[off+4],v[off+5])
        normal=abs(N)/A+radius*M/I;shear=4*V/(3*A)+T*radius/J
        values.append(dict(N_N=N,V_N=V,T_Nm=T,M_Nm=M,normal_envelope_Pa=normal,shear_envelope_Pa=shear,
            criterion=math.hypot(normal/sigma,shear/tau)))
    return dict(ends=values,criterion=max(v['criterion'] for v in values),
        scope='Conservative non-coincident maxima stress envelope; declared brittle strengths, not calibrated pointwise PLA')

def equilibrium(parameters,g,mask,displacement,guard=None):
    displacement=base.number(displacement,'nonnegative imposed displacement')
    require(0<=displacement<=parameters['maximum_displacement_m'],'Declared loading extent')
    if guard:guard.check(512*MIB)
    np,_,_,ref=base.numeric_modules();t=topology(g,mask);fixed=set(g['fixed']);loaded=set(g['loaded'])
    undefined={i for group in t['free_finite_components'] for i in group}|set(t['isolates'])
    attached=[e for e in t['edges'] if e[1] not in undefined];index={i:k for k,i in enumerate(g['order'])}
    rigid={i:0. for i in g['order']}
    for group in t['finite_components']:
        if set(group)&loaded and not set(group)&fixed:
            for i in group:rigid[i]=displacement
    network={'nodes':[{'id':i,'position':g['positions'][i]} for i in g['order']],
             'segments':[{'id':s,'node_i':a,'node_j':b} for s,a,b in attached]}
    length=parameters['characteristic_length_m']
    if attached:
        K,order,elements,forecast=base.assemble_sparse(network,diameter_m=parameters['diameter_m'],young_modulus_Pa=parameters['young_modulus_Pa'],
            poisson_ratio=parameters['poisson_ratio'],model='timoshenko',kappa=parameters['shear_coefficient'],guard=guard,profile_name=base.LARGE_PROFILE)
        prescribed={6*index[i]+d:((displacement-rigid[i]) if i in loaded and d==g['axis'] else 0.) for i in fixed|loaded for d in range(6)}
        omitted=[6*index[i]+d for i in undefined if i not in fixed|loaded for d in range(6)]
        u,reaction,_,metrics=base.solve_sparse(K,prescribed,omitted_dofs=omitted,guard=guard,
            characteristic_length_m=length,profile_name=base.LARGE_PROFILE,factorization_forecast=forecast)
        energy=.5*float(u@(K@u));work=.5*sum(float(reaction[i])*x for i,x in prescribed.items())
        require(abs(energy-work)<=1e-8*max(abs(energy),abs(work),1e-30),'Unchanged half-work1e-8 gate')
        members=[]
        for e in elements:
            a,b=e['node_ids'];_,R=ref.triad(g['positions'][a],g['positions'][b]);glob=e['stiffness']@u[e['dofs']]
            local=np.concatenate([R@glob[k:k+3] for k in (0,3,6,9)]).tolist();criterion=stress_envelope(local,parameters)
            members.append(dict(id=e['id'],local_end_resultants=local,**criterion))
        rows=reaction.reshape((-1,6));force=np.sum(rows[:,:3],axis=0)
        moment=sum((row[3:]+np.cross(g['positions'][i],row[:3]) for i,row in zip(g['order'],rows)),start=np.zeros(3))
        fscale=max(sum(float(np.linalg.norm(row[:3])) for row in rows),1e-30)
        mscale=max(length*fscale,sum(float(np.linalg.norm(row[3:])) for row in rows),1e-30)
        metrics['relative_global_force_balance']=float(np.linalg.norm(force))/fscale
        metrics['relative_global_moment_balance']=float(np.linalg.norm(moment))/mscale
        require(max(metrics['relative_global_force_balance'],metrics['relative_global_moment_balance'])<=1e-8,'Unchanged global wrench1e-8 gate')
    else:
        u=np.zeros(6*len(g['order']));reaction=u.copy();members=[];energy=work=0.;metrics={'relative_free_residual':0.,'scope':'No attached finite elements; not finite grip survival'}
    poses={};rotations={}
    for i in g['order']:
        if i in undefined and i not in fixed|loaded:poses[str(i)]=None;rotations[str(i)]=None;continue
        shift=[float(v) for v in u[6*index[i]:6*index[i]+3]];shift[g['axis']]+=rigid[i]
        if i in loaded and i in undefined:shift[g['axis']]=displacement
        poses[str(i)]=[g['positions'][i][j]+shift[j] for j in range(3)];rotations[str(i)]=u[6*index[i]+3:6*index[i]+6].tolist()
    return dict(displacement_m=displacement,positions_m=poses,rotations_rad=rotations,intact_segment_ids=[e[0] for e in t['edges']],
        mask=list(mask),topology=t,members=members,solver=metrics,strain_energy_J=energy,half_constraint_work_J=work,
        loaded_reaction_force_N=math.fsum(float(reaction[6*index[i]+g['axis']]) for i in loaded),
        nodal_reactions={str(i):None if i in undefined and i not in fixed|loaded else reaction[6*index[i]:6*index[i]+6].tolist() for i in g['order']},
        undefined_member_ids=[sid for sid,a,_ in t['edges'] if a in undefined],
        pose_scope='Linear small-displacement static model; free finite absolute pose undefined',
        finite_material_poses_defined=not t['free_finite_components'],finite_kinematics_verified=False,
        all_fragments_separation_verified=False,PLA_calibrated=False)

def advance(parameters,g,mask,current,guard=None,solve=equilibrium,observe=None):
    current=base.number(current,'current loading');require(0<=current<=parameters['maximum_displacement_m'],'Loading horizon')
    active=list(mask);events=[];reference=parameters['reference_displacement_m'];unit=solve(parameters,g,active,reference,guard)
    maximum=max((e['criterion'] for e in unit['members']),default=0.)
    if maximum==0:
        state=solve(parameters,g,active,current,guard)
        if observe:observe('equilibrated_post',state,[],None)
        return state,events,'zero_stress_or_no_load_path'
    threshold=reference/maximum;delta=max(current,threshold*(1+1e-12));delta=min(delta,parameters['maximum_displacement_m'])
    for cascade in range(parameters['max_avalanche_solves']):
        state=solve(parameters,g,active,delta,guard);failed=[e for e in state['members'] if e['criterion']>=1.-parameters['numerics']['event_tol']]
        if observe and (cascade>0 or not failed):observe('equilibrated_post',state,failed,cascade)
        if not failed:return state,events,'equilibrated' if events else 'loading_bound_or_roundoff_step'
        if observe:observe('event_pre',state,failed,cascade)
        for e in failed:
            sid=e['id'];slot=next(i for i,x in enumerate(g['original_edges']) if x[0]==sid);require(active[slot],'Irreversible genuine active stress selection')
            active[slot]=False;events.append(dict(segment_id=sid,displacement_m=delta,cascade=cascade,criterion=e['criterion'],local_end_resultants=e['local_end_resultants']))
    raise RuntimeError('Same-displacement avalanche cap; incomplete epoch not published')
