# SPDX-License-Identifier: MIT
"""Bounded, solver-neutral independent Tet4 reaction and energy recovery."""
import math

import numpy as np

from . import tet4 as core


# Bounded engineering range for batched cell recovery.
# Outside it, the reference path preserves exact safeguards.
def batched_response(mesh, u, *, guarded=False):
    def fallback():
        return None if guarded else core._prescribed_element_response(mesh, u)

    if np.max(np.abs(u),initial=0.) > .05 or np.any((u!=0.) & (np.abs(u)<1e-18)):
        return fallback()
    if any(not (1e3<=m['young_modulus_pa']<=1e13 and -.95<=m['poisson_ratio']<=.495)
           for m in mesh.materials.values()):
        return fallback()
    x=mesh.points*core._LENGTH_TO_M[mesh.length_unit]
    if np.max(np.abs(x),initial=0.)>10.:
        return fallback()
    force=np.zeros(u.shape)
    energies=[]
    keys,idx=np.unique(mesh.material_ids,return_inverse=True)
    E=np.array([mesh.materials[int(k)]['young_modulus_pa'] for k in keys])
    nu=np.array([mesh.materials[int(k)]['poisson_ratio'] for k in keys])
    mu=E/(2*(1+nu))
    bulk=E/(3*(1-2*nu))
    eye=np.eye(3)
    with np.errstate(divide='raise',over='raise',invalid='raise',under='raise'):
        for start in range(0,len(mesh.tetrahedra),4096):
            sl=slice(start,start+4096); t=mesh.tetrahedra[sl]
            offsets=(x[t[:,1:]]-x[t[:,:1]]).transpose(0,2,1)
            scale=np.max(np.abs(offsets),axis=(1,2))
            if np.any(scale<1e-12):
                return fallback()
            q=offsets/scale[:,None,None]
            condition = np.linalg.cond(q)
            if np.any(condition > core._MAX_ELEMENT_CONDITION):
                raise ValueError('element is too ill-conditioned for this solver')
            if guarded and np.any(condition > 1e4):
                return fallback()
            vol=np.abs(np.linalg.det(q))*scale**3/6.
            if not np.all(np.isfinite(vol)&(vol>0.)):
                raise ValueError('invalid volume in independent reaction and energy recovery')
            gr=np.linalg.inv(q)/scale[:,None,None]
            grad=np.concatenate([-gr.sum(axis=1,keepdims=True),gr],axis=1)
            # Difference from the anchor node removes exact rigid translation.
            du=u[t[:,1:]]-u[t[:,:1]]
            derivative=np.einsum('eni,enj->eij',du,gr)
            strain=0.5*derivative+0.5*derivative.transpose(0,2,1)
            if guarded:
                # A small symmetric strain can be the difference of large
                # local terms (near rigid rotation). Keep scalar recovery
                # when rounding in those terms could mask the strain.
                signal = np.max(np.abs(strain), axis=(1, 2))
                scale = np.max(np.einsum('eni,enj->eij', np.abs(du), np.abs(gr)),
                               axis=(1, 2))
                if np.any(scale > 1e3 * signal):
                    return fallback()
            trace=np.trace(strain,axis1=1,axis2=2)
            dev=strain-trace[:,None,None]*eye/3.
            shear=mu[idx[sl]]
            comp=bulk[idx[sl]]
            stress=2*shear[:,None,None]*dev+(comp*trace)[:,None,None]*eye
            local=vol[:,None,None]*np.einsum('eij,enj->eni',stress,grad)
            cell_e=vol*(shear*np.einsum('eij,eij->e',dev,dev)+0.5*comp*trace*trace)
            if not np.isfinite(local).all() or not np.all(np.isfinite(cell_e)&(cell_e>=0.)):
                raise ValueError('nonfinite or negative independent cell response')
            if np.any(np.any(strain!=0.,axis=(1,2)) & (cell_e<=0.)):
                return fallback()
            np.add.at(force,t,local)
            energies.append(math.fsum(cell_e))
    core._check_normal_or_zero(force,'independent batched mechanical forces')
    if not np.isfinite(force).all():
        raise ValueError('nonfinite independent mechanical forces')
    return force.ravel(),math.fsum(energies)


def guarded_banded_response(mesh, u):
    """Attempt fast recovery, or return None to select the reference path."""
    if len(mesh.tetrahedra) < 64:
        return None
    return batched_response(mesh, u, guarded=True)
