"""Circular Timoshenko operator extracted without mathematical changes.

See PROVENANCE.json and NOTICE.md. No LAMMPS/OpenSees code is included.
"""
from __future__ import annotations
import math
import numpy as np

def positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('Positive finite ' + label + ' required')
    return float(value)

def circular_properties(radius, young_modulus, poisson_ratio):
    radius, e = positive(radius, 'radius'), positive(young_modulus, 'Young modulus')
    if (isinstance(poisson_ratio, bool) or not isinstance(poisson_ratio, (int, float)) or
        not math.isfinite(poisson_ratio) or not -1. < poisson_ratio < .5):
        raise ValueError('Isotropic Poisson ratio must be finite in(-1,.5)')
    area, inertia = math.pi*radius**2, math.pi*radius**4/4.
    g = e/(2.*(1.+poisson_ratio))
    if not all(math.isfinite(x) and x > 0 for x in (area, inertia, g)):
        raise ValueError('Finite positive circular section properties required')
    return {'A': area, 'Iy': inertia, 'Iz': inertia, 'J': 2.*inertia, 'E': e, 'G': g}

def local_stiffness(length, section, *, model='timoshenko', kappa=1.):
    """DOF order [u,v,w,rx,ry,rz] at i, then j; units N,m,radians.

    rz gives positive v'=rz; ry gives w'=-ry. Hence the two bending
    coupling signs differ. Avy=Avz=kappa*A for a circular section.
    """
    length = positive(length, 'geometric chord length')
    if model not in ('timoshenko', 'euler_bernoulli'):
        raise ValueError('Unsupported linear beam model')
    kappa = positive(kappa, 'shear coefficient')
    e, g, area, iy, iz, torsion = (positive(section[k], k) for k in ('E', 'G', 'A', 'Iy', 'Iz', 'J'))
    matrix = np.zeros((12, 12))
    for dofs, stiffness in (((0, 6), e*area/length), ((3, 9), g*torsion/length)):
        matrix[np.ix_(dofs, dofs)] = stiffness*np.array([[1., -1.], [-1., 1.]])
    for dofs, inertia, rotation_sign in (((1, 5, 7, 11), iz, 1.), ((2, 4, 8, 10), iy, -1.)):
        phi = 12.*e*inertia/(kappa*g*area*length**2) if model == 'timoshenko' else 0.
        l = length
        # Exact static closed-form bending/shear block, not linear displacement
        # interpolation with full integration (which would cause shear locking).
        block = np.array([[12., 6.*l, -12., 6.*l],
            [6.*l, (4.+phi)*l*l, -6.*l, (2.-phi)*l*l],
            [-12., -6.*l, 12., -6.*l],
            [6.*l, (2.-phi)*l*l, -6.*l, (4.+phi)*l*l]])
        signs = np.array([1., rotation_sign, 1., rotation_sign])
        matrix[np.ix_(dofs, dofs)] = e*inertia/(l**3*(1.+phi))*block*signs[:, None]*signs[None, :]
    if not np.isfinite(matrix).all():
        raise ValueError('Nonfinite local stiffness')
    return matrix

def triad(start, end, roll_seed=None):
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    if start.shape != (3,) or end.shape != (3,) or not np.isfinite(start).all() or not np.isfinite(end).all():
        raise ValueError('Finite spatial endpoints required')
    delta = end-start
    length = positive(float(np.linalg.norm(delta)), 'geometric chord length')
    x = delta/length
    seed = np.eye(3)[np.argmin(np.abs(x))] if roll_seed is None else np.asarray(roll_seed, dtype=float)
    if seed.shape != (3,) or not np.isfinite(seed).all():
        raise ValueError('Finite three-vector roll seed required')
    y = np.cross(seed, x)
    ynorm = float(np.linalg.norm(y))
    if ynorm <= 1e-12*max(1., float(np.linalg.norm(seed))):
        raise ValueError('Roll seed parallel to beam axis')
    y /= ynorm
    z = np.cross(x, y)
    return length, np.stack((x, y, z))

def element_stiffness(start, end, section, *, model='timoshenko', kappa=1., roll_seed=None):
    length, rotation = triad(start, end, roll_seed)
    transform = np.kron(np.eye(4), rotation)
    local = local_stiffness(length, section, model=model, kappa=kappa)
    return transform.T@local@transform
