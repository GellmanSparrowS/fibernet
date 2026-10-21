"""Passive linear viscoelastic material, in Pa and seconds.

This constitutive law is distinct from numerical damping, fluid viscosity and
adhesion. A trial update is pure: callers own and commit the returned state.
"""
from dataclasses import dataclass
import math
import numpy as np


def _number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f'{name} must be a real number')
    result = float(value)
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
    return result


@dataclass(frozen=True)
class LinearViscoelasticMaterial:
    """Equilibrium spring, Maxwell branches and optional parallel dashpot.

    equilibrium_modulus is Pa; branches contain (modulus Pa, time s);
    parallel_viscosity is Pa s. Named constructors describe the same family.
    The material does not imply any empirical calibration or temperature law.
    """
    equilibrium_modulus: float
    branches: tuple = ()
    parallel_viscosity: float = 0.0

    def __post_init__(self):
        eq = _number(self.equilibrium_modulus, 'equilibrium_modulus')
        eta = _number(self.parallel_viscosity, 'parallel_viscosity')
        try:
            branches = tuple(tuple(x) for x in self.branches)
        except TypeError as exc:
            raise ValueError('branches must contain (modulus, relaxation_time) pairs') from exc
        if any(len(pair) != 2 for pair in branches):
            raise ValueError('branches must contain pairs')
        branches = tuple((_number(e, 'branch modulus', positive=True),
                          _number(t, 'relaxation_time', positive=True)) for e, t in branches)
        if not (eq or eta or branches):
            raise ValueError('an all-zero material is unsupported')
        if not math.isfinite(eq + sum(e for e, _ in branches)):
            raise ValueError('instantaneous spring modulus overflow')
        object.__setattr__(self, 'equilibrium_modulus', eq)
        object.__setattr__(self, 'parallel_viscosity', eta)
        object.__setattr__(self, 'branches', branches)

    @classmethod
    def maxwell(cls, modulus, viscosity):
        e = _number(modulus, 'modulus', positive=True)
        eta = _number(viscosity, 'viscosity', positive=True)
        return cls(0.0, ((e, eta / e),))

    @classmethod
    def kelvin_voigt(cls, modulus, viscosity):
        return cls(_number(modulus, 'modulus', positive=True), (),
                   _number(viscosity, 'viscosity', positive=True))

    @classmethod
    def standard_linear_solid(cls, equilibrium_modulus, branch_modulus, relaxation_time):
        return cls(_number(equilibrium_modulus, 'equilibrium_modulus', positive=True),
                   ((branch_modulus, relaxation_time),))

    @property
    def instantaneous_spring_modulus(self):
        """Spring modulus; a parallel dashpot has no finite step-strain modulus."""
        return self.equilibrium_modulus + sum(e for e, _ in self.branches)

    def to_dict(self):
        return {'schema': 'fibernet.linear_viscoelastic/1',
                'equilibrium_modulus': self.equilibrium_modulus,
                'branches': [list(pair) for pair in self.branches],
                'parallel_viscosity': self.parallel_viscosity,
                'units': {'modulus': 'Pa', 'time': 's', 'viscosity': 'Pa*s'}}

    @classmethod
    def from_dict(cls, record):
        if not isinstance(record, dict) or set(record) != {
                'schema', 'equilibrium_modulus', 'branches', 'parallel_viscosity', 'units'}:
            raise ValueError('invalid viscoelastic material record')
        if record['schema'] != 'fibernet.linear_viscoelastic/1' or record['units'] != {
                'modulus': 'Pa', 'time': 's', 'viscosity': 'Pa*s'}:
            raise ValueError('unsupported material schema or units')
        return cls(record['equilibrium_modulus'], record['branches'], record['parallel_viscosity'])

    def coefficients(self, dt):
        """Return branch implicit factors and total consistent tangent (Pa)."""
        dt = _number(dt, 'dt', positive=True)
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            tau = np.array([t for _, t in self.branches], dtype=float)
            # Evaluate tau/(tau+dt) without overflowing tau+dt or dt/tau.
            a = np.empty_like(tau)
            large = tau >= dt
            a[large] = 1.0 / (1.0 + dt / tau[large])
            ratio = tau[~large] / dt
            a[~large] = ratio / (1.0 + ratio)
            if np.any(a < 1e-6):
                raise ValueError(f'dt={dt!r} s exceeds the strain-history precision domain '
                                 '(tau/(tau+dt) < 1e-6); subdivide the time interval')
            moduli = np.array([e for e, _ in self.branches], dtype=float)
            tangent = self.equilibrium_modulus + float(np.dot(moduli, a)) + self.parallel_viscosity / dt
        if not math.isfinite(tangent) or tangent <= 0:
            raise ValueError('nonpositive or nonfinite algorithmic tangent')
        return a, float(tangent)

    def trial(self, strain, previous_strain, viscous_strain, dt):
        """Pure backward-Euler material-point update.

        strain has any finite shape S; viscous_strain has (branches, *S).
        Returned energy densities are J/m^3 for ordinary dimensionless strain.
        The beam adapter uses generalized strains and applies its A/I metric.
        No returned array aliases the input state.
        """
        dt = _number(dt, 'dt', positive=True)
        s = np.array(strain, dtype=float, copy=True)
        old = np.array(previous_strain, dtype=float, copy=True)
        v = np.array(viscous_strain, dtype=float, copy=True)
        if old.shape != s.shape or v.shape != (len(self.branches),) + s.shape:
            raise ValueError('material trial state shape mismatch')
        if not all(np.isfinite(x).all() for x in (s, old, v)):
            raise ValueError('material trial requires finite state')
        a, tangent = self.coefficients(dt)
        shape = (len(self.branches),) + (1,) * s.ndim
        e = np.array([x for x, _ in self.branches]).reshape(shape)
        tau = np.array([x for _, x in self.branches]).reshape(shape)
        a = a.reshape(shape)
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            new_v = a * v + (1.0 - a) * s
            q_old = e * (old - v)
            q = e * (s - new_v)
            ds = s - old
            rate = ds / dt
            stress = self.equilibrium_modulus * s + q.sum(axis=0) + self.parallel_viscosity * rate
            stored = 0.5 * (self.equilibrium_modulus * s**2 + (q**2 / e).sum(axis=0))
            previous_stored = 0.5 * (self.equilibrium_modulus * old**2 + (q_old**2 / e).sum(axis=0))
            physical = dt * ((q**2 / e / tau).sum(axis=0) + self.parallel_viscosity * rate**2)
            numerical = 0.5 * (self.equilibrium_modulus * ds**2 + ((q - q_old)**2 / e).sum(axis=0))
        values = {'stress': stress, 'viscous_strain': new_v,
                  'stored_energy_density': stored, 'previous_stored_energy_density': previous_stored,
                  'physical_dissipation_density': physical,
                  'numerical_dissipation_density': numerical,
                  'endpoint_work_density': stress * ds, 'tangent': tangent}
        if not all(np.isfinite(x).all() for x in values.values()):
            raise FloatingPointError('nonfinite material trial')
        return values
