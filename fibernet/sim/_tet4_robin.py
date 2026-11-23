"""Bounded Tet4 face terms for linear steady convection."""
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np
from scipy import sparse

from fibernet.sim import tet4 as core
from fibernet.sim.tet4_loads import _prepare_exterior_face_areas, _iter_exterior_face_areas


def _positive_sum(terms, label):
    s=math.fsum(terms)
    if len(terms)>1:
        least=min(range(len(terms)),key=terms.__getitem__)
        if s==math.fsum(x for i,x in enumerate(terms) if i!=least):
            raise ValueError(label+' lost a face contribution')
    return s


@dataclass(frozen=True)
class _RobinTerms:
    rows: np.ndarray
    hA_W_per_K: np.ndarray
    area_m2: np.ndarray
    coefficient: float
    ambient_K: float
    matrix: sparse.csr_matrix

    def _weighted_nodes(self, face_values):
        terms=[[] for _ in range(self.matrix.shape[0])]
        for row, flux in zip(self.rows,face_values):
            for i,q in zip(row,flux):
                terms[i].append(q)
        values=np.array([math.fsum(g) for g in terms])
        core._check_normal_or_zero(values,'nodal convection heat in W')
        return values

    def ambient_forcing(self, reference):
        return self._weighted_nodes(np.repeat(
            ((self.hA_W_per_K/3.)*(self.ambient_K-reference[self.rows[:,0]]))[:,None],3,axis=1))

    def _drive(self, indices, temperature_difference, reference, prescribed):
        a=Fraction.from_float(self.ambient_K)
        ref=Fraction.from_float(float(reference[indices[0]]))
        for i in indices:
            if prescribed is not None and prescribed[0][i]:
                # Use the exact admitted Kelvin value, not rounded centering.
                yield a-Fraction.from_float(float(prescribed[1][i]))
            else:
                yield a-ref-Fraction.from_float(float(temperature_difference[i]))

    def represented_drive_terms(self, temperature_difference, reference, prescribed=None):
        """Retain the h, area and centered drives to the final free row."""
        terms=[[] for _ in range(self.matrix.shape[0])]
        for indices, area in zip(self.rows,self.area_m2):
            offsets=list(self._drive(indices,temperature_difference,reference,prescribed))
            geometry=(area,self.coefficient)
            for j,i in enumerate(indices):
                terms[i].extend((geometry,value,12) for value in (*offsets,offsets[j]))
        return terms

    def outward_exact(self, temperature_difference, reference, prescribed=None):
        # Audit the solved boundary independently of the assembled Robin
        # block. Accumulate represented factors without rounding each face
        # before combining shared nodes or each node before closing a face.
        by_node=[Fraction(0)]*self.matrix.shape[0]
        by_face=[]
        coefficient=Fraction.from_float(self.coefficient)
        for indices, a in zip(self.rows,self.area_m2):
            d=[-v for v in self._drive(indices,temperature_difference,reference,prescribed)]
            w=coefficient*Fraction.from_float(float(a))/12
            total=sum(d)
            for i, di in zip(indices,d):
                by_node[i]+=w*(total+di)
            by_face.append(4*w*total)
        return by_node,by_face

    def outward(self, temperature_difference, reference):
        by_node,by_face=self.outward_exact(temperature_difference,reference)
        node=np.array([float(v) for v in by_node])
        face=np.array([float(v) for v in by_face])
        core._check_normal_or_zero(node,'nodal convection heat in W')
        core._check_normal_or_zero(face,'integrated face convection heat in W')
        return node,face


def _prepare_robin_terms(mesh, nodes, selection, coefficient, ambient):
    h=core._float64(coefficient,'convection coefficient in W/(m² K)')
    a=core._float64(ambient,'ambient temperature in K')
    if not 1.<=a<=2000.:
        raise ValueError('ambient temperature outside [1,2000] K')
    if not (h==0. or 1e-8<=h<=1e8):
        raise ValueError('convection outside pilot coefficient envelope')
    selected=_prepare_exterior_face_areas(mesh,selection)
    if h==0.:
        # Still validate face geometry, even with a disabled boundary.
        for _ in _iter_exterior_face_areas(selected):
            pass
        return _RobinTerms(np.empty((0,3),dtype=np.intp),np.zeros(0),np.zeros(0),h,a,
                           sparse.csr_matrix((len(nodes),len(nodes))))
    rows=[]; scale=[]; areas=[]; terms=defaultdict(list)
    try:
        for f, area in _iter_exterior_face_areas(selected):
            w=h*area
            q=w/12.
            core._check_normal_or_zero((w,q),'integrated convection in W/K')
            if not q:
                raise ValueError('unresolved nonzero convection area weight')
            row=tuple(nodes[i] for i in f)
            rows.append(row); scale.append(w); areas.append(area)
            for i in row:
                for j in row:
                    terms[i,j].append(q*(1+(i==j)))
    except OverflowError as exc:
        raise ValueError('convection area weight outside supported binary64 range') from exc
    ii,jj=zip(*terms)
    coefficients=[_positive_sum(t,'convection matrix entry') for t in terms.values()]
    C=sparse.coo_matrix((coefficients,(ii,jj)),shape=(len(nodes),len(nodes))).tocsr()
    core._check_normal_or_zero(C.data,'convection matrix in W/K')
    return _RobinTerms(np.array(rows,dtype=np.intp),np.array(scale),np.array(areas),h,a,C)


def _add_robin_operator(K,C,fixed):
    """Preserve both represented contributions on active free rows."""
    f=np.flatnonzero(~fixed)
    A=K+C
    if not np.isfinite(A.data).all():
        raise ValueError('nonfinite combined convection operator')
    c=C[f].tocoo()
    if not c.nnz:
        return A
    rows=f[c.row]
    known=np.asarray(K[rows,c.col]).reshape(-1)
    with np.errstate(over='ignore',invalid='ignore'):
        updated=known+c.data
    reject=(known!=0.)&((updated==known)|(updated==c.data))
    opposite=(known<0.)&(known!=-c.data)
    reject|=opposite&(abs(updated)<=1e-10*np.maximum(abs(known),c.data))
    if np.any(reject):
        raise ValueError('convection matrix addition loses a represented contribution')
    core._check_normal_or_zero(updated,'combined conduction and convection matrix in W/K')
    return A
