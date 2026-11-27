# SPDX-License-Identifier: MIT
"""Generate and solve a synthetic two-material, round cross-section 3D fiber network.

Three longitudinal cylinders are joined by four transverse and one diagonal
cylinder. A conformal material partition passes through the connected solid.
Use a fresh process and Gmsh 4.13.1, Meshio and FiberNet's solid-iterative extra.
"""
from collections import defaultdict
from dataclasses import replace
from hashlib import sha256
from itertools import combinations
from math import fsum, isclose, log, pi
from pathlib import Path
import argparse
import json

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from fibernet.examples.tet4_iterative_mechanoelectric_sweep import _power_weighted_strain
from fibernet.io.tet4_meshio import read_tet4_vtu_bundle
from fibernet.sim.tet4_dc import solve_dc_tet4

_TIERS = {"quick": (.075, .09), "reference": (.045, .04), "finer": (.033, .025)}
_CONDUCTIVITIES = {"uniform": {7: 1., 19: 1.},
                   "contrast": {7: 1., 19: .4},
                   "mirrored_contrast": {7: .4, 19: 1.}}


def _require(condition, message):
    if not condition:
        raise ValueError(message)

def _generate(gmsh, path, step):
    cut = 1.5
    if gmsh.isInitialized():
        raise ValueError("Run this example in a fresh Gmsh process")
    gmsh.initialize(readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal",0)
        gmsh.model.add("three-path-cylinders")
        occ=gmsh.model.occ
        def tube(p,q,r):
            return 3,occ.addCylinder(*p,*(b-a for a,b in zip(p,q)),r)
        ax=[tube((0,y,z),(3,y,z),.11) for y,z in ((.23,.26),(.80,.26),(.23,.80))]
        links=[tube((x,.23,.26),(x,.80,.26),.074) for x in (.68,2.15)]
        links += [tube((x,.23,.26),(x,.23,.80),.074) for x in (1.05,2.42)]
        links += [tube((1.20,.80,.26),(1.83,.23,.80),.07)]
        volume,_=occ.fuse(ax[:1],ax[1:]+links)
        occ.synchronize()
        _require(len(volume)==1 and volume[0][0]==3,
                     "Expected one connected solid after fusing all round fibers")
        mass=occ.getMass(*volume[0])
        _require(.36<mass<.51,
                     "Unexpected CAD solid volume")
        rectangle=[occ.addPoint(cut,y,z) for y,z in ((0,0),(1.10,0),(1.10,1.10),(0,1.10))]
        loop=occ.addCurveLoop([occ.addLine(p,q) for p,q in zip(rectangle,rectangle[1:]+rectangle[:1])])
        sheet=occ.addPlaneSurface([loop])
        _,mapping=occ.fragment(volume,[(2,sheet)])
        occ.synchronize()
        bodies=[i for i in mapping[0] if i[0]==3]
        _require(len(bodies)==2 and sorted(bodies)==sorted(gmsh.model.getEntities(3)),
                     "Expected two conformal material volumes")
        by_x=sorted(bodies,key=lambda a:occ.getCenterOfMass(*a)[0])
        _require(occ.getCenterOfMass(*by_x[0])[0]<cut<occ.getCenterOfMass(*by_x[1])[0],
                     "The volume split does not bracket the material interface")
        _require(abs(sum(occ.getMass(*a) for a in by_x)-mass)<1e-6*mass,
                     "The material partition changed the CAD volume")
        for name,tag,(_,entity) in zip(("material_low","material_high"),(7,19),by_x):
            gmsh.model.addPhysicalGroup(3,[entity],tag,name)
        boundary=gmsh.model.getBoundary(by_x,combined=True,oriented=False)
        side_by_tag={}
        for name,tag,x in (("hold_face",31,0.),("pull_face",37,3.)):
            selected=[]
            for d,q in boundary:
                box=gmsh.model.getBoundingBox(d,q)
                if max(abs(box[0]-x),abs(box[3]-x))<1e-5: selected.append(q)
            _require(len(selected)==3 and abs(sum(occ.getMass(2,k) for k in selected)-3*pi*.11**2)<1e-9,
                     "Expected three complete clamp disks per end")
            gmsh.model.addPhysicalGroup(2,selected,tag,name)
            side_by_tag[name]=len(selected)
        for name,value in {"General.NumThreads":1,"Mesh.MaxNumThreads1D":1,"Mesh.MaxNumThreads2D":1,
                "Mesh.MaxNumThreads3D":1,"Mesh.RandomSeed":1,"Mesh.MeshSizeFromPoints":0,
                "Mesh.MeshSizeFromCurvature":0,"Mesh.MeshSizeExtendFromBoundary":0,
                "Mesh.ElementOrder":1,"Mesh.MeshSizeMin":step,"Mesh.MeshSizeMax":step,
                "Mesh.MshFileVersion":4.1,"Mesh.Binary":0,"Mesh.Algorithm3D":1,"Mesh.SaveAll":0}.items():
            gmsh.option.setNumber(name,value)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(path))
        return {"volume_mm3":mass,"paths":3,"crosslinks":5,"grip_faces":side_by_tag,
            "physical_volume_count":len(bodies),"target_size_mm":step,"sha256":sha256(path.read_bytes()).hexdigest()}
    finally:
        gmsh.finalize()


def _electrical(root, cad_volume_mm3, expected_volume_tolerance):
    b=read_tet4_vtu_bundle(root/"tagged-solid.zip")
    summary=json.loads((root/"summary.json").read_text()); s=json.loads((root/"grips.json").read_text())
    _require(b.length_unit=="mm" and set(b.materials)=={7,19} and summary["axis"]=="x",
                     "Expected the generated two-material x-grip specimen in mm")
    ends=[]; electrodes={}
    for mark,vol in (("hold_ids",0.),("pull_ids",1.)):
        ids=tuple(map(int,s[mark])); at=np.isin(b.node_ids,np.array(ids,dtype=np.uint64))
        _require(at.sum()==len(ids),
                     "Stable electrode IDs no longer match the mesh")
        ends.append(at)
        electrodes.update(dict.fromkeys(ids, vol))
    left,right=ends
    _require(np.allclose(b.points[left,0],0.) and np.allclose(b.points[right,0],3.),
                     "The electrodes do not lie on the specimen caps")
    t=b.tetrahedra
    pairs=np.column_stack((t[:,0].repeat(3), t[:,1:].ravel()))
    g=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(b.points),len(b.points)))
    _require(connected_components(g,directed=False,return_labels=False)==1,
                     "The solid mesh is not connected")
    face_tags=defaultdict(list)
    for rows,tag in zip(t,b.material_ids):
        for face in combinations(sorted(rows.tolist()),3):
            face_tags[face].append(int(tag))
    _require(all(1<=len(tags)<=2 for tags in face_tags.values()),
                     "A tetrahedral face is nonmanifold")
    interfaces=[f for f,pair in face_tags.items() if set(pair)=={7,19}]
    _require(interfaces and max(abs(b.points[list(f),0]-1.5).max() for f in interfaces)<1e-7,
                     "Expected shared material faces on the interface plane")
    xyz=b.points[t]
    det0=np.linalg.det(xyz[:,1:]-xyz[:,:1])
    geom_volume=float(np.sum(np.abs(det0))/6)
    volume_error=abs(geom_volume/cad_volume_mm3-1)
    if volume_error > expected_volume_tolerance:
        raise ValueError("The straight-sided mesh is too coarse for this round geometry; choose a finer tier.")
    with np.load(root/"nodal-fields.npz",allow_pickle=False) as f:
        _require(np.array_equal(f["node_ids"],b.node_ids),
                     "Displacement node IDs no longer match the mesh")
        u=f["displacement_m"]
        _require(np.allclose(u[left],0.,rtol=0.,atol=1e-14) and np.allclose(u[right,0],1.5e-6,rtol=0.,atol=1e-14),
                     "Displacement does not match the prescribed clamp motion")
    reaction=summary["reaction_N"]
    def state(c,scale):
        changed=replace(b,points=b.points+1000.*scale*u)
        q=changed.points[t]
        ratios=np.linalg.det(q[:,1:]-q[:,:1])/det0
        _require(np.min(ratios)>.995 and np.max(ratios)<1.005,
                     "The perturbed geometry exceeds the local small-volume-change limit")
        dc=solve_dc_tet4(changed,conductivity_S_per_m_by_material=c,
            electrode_voltage_V_by_id=electrodes,solver="cg")
        _require(np.array_equal(b.node_ids,dc.node_ids) and np.array_equal(b.cell_ids,dc.cell_ids),
                     "DC results do not match the original mesh IDs")
        amp=fsum(dc.electrode_current_into_solid_A[right])
        _require(amp>0 and abs(dc.net_electrode_current_A)<1e-8*amp,
                     "DC terminal currents do not balance")
        _require(isclose(amp,dc.total_joule_power_W,rel_tol=1e-8),
                     "DC Joule power does not match the terminal work at one volt")
        return 1/amp,dc,float(np.min(ratios)),float(np.max(ratios))
    cases={}
    for lab, c in _CONDUCTIVITIES.items():
        ref,dc,_,_=state(c,0.)
        a,z=_power_weighted_strain(b.points*.001,b.tetrahedra,u,dc.current_density_A_per_m2,dc.joule_power_W_by_cell)
        d=(a+z)*ref
        lo,*vol1=state(c,-1.)
        hi,*vol2=state(c,1.)
        _require(lo<ref<hi,
                     "Expected monotonic resistance for this fixed synthetic specimen")
        fd=log(hi/lo)/2
        _require(isclose(d,fd,rel_tol=2e-6),
                     "Small-strain resistance tangent disagrees with re-solved DC")
        cases[lab]={"case":lab,"tetrahedra":len(t),"vertices":len(b.points),"topology_components":1,"shared_material_faces":len(interfaces),
            "meshed_volume_mm3":geom_volume,"cad_volume_relative_deficit":volume_error,"reaction_N":reaction,"R0_ohm":ref,"R_minus_ohm":lo,"R_plus_ohm":hi,
            "d_lnR_dN":d/reaction,"d_lnR_dm":d/s["stroke_m"],"shape_fd_rel_error":abs(fd/d-1.),
            "volume_ratios":[min(vol1[-2],vol2[-2]),max(vol1[-1],vol2[-1])]}
    return cases

def run(output, *, tier="quick", write_vtu=False):
    """Solve fixed-stroke elasticity and fixed-conductivity DC on the deformed geometry."""
    if tier not in _TIERS:
        raise ValueError("Choose quick, reference or finer")
    try:
        import gmsh
    except (ImportError,OSError) as exc:
        raise RuntimeError("Install the optional Gmsh 4.13.1 native runtime for your platform.") from exc
    if gmsh.__version__ != "4.13.1":
        raise RuntimeError("This deterministic example is validated for Gmsh 4.13.1.")
    from fibernet.examples.tet4_tagged_grip_iterative import prepare,solve
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    step,tolerance=_TIERS[tier]
    msh=output/"round-network.msh"
    geometry=_generate(gmsh,msh,step)
    bundle,grips_sha=prepare(msh,output,length_unit="mm",axis="x",stroke_m=1.5e-6)
    mechanical=solve(bundle,output/"grips.json",output,prepared_grip_sha256=grips_sha,
                     write_vtu=write_vtu,material_sensitivities=True)
    result={
        "straight_path_control": {
            "contrast_over_uniform_resistance_ratio": 1.75,
            "mirrored_over_normal_force_sensitivity_ratio": 4/3,
            "model":"three independent round rods; uniaxial free lateral contraction",
        },
        "tier":tier,
        "geometry":geometry,
        "mechanical":mechanical,
        "resistance":_electrical(output,geometry["volume_mm3"],tolerance),
        "model":"small-strain mechanics; one-way DC geometry, fixed conductivity",
    }
    (output/"round-mechanoelectric.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--tier",choices=_TIERS,default="quick")
    parser.add_argument("--write-vtu",action="store_true")
    args=parser.parse_args()
    result=run(args.output,tier=args.tier,write_vtu=args.write_vtu)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
