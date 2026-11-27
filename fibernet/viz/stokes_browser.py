"""Loopback VTK/Trame preview of a FiberNet quadratic velocity VTU."""

import argparse
from pathlib import Path

import numpy as np

from fibernet.viz._stokes_velocity_vtu import read_own_quadratic_vtu
from fibernet.viz.tet4_browser import _restrict_to_local_origin

try:
    from trame.app import get_server
    from trame.ui.html import DivLayout
    from trame.widgets import html,vtk
    from vtkmodules.util.numpy_support import numpy_to_vtk,numpy_to_vtkIdTypeArray,vtk_to_numpy
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray,vtkPolyData,vtkStaticCellLocator,vtkUnstructuredGrid,VTK_TETRA,VTK_QUADRATIC_TETRA
    from vtkmodules.vtkCommonMath import vtkRungeKutta45
    from vtkmodules.vtkFiltersCore import vtkTubeFilter
    from vtkmodules.vtkFiltersGeometry import vtkDataSetSurfaceFilter
except ImportError as exc:
    raise ImportError("The Stokes browser requires the optional FiberNet browser-viz extra") from exc


def vertices(x):
    p=vtkPoints()
    p.SetData(numpy_to_vtk(np.ascontiguousarray(x),deep=True))
    return p


def grid(x,c):
    if c.shape[1]!=4: raise ValueError("Expected tetrahedral transport grid")
    cells=vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(np.arange(0,4*(len(c)+1),4,dtype=np.int64),deep=True),
                  numpy_to_vtkIdTypeArray(np.ascontiguousarray(c,dtype=np.int64).ravel(),deep=True))
    g=vtkUnstructuredGrid()
    g.SetPoints(vertices(x)); g.SetCells(VTK_TETRA,cells)
    return g


def attach(pt,name,arr):
    a=numpy_to_vtk(np.ascontiguousarray(arr),deep=True); a.SetName(name);pt.AddArray(a)
    return a


def compact(lines, fluid, tolerance_mm=.003, tolerance_speed=.5):
    x=vtk_to_numpy(lines.GetPoints().GetData())
    velocity=vtk_to_numpy(lines.GetPointData().GetArray("speed_um_s"))
    loc=vtkStaticCellLocator()
    loc.SetDataSet(fluid)
    loc.BuildLocator()
    cells=lines.GetLines()
    index=vtk_to_numpy(cells.GetConnectivityArray())
    offsets=vtk_to_numpy(cells.GetOffsetsArray())
    coords,vectors,pieces=[],[],[]
    max_voxel_error=max_color_error=0.
    rejected=0
    for a,b in zip(offsets[:-1],offsets[1:]):
        ids=index[a:b]
        xyz=x[ids]; v=velocity[ids]
        chosen={0,len(ids)-1}
        stack=[(0,len(ids)-1)]
        while stack:
            low,high=stack.pop()
            if high<=low+1:
                continue
            vec=xyz[high]-xyz[low]
            length=float(np.linalg.norm(vec))
            query=xyz[low+1:high]-xyz[low]
            t=np.clip(query@vec/length**2,0,1) if length else np.zeros(len(query))
            geom=np.linalg.norm(query-t[:,None]*vec,axis=1)
            speed=abs(v[low+1:high]-((1.-t)*v[low]+t*v[high]))
            error=np.maximum(geom/tolerance_mm,speed/tolerance_speed)
            peak=float(error.max())
            split=low+1+int(np.argmax(error))
            if peak<=1.:
                # Audit interpolated segment positions on the reference P1
                # transport mesh; this is sampled, not a geometric certificate.
                points=np.linspace(xyz[low],xyz[high],max(3,1+int(np.ceil(length/.012))))
                if all(loc.FindCell(q)>=0 for q in points):
                    max_voxel_error=max(max_voxel_error,float(geom.max()))
                    max_color_error=max(max_color_error,float(speed.max()))
                    continue
                rejected+=1
                split=(low+high)//2
            chosen.add(split)
            stack.extend(((low,split),(split,high)))
        keep=np.asarray(sorted(chosen),dtype=int)
        coords.extend(xyz[keep])
        vectors.extend(v[keep])
        pieces.append(len(keep))
    out=vtkPolyData()
    out.SetPoints(vertices(np.array(coords)))
    lines2=vtkCellArray()
    off=np.r_[0,np.cumsum(pieces)]
    lines2.SetData(numpy_to_vtkIdTypeArray(off.astype(np.int64),deep=True),
                  numpy_to_vtkIdTypeArray(np.arange(len(coords),dtype=np.int64),deep=True))
    out.SetLines(lines2)
    attach(out.GetPointData(),"speed_um_s",np.array(vectors))
    return out, {"compressed_trace_points":len(coords),"max_checked_geometry_error_mm":max_voxel_error,
                 "max_checked_speed_error_um_s":max_color_error,"guard_rejected_chords":rejected}


def grid2(x,c):
    cells=vtkCellArray(); m=c.shape[1]
    cells.SetData(numpy_to_vtkIdTypeArray(np.arange(0,m*(len(c)+1),m,dtype=np.int64),deep=True),
                 numpy_to_vtkIdTypeArray(np.ascontiguousarray(c,dtype=np.int64).ravel(),deep=True))
    g=vtkUnstructuredGrid();g.SetPoints(vertices(x));g.SetCells(VTK_QUADRATIC_TETRA,cells)
    return g


def tracer(case,seeds,max_step,max_distance=8.):
    from vtkmodules.vtkFiltersFlowPaths import vtkStreamTracer
    src=vtkPolyData();src.SetPoints(vertices(seeds))
    r=vtkStreamTracer();r.SetInputData(case);r.SetSourceData(src)
    r.SetIntegrator(vtkRungeKutta45());r.SetIntegrationDirectionToForward()
    r.SetComputeVorticity(False);r.SetMaximumPropagation(max_distance);r.SetInitialIntegrationStep(.025)
    r.SetMinimumIntegrationStep(.0005);r.SetMaximumIntegrationStep(max_step)
    r.SetMaximumError(5e-8);r.SetMaximumNumberOfSteps(5000);r.Update()
    return r.GetOutput()


def tracks(poly,seeds):
    lines=poly.GetLines(); offsets=vtk_to_numpy(lines.GetOffsetsArray()); ids=vtk_to_numpy(lines.GetConnectivityArray())
    x=vtk_to_numpy(poly.GetPoints().GetData()); records={}
    startid=poly.GetCellData().GetArray("SeedIds")
    numbers=vtk_to_numpy(startid) if startid else np.full(poly.GetNumberOfLines(),-1,dtype=int)
    for i,n in enumerate(numbers):
        a=x[ids[offsets[i]:offsets[i+1]]].copy()
        key=int(n) if n>=0 else np.argmin(np.linalg.norm(a[0]-seeds,axis=1))
        assert key not in records and np.linalg.norm(a[0]-seeds[key])<1e-6
        records[key]=a
    return records


def _scenes(path, reverse=False):
    points_m,nodes,velocity_m_s=read_own_quadratic_vtu(path)
    npnt,ne=len(points_m),len(nodes)
    if not np.max(abs(velocity_m_s))>0.:
        raise ValueError("No nonzero flow for streamline tracing")
    # Convert both coordinates and vectors to mm-based units before
    # tracing; the legend later converts speed alone to µm/s.
    xyz=(points_m-points_m.min(axis=0))*1000.
    velocity_mm_s=velocity_m_s*1000.
    if not np.isfinite(xyz).all() or not np.isfinite(velocity_mm_s).all():
        raise ValueError("Invalid geometry or velocity scale")
    xyz10=xyz[nodes]
    edge_ids=np.array(((0,1),(1,2),(2,0),(0,3),(1,3),(2,3)),dtype=int)
    mid=xyz10[:,edge_ids].mean(axis=2)
    length=np.ptp(xyz,axis=0).max()
    if (not 0.<length<=2e4 or not np.allclose(xyz10[:,4:],mid,rtol=0.,atol=2e-12*length)
            or np.any(np.linalg.det(xyz10[:,1:4]-xyz10[:,:1])<=0.)):
        raise ValueError("Expected a finite, positively oriented, affine tetra10 domain")
    g=grid2(xyz,nodes)
    g.GetPointData().SetActiveVectors(attach(g.GetPointData(),"velocity_mm_s",velocity_mm_s).GetName())
    linear=grid(xyz,nodes[:,:4])
    lo,hi=xyz.min(axis=0),xyz.max(axis=0)
    size=hi-lo
    if not np.min(size)>0.:
        raise ValueError("The velocity domain must be three-dimensional")
    x=hi[0]-.003*size[0] if reverse else lo[0]+.003*size[0]
    values=np.linspace(.115,.885,7)
    loc=vtkStaticCellLocator();loc.SetDataSet(linear);loc.BuildLocator()
    checked=np.asarray([[x,lo[1]+i*size[1],lo[2]+j*size[2]] for i in values for j in values])
    inside=np.asarray([loc.FindCell(seed)>=0 for seed in checked])
    seeds=checked[inside]
    if not len(seeds):
        raise ValueError("None of the bounded seed candidates falls inside this cap")
    raw=tracer(g,seeds,.05,max_distance=2.5*np.linalg.norm(size))
    if raw.GetNumberOfLines()==0:
        raise ValueError("The saved velocity did not generate any streamlines from the selected cap")
    traced=tracks(raw,seeds)
    speed=vtk_to_numpy(raw.GetPointData().GetVectors())*1000.
    speed=np.sqrt((speed**2).sum(axis=1))
    attach(raw.GetPointData(),"speed_um_s",speed)
    span=float(np.percentile(speed,99.5))
    if not span>0: raise ValueError("The seeded paths do not contain nonzero speed")
    q,stats=compact(raw,linear,tolerance_mm=.001*float(np.max(size)),tolerance_speed=.01*span)
    tube=vtkTubeFilter();tube.SetInputData(q);tube.SetRadius(.003*float(np.min(size)))
    tube.SetNumberOfSides(6);tube.CappingOn();tube.Update()
    exterior=vtkDataSetSurfaceFilter();exterior.SetInputData(linear);exterior.Update()
    report=dict(cells=ne,traces=len(traced),point_count=npnt,span=span,**stats)
    return tube.GetOutput(),exterior.GetOutput(),report,(lo,hi)


def _serve(path,port,reverse):
    tube,skin,r,bounds=_scenes(path,reverse)
    span=r["span"]
    lo,hi=bounds
    mid=(lo+hi)/2
    width=max(hi-lo)
    pos=(mid+np.array([.85,-1.3,1.07])*width).tolist()
    server=get_server("fibernet-flow-file",client_type="vue3")
    server.state.update(dict(showShell=False))
    shell_anim="{visibility: showShell}"
    button="background:#e9f2ef;border:1px solid #7eaaa6;border-radius:5px;color:#194b48;margin:5px 0;padding:6px 9px;cursor:pointer"
    with DivLayout(server):
        with html.Div(style="display:flex;position:fixed;inset:0;flex-direction:column;background:#f4f7f2;font:14px system-ui;color:#18313d"):
            with html.Header(style="padding:16px 20px;border-bottom:1px solid #d2ddd3"):
                html.Div("FIBERNET / FILE VISUALIZATION",style="font-size:10px;letter-spacing:.17em;color:#397676;font-weight:650")
                html.Div("Explore a saved quadratic flow field",style="font-size:23px;font-weight:650;margin-top:3px")
            with html.Div(style="flex:1;min-height:0;position:relative;isolation:isolate"):
                with vtk.VtkView(ref="view",background=("bg",[.925,.941,.921]),
                                 style="position:absolute;inset:0;width:100%;height:100%;z-index:0") as view:
                    with vtk.VtkGeometryRepresentation(id="speed-traces",color_map_preset="Viridis (matplotlib)",
                           color_data_range=("speedRange",[0.,span]),mapper=("velocityMapper",{
                               "scalarVisibility":True,"scalarMode":3,"colorMode":1,"colorByArrayName":"speed_um_s"})):
                        vtk.VtkMesh("stokesStreamlines",dataset=tube,point_arrays=["speed_um_s"])
                    with vtk.VtkGeometryRepresentation(id="wall",actor=(shell_anim,),property=("shellMat",{
                           "color":[.24,.46,.55],"opacity":.24})):
                        vtk.VtkMesh("fluidDomainBoundary",dataset=skin)
                with html.Div(id="details",style="position:absolute;z-index:2;right:12px;top:12px;max-width:calc(100% - 24px);box-sizing:border-box;padding:10px 14px;background:#fbfdf9;border:1px solid #c9d6cf;border-radius:6px"):
                    html.Div(f"{r['traces']} streamlines · {r['cells']:,} tetra10 cells")
                    html.Div("Speed on streamlines (µm/s)",style="margin-top:9px;font-size:12px")
                    html.Div(style="height:7px;margin:5px 0;background:linear-gradient(to right,#440154,#3b528b,#21918c,#5ec962,#fde725);border-radius:4px")
                    html.Div(f"0 – {span:.1f} (clips above the 99.5th percentile)",style="font-size:11px")
                    html.Button("Domain boundary",id="wall-toggle",style=button,click="showShell = !showShell",
                                raw_attrs=[":aria-pressed=\"showShell ? 'true' : 'false'\""])
                    html.Button("Fit view",id="set-view",style=button+";margin-left:5px",
                                click=lambda:view.set_camera({"position":pos,"focalPoint":mid.tolist(),"viewUp":[0,0,1]}))
            html.Footer("Derived from the saved velocity field · steady streamlines, not measured particle paths",
                        style="padding:10px 20px;font-size:11px;border-top:1px solid #d2ddd3")
    _restrict_to_local_origin(server,port)
    server.start(port=port,host="127.0.0.1",open_browser=False)


def main():
    """Read and display a locally generated FiberNet quadratic velocity VTU."""
    p=argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("input",type=Path)
    p.add_argument("--port",type=int,default=8765)
    p.add_argument("--from-max-x",action="store_true",help="seed from the high-x cap")
    a=p.parse_args()
    if not 1024<=a.port<=65535:
        p.error("port must be 1024 through 65535")
    _serve(a.input,a.port,a.from_max_x)


if __name__=="__main__":
    main()
