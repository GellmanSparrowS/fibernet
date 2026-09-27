"""Analytic polygonal cylinders and round joints with manifold Boolean union."""
import numpy as np


def tube_mesh(points, edges, radius, progress=None, stop_cb=None, sides=16):
    import manifold3d as m
    unique,inverse=np.unique(np.round(points,8),axis=0,return_inverse=True)
    edges=np.unique(np.sort(inverse[np.asarray(edges,int)],axis=1),axis=0)
    edges=edges[edges[:,0]!=edges[:,1]]
    # Merge straight degree-two pieces only in the solid, never in the path graph.
    adjacency=[set() for _ in unique]
    for a,b in edges:
        adjacency[a].add(int(b)); adjacency[b].add(int(a))
    for node,neighbors in enumerate(adjacency):
        if len(neighbors)!=2:
            continue
        a,b=sorted(neighbors)
        u,v=unique[a]-unique[node],unique[b]-unique[node]
        if np.dot(u,v)<0 and np.linalg.norm(np.cross(u,v))<1e-8*np.linalg.norm(u)*np.linalg.norm(v):
            adjacency[a].discard(node); adjacency[b].discard(node)
            adjacency[a].add(b); adjacency[b].add(a); neighbors.clear()
    edges=[(a,b) for a,neighbors in enumerate(adjacency) for b in neighbors if a<b]
    used=sorted({n for edge in edges for n in edge})
    sphere=m.Manifold.sphere(radius,circular_segments=sides)
    if (len(edges)*4*sides+len(used)*sphere.num_tri())>12000000:
        raise MemoryError('tube mesh exceeds available geometry budget; reduce fibers or surface patches')
    # Spatial ordering keeps early Boolean batches local.
    edges.sort(key=lambda e: tuple((unique[e[0]]+unique[e[1]])*.5))
    pieces=[]; batches=[]
    total=len(edges)+len(used)
    def add(shape,index):
        if stop_cb and stop_cb():
            raise InterruptedError('solid export cancelled')
        pieces.append(shape)
        if len(pieces)>=96 or index==total-1:
            batch=m.Manifold.batch_boolean(pieces,m.OpType.Add)
            batch.num_tri()  # materialize bounded chunks, permitting cancellation between them
            batches.append(batch); pieces.clear()
        if progress and index%32==0:
            progress(int(80*index/max(total,1)))
    for i,(a,b) in enumerate(edges):
        delta=unique[b]-unique[a]; length=np.linalg.norm(delta); axis=delta/length
        helper=np.eye(3)[np.argmin(np.abs(axis))]
        x=np.cross(helper,axis); x/=np.linalg.norm(x)
        y=np.cross(axis,x)
        phase=(i*.61803398875 % 1.)*2*np.pi/sides
        x,y=x*np.cos(phase)+y*np.sin(phase),-x*np.sin(phase)+y*np.cos(phase)
        transform=np.column_stack([x,y,axis,unique[a]])
        add(m.Manifold.cylinder(length,radius,circular_segments=sides).transform(transform),i)
    for i,node in enumerate(used):
        add(sphere.translate(unique[node]),len(edges)+i)
    if progress: progress(85)
    levels = max(1, int(np.ceil(np.log2(max(1,len(batches))))))
    level = 0
    while len(batches)>1:
        merged=[]
        for start in range(0,len(batches),2):
            if stop_cb and stop_cb(): raise InterruptedError('solid export cancelled')
            part=m.Manifold.batch_boolean(batches[start:start+2],m.OpType.Add)
            if part.num_tri()>8000000:
                raise MemoryError("solid union exceeds geometry budget")
            merged.append(part)
        batches=merged
        level+=1
        if progress: progress(85+int(10*level/levels))
    result=batches[0]
    result=result.as_original().simplify(max(radius*1e-3,1e-5))
    for _ in range(2):
        raw=result.to_mesh()
        cleaned=m.Mesh(np.array(np.round(raw.vert_properties,4),dtype=np.float32,order="C"),np.array(raw.tri_verts,dtype=np.uint32,order="C"),tolerance=1e-4)
        result=m.Manifold(cleaned).as_original().simplify(max(radius*1e-3,1e-5))
    mesh=result.to_mesh()
    if stop_cb and stop_cb(): raise InterruptedError('solid export cancelled')
    if result.status()!=m.Error.NoError:
        raise ValueError('cylinder union failed: '+str(result.status()))
    vertices=np.asarray(mesh.vert_properties[:,:3],float)
    faces=np.array(mesh.tri_verts,dtype=np.int32,copy=True)
    # Separate zero-thickness contacts that STL readers would weld into non-manifold edges.
    for attempt in range(3):
        _,inverse,counts=np.unique(np.round(vertices,5),axis=0,return_inverse=True,return_counts=True)
        touching=np.flatnonzero(counts[inverse]>1)
        if not len(touching): break
        triangles=vertices[faces]
        face_normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        normals=np.zeros_like(vertices)
        for column in range(3): np.add.at(normals,faces[:,column],face_normals)
        lengths=np.linalg.norm(normals[touching],axis=1)
        if np.any(lengths<1e-12): raise ValueError('ambiguous zero-thickness solid contact')
        vertices[touching]-=normals[touching]/lengths[:,None]*max(radius*1e-3,1e-4)*(attempt+1)
    if len(np.unique(np.round(vertices,5),axis=0))!=len(vertices):
        raise ValueError('unresolved coincident solid vertices')
    return vertices,faces
