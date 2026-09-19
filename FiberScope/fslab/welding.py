"""Bounded initial-geometry crosslinks for intersecting planar fiber segments."""
from dataclasses import dataclass
import numpy as np


@dataclass
class IntersectionWelder:
    tolerance: float = 1e-7
    max_nodes: int = 40000
    max_pairs: int = 2000000

    def apply(self, graph):
        from .fibernet_bridge import ensure_fibernet
        ensure_fibernet()
        from fibernet.core.structure_graph import StructureGraph
        positions=np.asarray(graph.node_positions(),float)[:,:2]
        edges=np.asarray(graph.edge_array(),int)[:,:2]
        segments=positions[edges]
        length=np.linalg.norm(segments[:,1]-segments[:,0],axis=1)
        step=max(float(np.median(length[length>self.tolerance])),self.tolerance*100)
        cells={}; pairs=set(); entries=0
        for i,segment in enumerate(segments):
            lo=np.floor((segment.min(0)-self.tolerance)/step).astype(int)
            hi=np.floor((segment.max(0)+self.tolerance)/step).astype(int)
            if np.prod(hi-lo+1)>100000: raise MemoryError('weld spatial budget exceeded')
            for x in range(lo[0],hi[0]+1):
                for y in range(lo[1],hi[1]+1):
                    bucket=cells.setdefault((x,y),[])
                    pairs.update((j,i) for j in bucket)
                    bucket.append(i); entries+=1
                    if len(pairs)>self.max_pairs or entries>self.max_pairs:
                        raise MemoryError('weld intersection budget exceeded')
        nodes=[]; lookup={}
        def node(point):
            key=tuple(np.rint(point/self.tolerance).astype(np.int64))
            if key not in lookup:
                if len(nodes)>=self.max_nodes: raise MemoryError('weld node budget exceeded')
                lookup[key]=len(nodes); nodes.append(point)
            return lookup[key]
        ids=[node(p) for p in positions]
        cuts=[{0.:ids[a],1.:ids[b]} for a,b in edges]
        cross=lambda a,b:a[0]*b[1]-a[1]*b[0]
        for i,j in sorted(pairs):
            a,b=segments[i]; c,d=segments[j]
            u,v=b-a,d-c; denominator=cross(u,v)
            if abs(denominator)>self.tolerance*max(length[i],length[j],1.):
                t=cross(c-a,v)/denominator; q=cross(c-a,u)/denominator
                if -1e-9<=t<=1+1e-9 and -1e-9<=q<=1+1e-9:
                    t=float(np.clip(t,0,1)); q=float(np.clip(q,0,1))
                    index=node(a+t*u); cuts[i][t]=index; cuts[j][q]=index
            elif abs(cross(c-a,u))<=self.tolerance*max(length[i],1.):
                for source,target in ((i,j),(j,i)):
                    p0,p1=segments[target]; delta=p1-p0; square=float(delta@delta)
                    if square<=self.tolerance**2: continue
                    for endpoint,point in enumerate(segments[source]):
                        t=float((point-p0)@delta/square)
                        if -1e-9<=t<=1+1e-9:
                            cuts[target][float(np.clip(t,0,1))]=ids[edges[source,endpoint]]
        result=StructureGraph(dimension=2)
        for point in nodes: result.add_node(point,merge=False)
        trips=[]
        for index,cut in enumerate(cuts):
            chain=[]
            for _,n in sorted(cut.items()):
                if not chain or n!=chain[-1]: chain.append(n)
            for a,b in zip(chain,chain[1:]):
                if np.linalg.norm(np.asarray(nodes[a])-nodes[b])<=self.tolerance: continue
                original=graph.edges[index]
                result.add_edge(a,b,radius=original.radius,material=original.material)
            trips.extend(zip(chain,chain[1:],chain[2:]))
        result.metadata.update(weld_intersections=True,weld_triplets=list(trips),
                               unwelded_nodes=len(positions),unwelded_edges=len(edges))
        return result
