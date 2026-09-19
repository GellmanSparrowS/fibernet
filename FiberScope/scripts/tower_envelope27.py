"""Regular quad tower envelope fitted to the reference's height and outline."""
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d


class TowerEnvelope:
    def build(self, source, target):
        points=np.array([[float(x) for x in line.split()[1:4]]
                        for line in Path(source).open() if line.startswith('v ')])
        low=points.min(0); high=points.max(0)
        center=(low+high)/2
        height=high[1]-low[1]
        levels=np.linspace(low[1],high[1],40)
        profile=[]
        for y in levels:
            band=points[np.abs(points[:,1]-y)<height*.04]
            if not len(band): band=points[np.argsort(np.abs(points[:,1]-y))[:24]]
            profile.append(np.max(np.abs(band[:,[0,2]]-center[[0,2]]),axis=0))
        profile=np.maximum.accumulate(np.array(profile)[::-1],axis=0)[::-1]
        profile=gaussian_filter1d(profile,1.5,axis=0)
        vertices=[]; faces=[]
        sides=8; rings=18
        for row in range(rings+1):
            for side in range(4):
                for j in range(sides):
                    u=j/sides
                    arch=height*.17*np.sqrt(max(0.,1.-((u-.5)/.32)**2))
                    y=low[1]+arch if row==0 else low[1]+height*(.19+.81*(row-1)/(rings-1))
                    rx=np.interp(y,levels,profile[:,0]); rz=np.interp(y,levels,profile[:,1])
                    corners=np.array([[-rx,-rz],[rx,-rz],[rx,rz],[-rx,rz]])
                    x,z=corners[side]*(1-u)+corners[(side+1)%4]*u+center[[0,2]]
                    vertices.append([x,y,z])
        width=4*sides
        for row in range(rings):
            for j in range(width):
                a=row*width+j; b=row*width+(j+1)%width
                faces.append([a,b,b+width,a+width])
        with Path(target).open('w') as stream:
            for v in vertices: stream.write('v %.8g %.8g %.8g\n'%tuple(v))
            for f in faces: stream.write('f '+' '.join(str(i+1) for i in f)+'\n')
        return vertices,faces
