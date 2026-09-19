"""Generate original procedural quad assets; rerun with Python 3.10+."""
from pathlib import Path
import math


class DemoMeshes:
    def __init__(self, output):
        self.output = Path(output)

    def cloth(self, name, nx, ny, keep, shape):
        vertices, faces, ids = [], [], {}
        def vertex(i, j):
            if (i, j) not in ids:
                ids[i, j] = len(vertices)+1
                vertices.append(shape(i/nx, j/ny))
            return ids[i, j]
        for j in range(ny):
            for i in range(nx):
                if keep((i+.5)/nx, (j+.5)/ny):
                    faces.append([vertex(i, j), vertex(i+1, j), vertex(i+1, j+1), vertex(i, j+1)])
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output/name).open('w', encoding='utf-8', newline='\n') as f:
            f.write('# FiberScope original procedural demonstration asset; CC0-1.0\n')
            for p in vertices:
                f.write('v %.7f %.7f %.7f\n' % p)
            for face in faces:
                f.write('f %d %d %d %d\n' % tuple(face))

    def build(self):
        drape = lambda u, v: (3*(u-.5), 3*(v-.5), .15*math.cos(4*math.pi*u)*math.sin(math.pi*v))
        self.cloth('demo_tshirt.obj', 20, 20,
                   lambda u, v: ((.25 < u < .75) or .62 < v < .86) and not (.4 < u < .6 and v > .9), drape)
        self.cloth('demo_vest.obj', 16, 20,
                   lambda u, v: not ((u < .18 or u > .82) and v > .68) and not (.36 < u < .64 and v > .84), drape)
        self.cloth('demo_scarf.obj', 10, 28, lambda u, v: True,
                   lambda u, v: (u-.5, 4*(v-.5), .25*math.sin(3*math.pi*v)))
        self.cloth('demo_sleeve.obj', 20, 18, lambda u, v: True,
                   lambda u, v: ((.45+.15*v)*math.cos(2*math.pi*u), 2*(v-.5),
                                  (.45+.15*v)*math.sin(2*math.pi*u)))


if __name__ == '__main__':
    DemoMeshes(Path(__file__).resolve().parents[1]/'assets'/'obj').build()
