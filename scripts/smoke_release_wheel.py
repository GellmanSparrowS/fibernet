"""Run installed-wheel API checks without importing the source checkout.

Run after pip install --no-deps --target <target> <wheel>:
    python -I scripts/smoke_release_wheel.py --target <target>
"""

import argparse
from pathlib import Path
import sys

import numpy as np


class ReleaseWheelSmoke:
    def __init__(self, target):
        self.target = Path(target).resolve()

    def run(self):
        if not (self.target / "fibernet").is_dir():
            raise FileNotFoundError("wheel target does not contain fibernet")
        sys.path.insert(0, str(self.target))
        import fibernet
        from fibernet.analysis import analyze_tensile_recruitment
        from fibernet.ml import PhysicalRegressor

        if fibernet.__version__ != "4.2.0":
            raise AssertionError("wrong installed version")
        origin = Path(fibernet.__file__).resolve()
        if self.target not in origin.parents:
            raise AssertionError("import resolved outside isolated wheel target")
        graph = fibernet.pattern_3d(unit="gyroid", box=(2, 2, 2),
                                    grid=(2, 2, 2))
        if len(graph.nodes) != 203 or len(graph.edges) != 531:
            raise AssertionError("3D API changed")
        edges = np.array([[0, 1], [1, 2]])
        strain = np.array([[0.0, 0.0], [0.2, 0.2]])
        result = analyze_tensile_recruitment(strain, edges,
                                             left_nodes=[0], right_nodes=[2],
                                             n_nodes=3)
        if result.perc_frame != 1:
            raise AssertionError("recruitment API changed")
        if "fslab" in sys.modules:
            raise AssertionError("APP was imported by the library wheel")
        print("[wheel_smoke] version=4.2.0 3D=203/531 recruitment=1 ML=%s" %
              PhysicalRegressor.__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    ReleaseWheelSmoke(parser.parse_args().target).run()
