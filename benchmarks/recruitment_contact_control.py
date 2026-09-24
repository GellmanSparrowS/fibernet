"""Run bowed chiral contact as a positive control for recruitment screens.

Run: python benchmarks/recruitment_contact_control.py
This control checks contact engagement, not an independent force-flow map.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.recruitment_sensitivity import (
    RecruitmentSensitivity, SensitivityConfig)


class ChiralContactControl(RecruitmentSensitivity):
    def __init__(self):
        bow = ((0.0, 0.0), (0.0, -0.225), (0.0, -0.45),
               (0.0, -0.225), (0.0, 0.0))
        super().__init__(SensitivityConfig(
            units=("chiral",), points_per_edge=5,
            line_displacements=bow, target_stretch=2.2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "benchmarks" / "results" /
                                               "recruitment_contact_control.json"))
    args = parser.parse_args()
    ChiralContactControl().run(args.output)
