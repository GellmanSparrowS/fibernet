"""Historical module: the former BeamFEM class is no longer provided here.

Supported beam-frame FEM remains available through
``fibernet.simulate(graph, backend="fem", mode="stretch")`` (2-D/3-D),
``mode="history"`` (2-D viscoelastic), or the advanced
``fibernet.ml.beam_frame_fem.BeamFrameFEM`` interface.
3-D stretch is linear-only; 2-D also has incremental updated-geometry solves.
Spring dynamics uses TaichiEngine and is a distinct physical model, not a FEM
replacement. This module does not alias or silently substitute either model.
"""
