# FiberNet 4.2.0 / FiberScope 3.1.0

This release brings the existing Python package and Windows desktop APP closer to a shared scientific core. The APP defaults to English. The Python library provides public workflows for planar and curved continuous networks, custom cells, reduced stretch trajectories, tensile recruitment, snapshot features, physical-label datasets, surrogate learning, and target-curve inverse search. The original 2D/3D pattern and beam-frame FEM interfaces remain available.

## Downloads

- **Windows desktop:** download `FiberScope-3.1.0-Windows-x64.zip`, extract the entire folder, and run `FiberScope.exe`. Check the archive against `SHA256SUMS.txt` if needed. Python is not required for this executable. SAC, TD3 and DDPG training require a separately configured Python runtime and are not included in the frozen APP.
- **Python:** install `fibernet==4.2.0` from [PyPI](https://pypi.org/project/fibernet/4.2.0/). The `ml`, `rl`, and `manufacturing` extras install optional dependencies for those tasks.

The [README](https://github.com/GellmanSparrowS/fibernet#readme) contains executable API examples, a reproducible 3D gallery, eight source-backed animations, and the capability matrix. The [Work-mode brief](https://github.com/GellmanSparrowS/fibernet/blob/main/handoff/WORK_MODE_BRIEF_ZH.md) records the separate software-methods paper direction.

## Validation and interpretation

Library source tests passed in 12 Linux/macOS/Windows and Python 3.9–3.12 CI combinations. The Windows desktop suite passed 25/25 on the integration commit; the release workflow reruns it and gates the frozen binary on dependency, smoke, portability, and clean-directory checks before attaching the ZIP. A locally built 4.2.0 wheel passed isolated Python 3.10 API calls.

The recruitment visualization marks edges exceeding a positive axial-strain threshold and the components spanning both grips. It is not a map of total force flow. Cycle-removal and AI selection examples are numerical research workflows with negative and model-dependent results; they do not establish a material-performance improvement or printing feasibility. Third-party-machine installation and experimental material validation remain open.
