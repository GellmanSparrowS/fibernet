# FiberScope web demo

Run from the repository root:

~~~bash
python -m pip install -r web_demo/requirements.txt
python web_demo/app.py
~~~

Open http://localhost:7860. The seven tabs call the same fslab numerical core as the desktop software. ML labels are computed from physical simulations; held-out scores can be negative for small datasets. The web queue and bounds protect shared CPU memory; use the desktop for larger jobs. Models and work files are session-specific. API keys are neither required nor embedded.

Test: python web_demo/test_demo.py

The ModelScope deployment entry imports create_app from this module and serves port 7860. Free hardware is used. Deployment secrets must remain outside the repository.


## Showcase defaults

This hosted application is a presentation demo. The desktop release is the complete application.

| Control | Default | Maximum |
| --- | --- | --- |
| Surface | Lung | Seven desktop models or OBJ import |
| Target surface quads | 48 | 10,000 |
| Samples per fiber segment | 2 | 10 |
| Planar grid per direction | 3 | 12 |
| Physical training samples | 120 | 1,000 |
| Inverse evaluations | 60 | 1,000 |
| Stretch ratio | 2 | 5 |
| Print extent | 250 mm | 250 mm |

A higher target preserves source detail and applies conforming four-way subdivision when the next level fits. Actual face counts are shown in the result; the target is an approximate budget. Subdivision increases mapping density without inventing anatomical detail absent from the source OBJ.

STL preview loads the actual exported solid. The shared compute cluster can queue tasks; curved solids generally take longer than planar solids. Numerical geometry and memory guards remain enabled.

Start with the unchanged square/grid-3 structure, perturbation 0.35 and seed 7; map the lung at 48 quads with 2 samples per segment, then export at 2 mm diameter and 250 mm extent. This starter mesh has two closed oriented boundary components. These do not establish two separate material parts or material connectedness; containment has not been qualified. The example exercises geometry and file-format checks, not self-intersection freedom or printing suitability. Different densities, shapes and inverse-design results must pass their own validation. Unsafe geometry and STL coordinate collapse produce a visible error rather than an invalid export.

Validation: `python web_demo/test_showcase_defaults.py` retains dense-heart refinement and pyramid mapping checks and exports the default lung mesh. `python web_demo/test_demo.py` retains the real inverse-result pyramid refusal and verifies the starter STL and route readback.
