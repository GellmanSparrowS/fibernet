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
| Surface | Pyramid | Seven desktop models or OBJ import |
| Target surface quads | 1,000 | 10,000 |
| Samples per fiber segment | 5 | 10 |
| Planar grid per direction | 3 | 12 |
| Physical training samples | 120 | 1,000 |
| Inverse evaluations | 60 | 1,000 |
| Stretch ratio | 2 | 5 |
| Print extent | 250 mm | 250 mm |

A higher target preserves source detail and applies conforming four-way subdivision when the next level fits. Actual face counts are shown in the result; the target is an approximate budget. Subdivision increases mapping density without inventing anatomical detail absent from the source OBJ.

STL preview loads the actual exported solid. The shared compute cluster can queue tasks; curved solids generally take longer than planar solids. Numerical geometry and memory guards remain enabled.

Validation: `python web_demo/test_showcase_defaults.py` checks actual dense-heart refinement and exports the default pyramid solid.
