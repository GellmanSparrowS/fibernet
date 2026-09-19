# FiberScope web demo

Run from the repository root:

~~~bash
python -m pip install -r web_demo/requirements.txt
python web_demo/app.py
~~~

Open http://localhost:7860. The seven tabs call the same fslab numerical core as the desktop software. ML labels are computed from physical simulations; held-out scores can be negative for small datasets. The web queue and bounds protect shared CPU memory; use the desktop for larger jobs. Models and work files are session-specific. API keys are neither required nor embedded.

Test: python web_demo/test_demo.py

The ModelScope deployment entry imports create_app from this module and serves port 7860. Free hardware is used. Deployment secrets must remain outside the repository.
