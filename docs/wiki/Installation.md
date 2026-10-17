# Installation

## Requirements

- Python 3.9+
- pip

## Install from PyPI

```bash
pip install fibernet          # core
pip install fibernet[full]    # all modules (ML + RL + visualization + simulation)
```

## Install from source

```bash
git clone https://github.com/GellmanSparrowS/fibernet.git
cd fibernet
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Verify installation

```python
import fibernet as fn
print(fn.__version__)
```

## GPU acceleration

The simulation engine uses Taichi for GPU-accelerated dynamics. Taichi is included in `fibernet[full]`, or install separately:

```bash
pip install taichi==1.7.4
```

This checkout temporarily supports TaichiEngine with Taichi 1.7.4 only because
its runtime-lifetime guard uses version-verified private APIs. Other versions
fail explicitly; non-Taichi backends are unaffected. CPU lifecycle validation
does not establish GPU availability or cross-platform support. Request the
backend explicitly with arch="cpu" or arch="gpu"; there is no silent CPU fallback.

Construct and run engines on the same owning thread (a GUI worker is allowed).
Cross-owner use and unknown externally initialized runtimes fail safely. See
[the lifecycle contract](../architecture/taichi-lifecycle-audit.md).

## Troubleshooting

If you see import errors, ensure the full installation is up to date:

```bash
pip install -U fibernet[full]
```
