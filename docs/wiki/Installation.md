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
pip install taichi
```

Taichi automatically detects available GPUs (CUDA, Metal, Vulkan).

## Troubleshooting

If you see import errors, ensure the full installation is up to date:

```bash
pip install -U fibernet[full]
```
