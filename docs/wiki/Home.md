# FiberNet

FiberNet is a Python toolkit for computational design of fiber network metamaterials. It provides a closed-loop workflow from parametric structure generation through GPU-accelerated simulation to machine learning and reinforcement learning optimization.

```
Generation → Simulation → Feature Extraction → Machine Learning → Reinforcement Learning
```

## Documentation

| Module | Description |
|--------|-------------|
| [[Framework Overview]] | Architecture, design philosophy, module relationships |
| [[Unit Types]] | Built-in structural units and their properties |
| [[Simulation Engine]] | Taichi-based mass-spring dynamics |
| [[Feature Extraction]] | 94-dimensional feature vector design |
| [[Machine Learning]] | Prediction, classification, model comparison |
| [[Reinforcement Learning]] | Parametric optimization, CEM, Bayesian methods |

## Quick Links

- **PyPI**: [pypi.org/project/fibernet](https://pypi.org/project/fibernet/)
- **Source**: [github.com/GellmanSparrowS/fibernet](https://github.com/GellmanSparrowS/fibernet)
- **Lab**: [ML-BioMat Lab, BMG-FDU](https://ml-biomat.com/)

## Installation

```
pip install fibernet        # core
pip install fibernet[full]  # all modules
```

## Version

Current release: **v4.0.5** (see [CHANGELOG](https://github.com/GellmanSparrowS/fibernet/blob/main/CHANGELOG.md))
