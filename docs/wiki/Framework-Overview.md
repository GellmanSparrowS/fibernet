# Framework Overview

FiberNet follows a modular architecture where each component can be used independently or composed into a full pipeline.

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
│     gen/     │───▶│    sim/      │───▶│    analysis/     │
│  (structure  │    │  (Taichi     │    │  (feature        │
│  generation) │    │  dynamics)   │    │   extraction)    │
└─────────────┘    └──────────────┘    └──────────────────┘
                                              │
                          ┌───────────────────┼───────────────────┐
                          ▼                   ▼                   ▼
                   ┌────────────┐    ┌──────────────┐    ┌──────────────┐
                   │    ml/     │    │     rl/      │    │    viz/      │
                   │ (learning) │    │ (optimization│    │ (rendering)  │
                   │            │    │   & control) │    │              │
                   └────────────┘    └──────────────┘    └──────────────┘
```

## Core Data Model

All modules operate on a shared `StructureGraph` object:

- **Nodes**: positions, optional attributes (mass, boundary flag)
- **Edges**: connectivity, optional attributes (stiffness, rest length)
- **Metadata**: dimension (2D/3D), box size, grid info, unit type

## Module Responsibilities

### gen/ — Structure Generation
Generates `StructureGraph` instances from unit type definitions. Supports parametric displacement control for RL action spaces.

### sim/ — Simulation Engine
Runs mass-spring dynamics on `StructureGraph`. Produces `SimResult` with forces, stretches, trajectories, and energy.

### analysis/ — Feature Extraction
Extracts numerical feature vectors from `StructureGraph` for downstream ML/RL tasks.

### ml/ — Machine Learning
Provides regression, classification, cross-validation, and automated pipelines on extracted features.

### rl/ — Reinforcement Learning
Defines parametric environments and optimization methods (CEM, Bayesian optimization).

### viz/ — Visualization
Renders structures, trajectories, stress distributions, and analysis results.

## Design Principles

1. **Composability**: Each module accepts and returns standard data structures
2. **Minimal coupling**: Modules communicate through `StructureGraph` and `SimResult`
3. **Progressive disclosure**: Simple one-line API for common tasks, full control when needed
4. **Performance**: Taichi GPU backend for simulation, cached field allocation
5. **Reproducibility**: Deterministic generation with seed control

## Data Flow

```python
# 1. Generate
g = fn.pattern_2d(unit="square", grid=(3,3), n_pts_per_side=5, ...)

# 2. Simulate
r = fn.TaichiEngine().stretch_test(g, target_stretch=1.5, ...)

# 3. Extract features
features = fn.GraphFeatureExtractor().extract(g)

# 4. Learn / Optimize
model, metrics = fn.ml.train_predictor(X, y)
# or
result = fn.rl.run_bayesian_optimization(objective_fn, param_space)
```
