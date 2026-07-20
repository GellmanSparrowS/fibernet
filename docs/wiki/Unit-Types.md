# Unit Types

FiberNet provides built-in unit cell types organized into structural families. Each unit can be tiled into larger structures via `pattern_2d()` or `pattern_3d()`.

## 2D Units

| Family | Units | Characteristics |
|--------|-------|-----------------|
| **Regular Lattices** | `square`, `triangle`, `hexagon` | Classic periodic tessellations |
| **Cellular** | `honeycomb`, `reentrant`, `missing_rib` | Auxetic and cellular solid models |
| **Chiral/Auxetic** | `chiral`, `star` | Negative Poisson's ratio structures |
| **Braced** | `cross`, `diamond` | Cross-braced and diamond patterns |
| **Tri-hexagonal** | `kagome` | Kagome lattice |
| **Disordered** | `voronoi` | Random Voronoi tessellation |

### Querying Available Units

```python
fn.list_units()
# ['chiral', 'cross', 'diamond', 'hexagon', 'honeycomb', 'kagome',
#  'missing_rib', 'reentrant', 'square', 'star', 'triangle', 'voronoi']
```

## 3D Units

| Family | Units | Characteristics |
|--------|-------|-----------------|
| **Cubic Lattices** | `cubic`, `bcc`, `fcc` | Body/face-centered cubic |
| **Close-packed** | `hcp` | Hexagonal close-packed |
| **TPMS** | `gyroid`, `schwarz_p`, `schwarz_d`, `iwp`, `neovius`, `lidinoid` | Triply periodic minimal surfaces |
| **Truss** | `octet`, `diamond_3d` | Truss-based lattices |
| **Auxetic 3D** | `chiral_3d`, `reentrant_3d` | 3D negative Poisson's ratio |

```python
fn.list_units_3d()
# ['bcc', 'chiral_3d', 'cubic', 'diamond_3d', 'fcc', 'gyroid',
#  'hcp', 'iwp', 'lidinoid', 'neovius', 'octet', 'reentrant_3d',
#  'schwarz_d', 'schwarz_p']
```

## Parametric Control

Each unit edge supports `n_pts_per_side` internal nodes with programmable `(dx, dy)` displacement:

```python
g = fn.pattern_2d(
    unit="square",
    grid=(3, 3),
    n_pts_per_side=5,           # 4 edges x 5 pts = 20 displacement pairs
    point_displacements=disps,  # [(dx, dy), ...]
)
```

This parametric interface provides a continuous action space for reinforcement learning, equivalent to `move_AB(G, num, dx, dy)` in research code.

## Structure Generation API

```python
# 2D
g = fn.pattern_2d(
    unit="honeycomb",
    box=(10, 10),       # cell dimensions
    grid=(4, 4),        # tiling grid
    seed=42,            # reproducibility
)

# 3D
g = fn.pattern_3d(
    unit="gyroid",
    box=(10, 10, 10),
    grid=(2, 2, 2),
)
```

## Extending with Custom Units

Custom unit types can be registered via the unit factory system. See the source code in `fibernet/gen/` for the unit registration pattern.
