# FiberNet — 项目进度报告

**最后更新**: 2026-07-09

---

## 当前 API 清单

### 1. Pattern Engine v7 (analysis_scripts/pattern_engine_unified.py)
- **`pattern_2d()`** — 统一polyline基元生成器
- 核心: 自定义点/polygon presets → Cn对称扰动 → mirror_x/mirror_y/rotation → tiling → intersection welding
- 参数: box, points, fit_to_box, polygon_type, n_pts_per_side, perturbation, grid, mirror_x/y, rotation, boundary_mode, stagger
- ⚠️ **未集成到 fibernet/gen/ 正式包**

### 2. Unified Generators (fibernet/gen/unified.py)
- **`lattice_2d(topology)`** — square/triangular/honeycomb/kagome
- **`metamaterial_2d(mode)`** — reentrant/star/arrowhead/chiral/missing_rib
- **`lattice_3d(topology)`** — cubic/octet/diamond
- **`curved_random_2d()`** — 随机曲线纤维
- **`entangled_3d()`** — 3D缠结纤维
- **`biomimetic_network()`** — 仿生网络
- **`hierarchical_lattice()`** — ⚠️ 不是真自相似层级(递归细分+cross-brace)

### 3. Stochastic (fibernet/gen/disordered.py)
- `random_straight_2d/3d`, `oriented_random_2d/3d`, `poisson_line_network_2d`, `random_walk_fibers`, `random_curved_fibers_3d`

### 4. Advanced (fibernet/gen/advanced.py)
- `voronoi_network_2d/3d`, `electrospun_network`, `meltblown_network`
- `biomimetic_collagen/fibrin`, `defected_lattice`, `composite_network`, `graded_network`
- `auxetic_structure`, `kirigami_structure`

### 5. Fractal (fibernet/gen/fractal.py)
- `sierpinski_triangle`, `koch_curve`, `fractal_tree`, `hilbert_curve`

### 6. TPMS (fibernet/gen/tpms.py)
- `tpms_sheet`, `tpms_lattice`, `tpms_gradient`

### 7. Bundles (fibernet/gen/bundles.py)
- `parallel_bundle_2d`, `twisted_bundle_2d`, `random_bundle_3d`, `braided_bundle_3d`, `tendon_like_bundle_3d`

### 8. Curved (fibernet/gen/curved.py)
- `sinusoidal_fiber_2d`(返回Fiber非Network), `helical_fiber_3d`, `arc_fiber_2d`, `bezier_fiber_3d`, `random_curved_network_3d`, `crimped_network_2d`

### 9. Laminates (fibernet/gen/laminates.py)
- `unidirectional_laminate`, `crossply_laminate`, `angle_ply_laminate`, `quasi_isotropic_laminate`, `custom_laminate`, `sandwich_laminate`

### 10. Metamaterials — 旧独立版 (fibernet/gen/metamaterials.py)
- `reentrant_honeycomb_2d/3d`, `chiral_honeycomb_2d`, `star_honeycomb_2d`, `arrowhead_auxetic_2d`
- `hierarchical_lattice_2d`, `missing_rib_auxetic_2d`, `kagome_lattice_2d`
- `proper_octet_truss_3d`, `diamond_lattice_3d`, `gyroid_lattice_3d`, `plate_lattice_3d`

### 11. Woven (fibernet/gen/woven.py)
- `plain_weave_2d`, `twill_weave_2d`, `satin_weave_2d`, `woven_3d_orthogonal`

### 12. Gradient (fibernet/gen/gradient.py)
- `density_gradient_2d`, `property_gradient_2d`, `multi_zone_2d`

### 13. Field-Guided (fibernet/gen/field_guided.py)
- `field_guided_network`, `OrientationField`, `FieldGuidedConfig`

### 14. Specialized (fibernet/gen/specialized.py)
- `cnt_network_2d/3d`, `paper_network`, `textile_weave`, `electrospun_mat`, `fiber_reinforced_composite`

### 15. Variants (fibernet/gen/variants.py)
- `lattice_2d_to_3d`, `curved_lattice`, `multi_radius_network`, `variable_stiffness_network`, `gyroid_infill`, `diamond_lattice_3d`, `foam_like_3d`

### 16. Regular/ZigZag (fibernet/gen/regular.py, zigzag.py)
- `RegularNetworkGenerator` — 仅square(被Pattern Engine取代)
- `ZigZagGenerator` — zigzag折线(被Pattern Engine取代)

---

## 可视化输出 (output_viz/)

| 文件 | 内容 | 有效结构数 |
|---|---|---|
| `01_pattern_engine.png` | Pattern Engine: 自定义形状、polygon、变换、open+extend、grid多样性 | 25 |
| `02_unified_generators.png` | lattice_2d + metamaterial_2d + 旧metamaterials | 25 |
| `03_stochastic_fractal_2d.png` | random/oriented/poisson/walk + voronoi/electrospun/collagen + fractal + gradient | 15 |
| `04_bundles_curved_laminates.png` | bundles + curved + laminates + woven + hierarchical | 20 |
| `05_3d_tpms_field_specialized.png` | 3D lattice/metamaterials/random/entangled + TPMS + field-guided + specialized | 25 |

---

## 已知Bug/问题

1. **Pattern Engine未集成** — 在analysis_scripts/，不在fibernet/gen/
2. **hierarchical_lattice不是真层级** — 当前是"中点细分+cross-brace"，应为"beam→lattice替换"
3. **curved单纤维返回Fiber而非FiberNetwork** — sinusoidal/helical/arc/bezier
4. **voronoi_2d/electrospun/meltblown有解包错误** — `cannot unpack non-iterable int object`
5. **field_guided不支持spiral** — 仅radial/uniform
6. **diamond_lattice_3d crosslinks=0** — 焊接问题
7. **大量冗余API** — 90+导出函数，很多重复

---

## 下一步

1. 用户review可视化 → 决定保留/删除哪些API
2. 集成Pattern Engine到fibernet/gen/pattern.py
3. 重写hierarchical_lattice为真自相似层级
4. 精简冗余API
5. 修复已知bug
