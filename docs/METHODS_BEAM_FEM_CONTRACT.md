# 梁框架 FEM 公开接口与量纲口径

库端 `simulate(..., backend="fem", mode="stretch", strain=λ)` 的 `strain` 参数在当前 API 中表示**目标伸长比** λ，而不是工程应变；例如 λ=1.01 对应名义伸长 1%。二维梁框架节点有水平、竖直和平面转角三个自由度。每条边按自身半径 r 赋予圆截面积 A=πr² 与截面二次矩 I=πr⁴/4；线性解组装平面梁的轴向和弯曲刚度，位移控制在右侧节点施加沿 x 位移 (λ−1) 乘初始 x 跨度，并约束左侧节点。非线性选项使用原实现的增量更新；其具体收敛及反力近似尚需额外验证，不能与实验材料本构等同。

图的独立平行边代表独立纤维。公开拉伸快捷入口、直接梁组装/求解入口及稀疏梁组装/求解入口现在均默认**保留每个边 ID**，把全部边分别组装到刚度矩阵；若用户为了复现旧的简单图结果显式设置 `deduplicate=True`，则按无序端点对只保留一条边，不能用于独立平行纤维的力学分析。两节点同半径双梁的夹持反力和轴向能量均为单梁的两倍，直接及稀疏刚度组装亦给出相同的双边刚度，为该约束提供解析回归。线性 NumPy/SciPy 梁求解无需 PyTorch；仅当输入为 PyTorch 张量时才需该可选依赖。

统一结果中的 `max_force` 是所有已组装梁的**最大单边轴向力绝对值**，由逐梁截面及轴向应变得到，不是左右夹持反力总和。逐梁有符号轴向力和变形后长度/初始长度数组分别保存在 `edge_forces`、`edge_stretches`；`max_stretch`、`mean_stretch` 与标准差从后者计算，压缩边不会因为取应力绝对值而被误记为拉伸。线性解的夹持反力位于 `metadata["reactions"]`，须按边界节点及 x 分量聚合，并检查平衡；当前二维非线性增量路径的该字段为 `None`，不可聚合或视为零，详见下方可用性约束。`energy` 当前只按轴向应力积分近似，结果元数据标为 `axial_stress_proxy`；它不是包括弯曲、接触或损伤的总功。

本地修正前的非线性梁 FEM 方向性 18 案例位于 `benchmarks/results/recruitment_fem_validation.json`，旧最大边力以固定半径 0.05 换算，不能引用其力值。修正后含真实半径与平行边语义的复算结果位于 `benchmarks/results/recruitment_fem_validation_parallel.json`；18/18 个预设阈值下的贯通判定没有变化，而记录的最大单边轴力为旧值四倍。该结果只支持在两种不同数值假设下检查方向性阈值图，不证明简化求解器在定量力值或实验材料上准确。独立平行边、不同半径和压缩伸长的公开回归位于 `tests/test_fem_parallel_contract.py`。

## Start with explicit small-deformation settings

```python
import fibernet as fn

# SI units: coordinates and radius in m; E in Pa; forces in N.
graph = fn.pattern_2d(unit="honeycomb", box=(10, 10), grid=(2, 2), radius=0.05)
result = fn.simulate(
    graph, backend="fem", mode="stretch", stretch_ratio=1.01,
    nonlinear=False, E=1e9, nu=0.3,
)
```

Here `stretch_ratio=1.01` is 1% extension, `nu` is dimensionless, and displacement
is in m. These illustrative parameters are not an experimental calibration.
The shortcut still defaults to a stretch ratio of 2.0 for compatibility.
In 2-D, `nonlinear=None` selects incremental updated-geometry linear solves when
|λ−1|>0.3; explicit `nonlinear=True` selects them at any ratio. Both selections
warn that this is an experimental model with unverified final nonlinear
equilibrium and unavailable final stiffness, reactions and nodal moments.
With `nonlinear=False`, the solver stays linear even at large stretch and warns
when |λ−1|>0.3. The warnings describe the actual branch; they do not change its
numerical results. The legacy 0.3 switch is not a validated small-deformation
limit. A call without a warning still needs an appropriate model and loading.

## Public shortcut dimension selection

`simulate(graph, backend="fem")` selects the existing beam solver from
`graph.dimension` when `dim` is omitted. A planar graph still uses the 2D model,
including when its stored coordinates have a zero z column. A generated 3D
graph uses six nodal degrees of freedom and the existing 3D linear beam solver.
An explicit `dim=2` or `dim=3` remains an override; other values are rejected.
The lower-level `BeamFrameFEM.stretch_test` default is unchanged.

Earlier easy-API versions silently used the lower-level default `dim=2` for
3D inputs. That projected the geometry and could omit beams with zero projected
length. Historical 3D shortcut outputs without an explicit dimension should be
recomputed before interpretation. This is a software model-selection correction,
not evidence of improved experimental agreement. The 3D solver remains linear;
selecting it does not introduce a nonlinear 3D formulation.

## Incremental nonlinear result availability

The 2D `solve_2d_nonlinear` path uses
`incremental_updated_geometry_linear_solves`. This availability guard does not
change its assembly, iteration order, or physical formulation. It does not establish
nonlinear equilibrium. In the raw result, `K`, `reactions`, and `node_moment` are
now `None`: their previous values belonged to the last inner linear increment,
not the accumulated displacement state. They must not be aggregated as final
clamp reactions or interpreted as physical zeros. No original-stiffness-times-total-
displacement replacement is supplied.

`field_status` explicitly describes each unavailable field with
`status="not_supported"`, its model, and these machine-readable reasons:

- `K`: `final_nonlinear_stiffness_not_recovered`
- `reactions`: `final_nonlinear_equilibrium_reactions_not_recovered`
- `node_moment`: `final_total_nodal_moments_not_recovered`

The existing recovery of `sigma_axial`, `sigma_bending`, `sigma_total`, `moments`,
`edge_forces`, and `node_stress` is unchanged but labeled `status="approximate"`
and `approximation="original_geometry_total_displacement_linear_reference"`.
These are original-geometry, total-displacement linear-reference estimates, not
verified nonlinear equilibrium stress/resultants. The unified result's signed
axial `edge_forces` inherits this label; its full `beam_resultants` metadata has
an explicit matching label. The energy remains `energy_kind="axial_stress_proxy"`
and must not be interpreted as a validated nonlinear stored energy.

`solver_status.equilibrium` is always `"unverified"` on this path. Per-step
`update_norm`, `update_threshold`, `update_criterion_met`, `iterations`, and
`iteration_limit_reached` describe only the existing translation-update stop
criterion. A limit is reported when the criterion was not met by `max_iter`;
meeting it on the final allowed iteration does not report a limit stop. Neither
case establishes equilibrium convergence. `update_criterion_is_equilibrium_check`
is false. Raw history's `max_disp` describes accumulated translation, whereas
`max_stress` is from the last inner linear increment; `history_field_scope`
records this distinction without changing historical values.

`to_sim_result` preserves availability and solver status in metadata, along with
null unavailable fields. Both existing summary and full JSON exports retain
these values and node/edge identity maps. Full JSON also carries the displacement
arrays needed for rendering. The renderer skips null reaction metadata and does
not synthesize replacement values. Linear 2D/3D outputs and metadata are
unchanged. Historical raw results and files without these status objects retain
unknown/legacy validity: absence does not certify them or convert old zero arrays
to missing fields.

Consumer limit: the standalone research exporter
`scripts/deformation_test/run_large_deformation_fem.py` safely writes null
reactions, verified using its real `run_fem_2d` function and nonlinear solver.
It does not yet copy `field_status` or `solver_status`, and its force aggregation
uses the approximate recovered edge resultants. Its JSON must therefore not be
used as proof of nonlinear reaction availability or equilibrium. Batch CSV
summaries likewise omit complete solver/availability provenance. These separate
export schema changes are outside this guard. Regression coverage is in
`tests/test_nonlinear_result_availability.py`; it is output-contract coverage,
not nonlinear physics validation.

Nonlinear iteration controls are validated before any solve: `n_steps` and
`max_iter` must be positive integers (including NumPy integer scalars, excluding
booleans), and `tol` must be a finite nonnegative real number (excluding booleans
and strings). Zero tolerance is allowed: the existing strict comparison cannot
be satisfied, so each step exhausts its iteration limit. Invalid controls raise
`ValueError` rather than divide by zero or return an uninitialized last result.
This validation does not add a finite-rotation or constitutive benchmark.

## Direct loads and material scope

Graph-based beam inputs accept straight members between actual graph nodes.
Nonempty edge-owned `internal_points` are rejected before array conversion,
including collinear visualization points. The beam assembly does not use these
points as finite-element nodes. To model a curved centerline with straight
elements, supply actual intermediate graph nodes, their connections, and the
intended joint and material assumptions. This discretization still requires
resolution and mechanical-model checks.

Earlier graph-based stretch calls ignored `internal_points` and solved the
endpoint chord while rendering could show the curve. Results from those inputs
should be recomputed with an explicitly constructed beam mesh.

`BeamFrameFEM.solve_2d` accepts `forces` with exactly `(N, 2)` columns
`Fx, Fy`; `solve_3d` requires `(N, 3)` columns `Fx, Fy, Fz`. Rows follow
`node_pos` array order, not numerical stable node IDs. `None` means zero load.
Applied nodal moments are not supported; extra rotational columns are rejected.
Rotational displacement DOFs and recovered reaction couples do not add a nodal
moment loading API. See the [force-shape contract](architecture/fem-force-shape-contract.md).

This beam backend uses the solver-wide `E` and `nu`, with `G=E/[2(1+nu)]`,
and per-edge radii. The public FEM shortcut defaults to `E=1e9`, `nu=0.3`
unless explicitly supplied. Stored graph `Material` records, including tensors
and unequal per-edge moduli, are not automatically applied as heterogeneous beam
constitutive parameters. Material persistence alone does not imply that capability.
These statements concern `BeamFrameFEM`, not every other backend.

Use consistent length and force units: translations have length units, rotations
are in radians, forces have force units and couples have force-times-length units.
`E` has force/length² units. No automatic geometry or material unit conversion is
performed; entering an SI modulus with geometry numerically in millimetres does
not produce an SI-consistent model without explicit conversion.

For the linear residual/status fields and a bounded usage example, see the
[linear FEM diagnostics usage guide](architecture/fem-linear-diagnostics-usage.md).
