# FiberScope 2.6 extension interfaces

## 2.6 topology and workflow contract

`StructureFactory(topology='topnet26')` defaults to the shared analysis/manufacturing graph. Explicit `topology='legacy'` reproduces historical generators. Graph metadata includes `topology_id`, `route_nodes`, and `route_edges`; see `docs/METHODS_2_6.md`.

`PrintWorkflow.start(samples=10, iterations=20, curved=False, open_slicer=True)` starts eleven asynchronous stages. `status()` returns `busy`, `state`, per-stage states, options, output folder and `printer_state`; `cancel()` cooperatively stops. AI tools: `start_print_workflow`, `get_print_workflow_status`, `stop_print_workflow`. Preparation completion never asserts slicing or physical print completion. Future device backends must require an explicit printer/material/profile and distinguish submission from hardware completion; this release has no device transport.

`load_obj(path, return_info=True, target_faces=1500)` handles negative indices, bounded polygons, dense mixed faces and quad conversion. `assets/obj/reference26.json` records compact model provenance.

## Structure and periodic rules

`fslab.structure.StructureFactory` is the shared serializable specification. Use
`dataclasses.replace(factory, ...)` to preserve the caller's topology, grid,
radius, seed and expansion rule. `build()` returns a StructureGraph. `key()`
includes the entire specification and custom cell geometry.

`expansion_rule`: `translate`, `rotate`, `mirror`, `mirror_rows`, `quarter_turn`,
`checker_mirror`, `custom`. A custom rule is a rectangular list of 1–4 rows and
1–4 columns. Each entry is `{"turn": 0..3, "flip": false|true}`. Reflection in
local x precedes counterclockwise quarter turns; the table repeats by row and
column. Expressions and executable Python are not accepted as user rules.

```python
from fslab.structure import StructureFactory
factory = StructureFactory(unit='octagon', expansion_rule='custom',
    custom_rule=[[{'turn': 0, 'flip': False}, {'turn': 1, 'flip': True}]])
graph = factory.build()
```

## Supervised models

`fslab.learning.Regressor(key, params, seed).train(X, Y, ...)` returns loss
history and held-out count. `predict(X)` returns three columns: peak force,
early-curve slope, integrated force–strain area. Inputs are exactly the 14
columns in `FEATURE_NAMES`, schema `structure14-v3-topnet26`.

`register_model(key, (zh, en), parameters, factory)` adds a model adapter.
`factory(parameters, seed)` returns an estimator with `fit(X,Y)` and
`predict(X)`. A parameter descriptor is `(default, minimum, maximum, zh, en)`;
an integer default selects an integer UI control. The dialog reads the same
registry as execution. New models must support multi-output regression or
provide a wrapper. Training normalizes only training rows and scores held-out
rows; do not pass predicted labels off as physical observations.

Built-ins: shallow MLP, 2–5-layer MLP, Ridge, nearest neighbors, random forest,
extra trees. Neural models use epoch callbacks and early stopping. Forests report tree counts in increments of ten and retain the full requested forest. Ridge/nearest-neighbor fits report one completed fit. Default mlp restores the original NumPy tanh implementation.

## Acquisition and datasets

`register_acquisition(key, names, select)` adds a selector with signature
`select(pool_X, labelled_X, labelled_Y, numpy_rng) -> pool_index`.
Built-ins are random, max-min diversity, bootstrap ridge committee disagreement,
and equal-weight normalized disagreement/diversity. A committee requires eight
finite labels; otherwise its explicit fallback is diversity.

`DatasetStream(config, path, model, pool_size=24).generate(total_n, progress_cb,
stop_cb)` requests a **total** size, rather than appending that many rows on
each restart. Modes: `generate` writes NaN labels; `physics` runs Engine2;
`surrogate` writes model predictions and requires a trained model. The UI and AI learning tools expose only physics mode: generation includes physical labels. Other modes are legacy low-level storage APIs. A dataset stores X, Y, JSON metadata,
and JSON factory specifications in a non-pickle NPZ. Configuration and feature
schema must match before resuming. Each checkpoint uses a same-directory
temporary file and atomic replacement. Candidate seeds are committed even
when candidates are not selected. Prediction jobs use a fresh dataset path.

Source cache: `_cache`; the EXE prefers `%LOCALAPPDATA%/FiberScope/datasets`. Directories are probed for atomic creation and replacement; fallback candidates are the user .fiberscope directory, a writable application cache and the system temporary directory.
Limits: 2,000 API rows, 200 UI rows, 128 candidate pool rows, one physics graph
evaluated at a time. Requests above these limits are rejected.

## Inverse design and reinforcement learning

`run_inverse(builder, target, budget, ..., initial_spec=None, evaluator=None, amplitude=.6)`
uses the authoring spectrum and the independently selected inverse topology. The callback receives (record, current_run) on every evaluation; compare record.dist to record.best_dist to detect improvements. `builder(unit,
perturbation, spectrum)` builds a graph. Optional `evaluator(graph, target)`
returns `(objective_to_minimize, run_or_None)`. This is the interface for a
future surrogate, laboratory callback or another solver. A scalar surrogate
does not automatically support full force-curve objectives. The built-in GUI
uses physical evaluation, with CEM as its default search.

`register_search(key, names, parameters, runner)` extends the search registry.
`runner(objective, initial_vector, budget, seed, parameters, checkpoint,
stop_cb)` returns best_x, best_value and evaluations. Learned strategies use a
bounded vector of tangent/normal spectrum offsets plus perturbation. DQN has
discrete coordinate-increment actions; other built-ins have continuous actions.
PPO, A2C, DQN, SAC, TD3 and DDPG are actual Stable-Baselines3 implementations.

Custom search registrations must also be imported by the worker process; registrations made only in the GUI are not serialized across the process boundary. Rebuild the bundled worker when shipping a new plugin.

The optional training process runs `scripts/rl_worker.py request.json` and emits
JSON records on stdout. Results, replay arrays and checkpoints remain in the
job directory. `FIBERSCOPE_TRAIN_PYTHON` selects an interpreter with the optional
requirements installed. The source version defaults to its interpreter; the
EXE also recognizes a local `anaconda3/envs/ml310` or `miniconda3/envs/ml310`.
`FIBERSCOPE_TRAINING_DIR` overrides the preferred LocalAppData training directory; the same verified fallback policy applies.
The EXE carries the worker source and light structure package, not PyTorch.

CPU threads are limited to one; networks have two 32-unit layers; replay buffers
hold at most 10,000 transitions. STOP requests are observed between mechanical
evaluations. run_external(..., amplitude=.6, resume=False) starts a fresh job. Explicit resume=True continues the most recent matching policy, best state and off-policy replay buffer. A progress event contains a record and an atomic candidate snapshot that the parent consumes and removes. Checkpoints are local trusted artifacts. Resumption
is a warm restart; random trajectories need not match uninterrupted execution
bit for bit. A short demonstration budget is not evidence of a converged policy.

## Surface mapping

`surface_mapping.map_cells(V, quad_faces, spectrum, unit, MappingConfig)`
returns fiber points and segment indices, with no support-mesh edges. The
default overscale is 1.06. Each shared patch boundary is stitched through its
midpoint to the nearest fiber point in each adjacent patch. This explicitly
adds connections; it does not infer adhesion from projected overlap. See
Methods for the normalization and bilinear map. `load_obj` accepts polygon and
negative OBJ indices. Non-quad polygon meshes are subdivided into corner quads.
Dense triangle imports can use load_obj(path, return_info=True, target_faces=1500). It returns vertices, quads and conversion counts. Mapping may coarsen patches to respect the point/segment budget; coarsening can drop tiny disconnected accessories while retaining components covering at least 95% of area. The UI reports the reduction. Imported geometry is never rewritten.

## AI tools and task states

Tool definitions and implementation are in `studio/ai_assistant.py`. Semantic
page names remain stable if tabs move. New tools include `configure_learning`,
`generate_learning_data`, `get_learning_status`, `configure_search`, and
`get_design_status`. Learning generation always includes physical simulation. Learning and inverse
jobs return `state=started`; this is not a completed result. Query status after
completion. Existing direct simulation tools still return their simulated
results. API credentials remain in the user's private configuration and must
not be included when publishing the repository.

## Validation entry points

`python scripts/run_tests.py` runs numerical, model, acquisition, checkpoint,
surface and GUI suites. `python scripts/build_exe.py` additionally runs excluded
dependency checks, frozen smoke, DLL portability audit and a clean-room smoke.
`python scripts/update_code_registry.py` refreshes `files_registry.json`.

Implementations: [scikit-learn](https://github.com/scikit-learn/scikit-learn),
[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).

## 2.4 evaluator and authoring additions

- `fslab.model_inverse.run_model_inverse(factory, model, target, budget=40, seed=7, amplitude=.6, callback=None, stop_cb=None)`: CEM with the trained scalar predictor; `target` must be in `SCALARS`.
- `PredictedStructure(positions, edges, prediction)`: static candidate payload. Callback consumers must branch by payload type and must not interpret it as a physical `Run`.
- `fslab.rl_process.run_external(..., model=None)`: optional trusted in-process model snapshot; default `None` preserves physical evaluation. Model results contain `evaluation_mode='surrogate'`, `best_preview`, `best_prediction`, and `best_run=None`.
- `run_inverse` results include cumulative `evaluations`. Records retain the latest 2,000 entries; export clients must disclose this window.
- `studio.count_input.CountInput`: positive integer `value()/setValue()` control without an application-level upper bound.
- Cell vertices may extend outside [0,1]^2; this square is the repeat period. Geometry validation and node budgets still apply.
- `studio.rule_dialog.ExpansionDialog` exposes built-in rules and optional custom period editing within the cell workbench.
- `AUTHOR_SIGNATURE` is centralized in `fslab/version.py`; hidden UI and executable metadata use that value.

The model snapshot is private local worker transport. There is no public pickle import endpoint. See [2.4 methods](../METHODS_2_4.md) for labels, checkpoint identity and display semantics.

## 2.5 continuous-fabrication interfaces

- `compile_planar(factory) -> ManufacturingNetwork` in `fslab.manufacturing`: closed-cell multigraph, independent shared/boundary fibers and stable route.
- `compile_surface(vertices, quad_faces, factory, overscale=1.06)`: fixed reference attachments and paired seam connectors; independent of overlap during deformation.
- `ManufacturingNetwork` provides `reference`, `positions`, `edges`, `route_nodes`, `route_edges`, `topology_id`, `health`, `added_edges`, and `bridges`. Route edge IDs are a permutation of every edge index; the first and last route nodes coincide. Coincident edges must not be deduplicated by consumers.
- `PrintSettings(width=100, depth=100, height=2, diameter=2, curved=False, max_voxels=8000000)`: millimetre fabrication configuration. Curved mode keeps aspect ratio and uses diameter as thickness.
- `build_solid(network, settings=None, progress=None, stop_cb=None) -> FiberSolid`: bounded worker-friendly distance-field tube union. `FiberSolid.centers` and mesh vertices share the final physical coordinate transform.
- `export_solid(path, solid)`: atomic closed STL or unit-declared 3MF. `export_route(path, network, solid.centers)`: atomic CSV retaining edge IDs and millimetres.
- `find_bambu()` and `open_in_bambu(model_path, executable=None)`: local executable discovery and shell-free model opening. No printer-specific slicing settings or printer-job submission are assumed.
- `ManufacturingDialog(factory, surface=None, mode='dark', parent=None)`: isolated asynchronous workspace; surface is `(vertices, faces, overscale)`. Closing requests cooperative worker cancellation.

Since 2.6, new `StructureFactory.build()` calls use the shared manufacturing topology; historical datasets remain isolated by topology/schema. Explicit legacy mode preserves old generators. See [current methods](../METHODS_2_6.md) for topology/process distinctions and [2.5 export methods](../METHODS_2_5.md) for solid approximation and budgets.


## 2.7 geometry and print interfaces

- `compile_planar(factory, boundary_twins=True)` applies seeded jitter to all generated planar nodes. Surface compilation disables local boundary mirroring and shares known mesh corners.
- `PrintSettings(..., max_height=250, up_axis='z')` retains existing positional arguments; `up_axis='y'` rotates Y-up assets to printable Z-up. Curved dimensions preserve aspect ratio and fit a 250 mm cube.
- `build_solid` now computes polygonal cylinder and joint-sphere Boolean unions using manifold3d. `build_voxel_solid` retains the historical distance-field implementation for explicit use. Cancellation is checked between batches and after the final native union.
- `ManufacturingWorker(..., network=None)` / `ManufacturingDialog(..., network=None)` accept the exact mapped graph. A supplied graph is not recompiled.
- Large display meshes may be simplified to 100000 triangles; export uses the full closed mesh. `FiberSolid.resolution` represents the cylinder chord approximation, not a global surface error bound.
- `start_print_workflow`, `get_print_workflow_status`, `stop_print_workflow` remain internal AI tools. Local text “准备打印” or “完整流程” starts the curved workflow; “停止流程” cancels it. No main-panel workflow button is shown.
- Learning cache schema: `structure14-v4-geometry27`. Default model and acquisition choices are unchanged.
- `find_bambu()` now checks common directories across Windows drive letters, including `D:/Bambu/Bambu Studio/bambu-studio.exe`.

See [2.7 methods](../METHODS_2_7.md). Local Bambu validation follows its [official CLI interface](https://github.com/bambulab/BambuStudio/wiki/Command-Line-Usage); opening or converting a model does not send a printer job.


## 2.8 接口增补

- RunConfig.weld_intersections=False：初始二维中心线交点焊接；拆边保留纤维身份，缓存与未焊接结果隔离。实现：fslab/welding.py。
- SurfaceTab 的每个 OBJ 可设置 target_faces（12–10000）及 subdivision_levels（0–3），细分输出受 10000 面预算约束。四边面不保证严格正方形；超预算明确报错。
- DesignTab.follow_type 默认开启；model_btn 为默认关闭的模型加速复选框。模型来自 MLTab.trained_model / trained_unit，固定训练结构类型。现有模型预测标量，J/C 曲线目标使用物理仿真。
- PrintWorkflow.start(samples=10, iterations=20, curved=False, open_slicer=True, target='C', stretch=2.0, surface_model=None, create_demo=False)。录像预设使用 300 样本、60 次预算、J、金字塔。
- AI 本地指令前缀“请执行决赛录像流程”无需联网。生成学习数据支持 10–2000 个真实物理标注样本；set_surface 默认跟随当前结构。
- open_in_bambu 仅接受 STL/OBJ。独立导出仍可保存 3MF，但不作为拓竹交接格式。
- 详见 docs/METHODS_2_8.md 和 docs/finals/AI_RECORDING_2_8.md。


## 2.9 接口更新

- StructureFactory.effective_spectrum()：返回按seed和perturbation比例扰动后的共享位移谱；零值保持零。
- fslab.solid_process.build_isolated(network, settings, progress=None, stop_cb=None)：隔离进程、可取消、600秒超时，返回FiberSolid；内部入口run.py --solid-worker。
- MainWindow.tab_manufacturing：位于曲面之后。open_source(curved)选取当前网络，mount(dialog)可承接工作流成品。
- AI set_tab增加manufacturing；离线“打开制造”进入制造工作区。
- SimTab._export_gif(silent=None)：异步1600×1000 GIF，_gif.active表示进行中，cancel()中止；平均48FPS。
- 界面默认焊接开启、接触关闭，RunConfig历史默认保留；模型训练标签使用其自身显式配置。
- 构建增加Pillow12.2.0，许可证已在assets/licenses/pillow。细节及限制见docs/METHODS_2_9.md。

## 3.0 additions

StructureFactory.spectrum_resolved marks already-resolved editor values. Dataset configuration includes engine_source. Recording options are parsed by studio.recording_command.recording_options. Workflow.start additionally accepts demo_unit, diameter and print_size; the finals command supplies square, 2 mm and 250 mm. Manufacturing readiness_changed(bool) drives the host route-playback button.
