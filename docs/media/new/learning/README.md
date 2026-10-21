# Recorded graph learning and environment actions

Explore conductance learning across material-assigned networks and prescribed actions in the FiberNet design environment. Both animations visualize recorded results.

## GINE research example

`media/gine_training_history.gif` reveals a saved 40-epoch GINE training history with a fixed solved electrical field and final predictions. The synthetic data comprise 48 networks from 12 parent geometries. Development uses 24 training and 8 validation specimens; 16 final specimens share 4 untouched parents. Split by parent, not material variant. Training uses AdamW with seed 42 over 40 epochs, selecting epoch 38. Final RMSE is 0.480250054 mS and R² is 0.83084875. Placement-contrast signs are wrong for 3 of 4 final parents, and the trial-energy baseline wins both uniform-material strata.

The field panel is a static recorded solve, not a prediction at each training epoch. Replaying the learning curve does not re-evaluate the final set. The archival source tree is `8fabc86ee457c1002bab48544ef7b53f8737d185`; it used a research trainer not shipped in current main. Current `train_gnn` does not accept `GINERegressor`. This media bundle supplies a saved-data rendering recipe, not a supported training-replay command or runtime upgrade.

## RL environment action records

`media/rl_environment_fixed_actions.gif` shows three discrete archived stages for three networks: reset geometry, prescribed action geometry, native transient response with recorded reward and termination. Each case called the environment once. Prescribed actions; no trained policy. Discrete states preserve the recorded samples. The nondimensional axial model uses unit nodal mass and global spring constant 1000 with fixed horizon 0.02. These are finite-horizon transient responses rather than equilibrium solutions.

## Reproduce the figures

With NumPy, Matplotlib and Pillow available, run `python render_gine_history.py` and `python render_rl_step.py`. The scripts read only the supplied JSON/NPZ files, save GIF, PNG and editable SVG under `media`, and do not import FiberNet or optional torch/Gym packages. NPZ uses `allow_pickle=False`. `DATA_PROVENANCE.json` lists exact source archive hashes, data projection rules and all payload hashes. Raw GINE history is retained; RL data are a lossless projection of the plotted coordinates, stable IDs, actions, rewards and terminal flags. Full recovery archives are not runtime dependencies.

FiberNet material is under the included MIT license. GINE's original operator was upstream PyTorch Geometric, not copied in this bundle. This archive includes neither a model executable nor third-party dependency binaries.
