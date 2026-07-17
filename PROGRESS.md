# FiberNet Tutorial Notebook — Task Progress

## Goal
Fix and improve the tutorial notebook (`tutorials/fibernet_v4_tutorial_updated.ipynb`)
Sync to `/media/sf_share/` via `./sync_notebook.sh to_share`

## Completed

### Phase 1: Fix plt inline display in Jupyter ✅
- Replaced `plt.show()` with `display(fig)` (from IPython.display) in all 11 viz cells
- `display(fig)` is reliable in Jupyter inline mode; `plt.show()` was not
- `plt.close(fig)` called after display for all themes (prevents memory leak)
- Commit: `1ebadd4`

### Phase 2: Add skip logic for batch generation ✅
- Cell 16: Check if `all_structures` already exists in memory with correct count
- Cell 20: Check if first JSON file exists, skip entire save loop if so
- Uses compact JSON format (no indent) for faster I/O
- Commit: `5985c32`

### Phase 3: Fix trajectory Cell 23 ✅
- Check if trajectory PNGs exist → skip entirely
- Otherwise re-run one structure with save_interval=500 to get trajectory
- Vectorized edge computation preserved
- display(fig) for dark theme
- Commit: `12239cf`

## In Progress

### Phase 4: Fix 07_batch_stats (Cell 27)
- **Status**: pending
- Replace ForceByStructure (too many Y-axis labels) with meaningful alternative
- Fix Energy x-axis range (data mostly within 20kJ)
- Add data path prints

## Pending

### Phase 5: Add data path prints for all figure cells

### Phase 6: ML interpretation/explanation print

### Phase 7: Overhaul RL section
- Analyze negative rewards (metric issue?)
- Fig 11: 2 panels only — reward curve + monotonically increasing best-reward curve
- Save 5 structure visualizations from the process
- Save structures at each reward increase point to a separate folder
- Print analysis

### Phase 8: Build standalone Python runner script with checkpoint support

### Phase 9: Final verification + cleanup

## Git Checkpoints
- `0a81aa9` — v4.0.5 with field cache + kernel cache + vectorized save
- `1ebadd4` — notebook in repo + display(fig) fix
- `5985c32` — skip logic for generation and JSON save
- `12239cf` — trajectory PNG skip + re-run fallback
