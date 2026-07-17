# FiberNet Tutorial Notebook — Task Progress

## Goal
Fix and improve the tutorial notebook (`/media/sf_share/fibernet_v4_tutorial_updated.ipynb`)

## Task List

### Phase 1: Fix plt inline display in Jupyter
- **Status**: pending
- **Problem**: `plt.show()` is unreliable in Jupyter inline mode; figures don't appear
- **Solution**: Use `IPython.display.display(fig)` instead of `plt.show()`
- **Files**: notebook cells 9,11,14,18,23,25,27,31,34,36,40

### Phase 2: Add skip logic for batch generation
- **Status**: pending
- **Cells**: Cell 15 (structure generation), Cell 19 (JSON save)
- **Logic**: Check if structures already exist in memory/JSON, skip if so

### Phase 3: Fix trajectory Cell 23
- **Status**: pending
- **Logic**: If checkpoint JSON exists with trajectory, load and render. Otherwise re-run one sample.

### Phase 4: Fix 07_batch_stats
- **Status**: pending
- **Changes**:
  - Replace ForceByStructure (too many Y-axis labels) with structure-property scatter or something meaningful
  - Fix Energy x-axis range (data mostly within 20kJ)

### Phase 5: Add data path prints for all figure cells
- **Status**: pending
- **Logic**: Every figure cell should print where data/files are saved

### Phase 6: ML interpretation/explanation print
- **Status**: pending
- **Logic**: After ML analysis, print human-readable interpretation of results

### Phase 7: Overhaul RL section
- **Status**: pending
- **Changes**:
  - Analyze why rewards are all negative (metric issue?)
  - Fig 11: only 2 panels — reward curve + monotonically increasing best-reward curve
  - Save 5 structure visualizations from the process
  - Save structures at each reward increase point to a separate folder
  - Print analysis of what the RL data means

### Phase 8: Build standalone Python runner script
- **Status**: pending
- **Requirements**: Checkpoint/resume, memory limits, import-friendly

### Phase 9: Final verification + cleanup
- **Status**: pending

## Git Checkpoints
- `0a81aa9` — v4.0.5 with field cache + kernel cache + vectorized save
