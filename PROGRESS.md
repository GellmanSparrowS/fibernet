# FiberNet Tutorial Notebook — Task Progress

## Goal
Fix and improve the tutorial notebook (`tutorials/fibernet_v4_tutorial_updated.ipynb`)
Sync to `/media/sf_share/` via `./sync_notebook.sh to_share`

## Latest Update: Phase 12 — CEM RL Improvements ✅

### Phase 12: CEM RL Improvements ✅
- **Elitism**: Always inject best action into top-K pool each generation
- **Momentum**: Mean update uses alpha=0.5 smoothing (prevents oscillation)
- **Adaptive std floor**: Decays over generations (`0.1 * 0.95^gen`) to balance exploration/exploitation
- **Graph reconstruction**: Fixed API (`StructureGraph(dimension=2)`, `add_node(merge=False)`, `add_edge(i,j)`)
- **Figure 11 PNGs renamed** to `_v2_` to force regeneration with improved CEM data
- Commit: `b9346be`

### Phase 11: CEM RL + Energy Fix ✅
- Energy Distribution: forced kJ, x-limit=20, PNG renamed to `_v2_`
- RL upgraded to CEM (Cross-Entropy Method): POP_SIZE=10, TOP_K=3
- Commit: `d0105b3`

## All Phases Completed ✅

### Phase 1-10: Previous work (see git log)
- Phase 1: display(fig) fix
- Phase 2: Skip logic for generation
- Phase 3: Trajectory fix
- Phase 4-5: Batch stats + data path prints
- Phase 6: ML interpretation
- Phase 7: RL overhaul
- Phase 8: Standalone runner script
- Phase 9: Cleanup
- Phase 10: Display fix + RL robustness

## Git History (recent)
```
b9346be Phase 12: CEM RL improvements (elitism, momentum, adaptive std)
d0105b3 Phase 11: CEM RL + energy fix
cfc0510 Phase 10: display fix + RL robustness
```

## CEM RL Test Results

### Phase 11 (basic CEM, 100 episodes test)
```
Gen 1: mean=-100694, best=-71695, std_norm=1.196
Gen 2: mean=-94770,  best=-79661, std_norm=0.857
Gen 3: mean=-107630, best=-66859, std_norm=0.696
5 improvements, std shrinking = convergence
```

### Phase 12 (improved CEM, 100 episodes test)
```
Gen  1: best=-66824, mean=-108350, std_norm=1.168
Gen  5: best=-78853, mean=-96779,  std_norm=0.564
Gen 10: best=-67082, mean=-85992,  std_norm=0.417
Best: -64089 (64.1 kN), 5 improvements, 39.6% force reduction
Final std_norm: 0.417 (good convergence)
```

**Note**: 200 episodes needed for full benefit (user's previous run: 69.1% reduction with 200 eps)

## Library Version: v4.0.5 (on PyPI)

## Usage Notes
- Delete old `07_batch_stats_*.png` (renamed to `_v2_`)
- Delete old `11_rl_reward_*.png` (renamed to `_v2_`)
- Delete old `rl_results.json` to re-run CEM (old random search results won't have cem_mean/cem_std)

## Figure 11 Checklist ✅
- ✅ Panel 1: Reward curve with CEM generation markers
- ✅ Panel 2: Monotonically increasing best reward with improvement points marked
- ✅ 5 structure visualizations saved to `rl_structures/`
- ✅ All improvement structures saved to `rl_improved_structures/` (one JSON per monotonic rise)
- ✅ Graph reconstruction from checkpoint for resume support
