#!/usr/bin/env python3
"""
FiberNet End-to-End Pipeline: 100 Voronoi Structures

Steps:
1. Generate 100 voronoi structures with varying parameters
2. Run mass-spring stretch simulation (1.5x)
3. Extract structural features (94-dim)
4. Output CSV with all results

Usage:
    python3 analysis_scripts/pipeline_voronoi_100.py
    python3 analysis_scripts/pipeline_voronoi_100.py --resume
    python3 analysis_scripts/pipeline_voronoi_100.py --count 50  # fewer for quick test
"""
import sys, os, json, time, csv, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')

from fibernet import pattern_2d, TaichiEngine, GraphFeatureExtractor

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTPUT_DIR / "voronoi_100_results.csv"
CHECKPOINT = OUTPUT_DIR / "_pipeline_checkpoint.json"


def load_checkpoint():
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text())
    return {"completed": []}


def save_checkpoint(state):
    CHECKPOINT.write_text(json.dumps(state, indent=2))


def generate_configs(count: int):
    """Generate diverse voronoi configurations."""
    np.random.seed(42)
    configs = []
    for i in range(count):
        grid_x = np.random.randint(2, 6)
        grid_y = np.random.randint(2, 6)
        n_internal = np.random.randint(5, 25)
        seed = i * 7 + 13  # different seeds
        configs.append({
            "id": i,
            "unit": "voronoi",
            "grid_x": int(grid_x),
            "grid_y": int(grid_y),
            "n_internal": int(n_internal),
            "seed": int(seed),
        })
    return configs


def run_pipeline(count: int, resume: bool):
    """Main pipeline: generate → simulate → extract → CSV."""
    configs = generate_configs(count)
    state = load_checkpoint() if resume else {"completed": []}
    completed_ids = set(state["completed"])

    engine = TaichiEngine()
    extractor = GraphFeatureExtractor(canvas_size=128)  # smaller for speed

    rows = []
    # Load existing rows if resuming
    if CSV_PATH.exists() and resume:
        with open(CSV_PATH, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            print(f"Loaded {len(rows)} existing rows from {CSV_PATH}")

    print(f"Pipeline: {count} voronoi structures, {len(completed_ids)} already done")
    print(f"Output: {CSV_PATH}")
    print()

    fieldnames = None
    total_time = 0

    for i, cfg in enumerate(configs):
        if cfg["id"] in completed_ids:
            continue

        t0 = time.time()
        print(f"[{i+1}/{count}] Voronoi grid=({cfg['grid_x']},{cfg['grid_y']}) "
              f"n_internal={cfg['n_internal']} seed={cfg['seed']}")

        try:
            # 1. Generate structure
            g = pattern_2d(
                unit="voronoi",
                box=(10, 10),
                grid=(cfg["grid_x"], cfg["grid_y"]),
                seed=cfg["seed"],
                n_internal=cfg["n_internal"],
            )

            # 2. Simulate (1.5x stretch, auto steps)
            result = engine.stretch_test(
                g, target_stretch=1.5, stiffness=1e5, damping=0.3,
                auto_steps=False, num_steps=8000, save_interval=10000,
            )

            # 3. Extract features
            features = extractor.extract(g)

            # 4. Build row
            row = {
                "id": cfg["id"],
                "grid_x": cfg["grid_x"],
                "grid_y": cfg["grid_y"],
                "n_internal": cfg["n_internal"],
                "seed": cfg["seed"],
                "n_nodes": g.num_nodes,
                "n_edges": g.num_edges,
                "target_stretch": 1.5,
                "max_force": result.max_force,
                "max_stretch": result.max_stretch,
                "mean_stretch": result.mean_stretch,
                "std_stretch": result.std_stretch,
                "max_displacement": result.max_displacement,
                "energy": result.energy,
                "sim_time": result.time_seconds,
            }
            # Add all feature columns
            for k, v in features.items():
                row[f"feat_{k}"] = v

            rows.append(row)
            if fieldnames is None:
                fieldnames = list(row.keys())

            dt = time.time() - t0
            total_time += dt
            print(f"  ✓ {g.num_nodes} nodes, {g.num_edges} edges, "
                  f"max_force={result.max_force:.1f}, "
                  f"max_stretch={result.max_stretch:.2f}, "
                  f"sim={result.time_seconds:.1f}s, total={dt:.1f}s")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback; traceback.print_exc()
            continue

        # Checkpoint
        state["completed"].append(cfg["id"])
        save_checkpoint(state)

        # Write CSV every 5 items
        if len(rows) % 5 == 0 or i == count - 1:
            with open(CSV_PATH, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    # Final write
    if rows and fieldnames:
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"\n{'='*60}")
    print(f"Done: {len(rows)} structures processed")
    print(f"Total time: {total_time:.0f}s ({total_time/max(1,len(rows)):.1f}s avg)")
    print(f"Output: {CSV_PATH}")
    print(f"{'='*60}")

    # Summary stats
    if rows:
        forces = [float(r["max_force"]) for r in rows]
        stretches = [float(r["max_stretch"]) for r in rows]
        print(f"\nSummary:")
        print(f"  max_force: mean={np.mean(forces):.1f}, std={np.std(forces):.1f}, "
              f"range=[{min(forces):.1f}, {max(forces):.1f}]")
        print(f"  max_stretch: mean={np.mean(stretches):.2f}, std={np.std(stretches):.2f}, "
              f"range=[{min(stretches):.2f}, {max(stretches):.2f}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    run_pipeline(args.count, args.resume)


if __name__ == "__main__":
    main()
