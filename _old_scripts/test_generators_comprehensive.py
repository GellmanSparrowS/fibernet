"""
Comprehensive generator test - memory safe, data-only analysis.
Tests all registered generators via fn.create() and saves results to JSON.
"""
import sys, gc, time, json, traceback
import numpy as np
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import fibernet as fn
from fibernet.core.network import FiberNetwork

OUT_FILE = Path(__file__).parent / "analysis_results" / "test_results_v2.json"
OUT_FILE.parent.mkdir(exist_ok=True)


def check_connectivity(net):
    """Count connected components."""
    adj = defaultdict(set)
    for cl in net.crosslinks:
        adj[cl.fiber_i].add(cl.fiber_j)
        adj[cl.fiber_j].add(cl.fiber_i)
    
    visited = set()
    comps = []
    for s in range(net.num_fibers):
        if s not in visited:
            c = 0
            q = [s]
            while q:
                n = q.pop(0)
                if n in visited: continue
                visited.add(n)
                c += 1
                q.extend(adj[n] - visited)
            comps.append(c)
    
    return {
        "num_components": len(comps),
        "largest_component": max(comps) if comps else 0,
        "largest_pct": round(max(comps)/net.num_fibers*100, 1) if comps and net.num_fibers > 0 else 0,
        "is_connected": len(comps) == 1,
    }


def compute_graph_stats(net):
    """Compute graph statistics for a FiberNetwork."""
    nf = net.num_fibers
    nc = net.num_crosslinks
    
    # Degree distribution
    deg = defaultdict(int)
    for cl in net.crosslinks:
        deg[cl.fiber_i] += 1
        deg[cl.fiber_j] += 1
    
    if nf > 0:
        all_deg = [deg.get(i, 0) for i in range(nf)]
        degree_stats = {
            "mean": round(float(np.mean(all_deg)), 3),
            "std": round(float(np.std(all_deg)), 3),
            "max": int(np.max(all_deg)),
            "min": int(np.min(all_deg)),
            "isolated": int(sum(1 for d in all_deg if d == 0)),
        }
    else:
        degree_stats = {"mean": 0, "std": 0, "max": 0, "min": 0, "isolated": 0}
    
    # Fiber lengths
    lengths = []
    for f in net.fibers:
        pts = f.centerline
        if pts is not None and len(pts) >= 2:
            dl = np.diff(pts, axis=0)
            lengths.append(float(np.sum(np.linalg.norm(dl, axis=1))))
    
    length_stats = {}
    if lengths:
        length_stats = {
            "mean": round(float(np.mean(lengths)), 3),
            "std": round(float(np.std(lengths)), 3),
            "total": round(float(np.sum(lengths)), 3),
        }
    
    # Bounding box
    try:
        bb_min, bb_max = net.bounding_box()
        bbox_size = (bb_max - bb_min).tolist()
    except:
        bbox_size = None
    
    return {
        "num_fibers": nf,
        "num_crosslinks": nc,
        "dimension": net.dimension,
        "degree": degree_stats,
        "length": length_stats,
        "bbox_size": bbox_size,
    }


def test_generator(name, skip_large=False):
    """Test a single generator."""
    result = {"name": name, "status": "unknown"}
    
    try:
        t0 = time.time()
        net = fn.create(name)
        dt = time.time() - t0
        
        if not isinstance(net, FiberNetwork):
            result["status"] = "wrong_type"
            result["type"] = str(type(net))
            result["time_s"] = round(dt, 4)
            return result
        
        result["status"] = "ok"
        result["time_s"] = round(dt, 4)
        result["graph"] = compute_graph_stats(net)
        result["connectivity"] = check_connectivity(net)
        
    except MemoryError:
        result["status"] = "oom"
        result["error"] = "Out of memory"
    except Exception as e:
        result["status"] = "fail"
        result["error"] = str(e)[:300]
        result["traceback"] = traceback.format_exc()[-500:]
    
    gc.collect()
    return result


def main():
    gens = fn.list_generators()
    print(f"Testing {len(gens)} registered generators...")
    print(f"{'Name':<35} {'Status':<8} {'Fibers':>7} {'CL':>6} {'Comp':>5} {'Conn':>5} {'Time':>6}")
    print("-" * 85)
    
    results = []
    skip_large = {"field_guided", "gyroid_infill"}  # Known slow/large
    
    for name in gens:
        if name in skip_large:
            print(f"{name:<35} {'SKIP':<8} (too large/slow)")
            results.append({"name": name, "status": "skipped", "reason": "too large/slow"})
            continue
        
        r = test_generator(name)
        results.append(r)
        
        status = r["status"]
        if status == "ok":
            g = r["graph"]
            c = r["connectivity"]
            conn = "Y" if c["is_connected"] else f"{c['largest_pct']}%"
            print(f"{name:<35} {status:<8} {g['num_fibers']:>7} {g['num_crosslinks']:>6} "
                  f"{c['num_components']:>5} {conn:>5} {r['time_s']:>5.2f}s")
        else:
            err = r.get("error", "")[:40]
            print(f"{name:<35} {status:<8} {err}")
    
    # Summary
    ok = [r for r in results if r["status"] == "ok"]
    fail = [r for r in results if r["status"] == "fail"]
    oom = [r for r in results if r["status"] == "oom"]
    wrong = [r for r in results if r["status"] == "wrong_type"]
    skip = [r for r in results if r["status"] == "skipped"]
    
    connected = [r for r in ok if r["connectivity"]["is_connected"]]
    zero_cl = [r for r in ok if r["graph"]["num_crosslinks"] == 0]
    
    print(f"\n{'='*85}")
    print(f"SUMMARY")
    print(f"{'='*85}")
    print(f"Total registered: {len(gens)}")
    print(f"  OK: {len(ok)}")
    print(f"  FAIL: {len(fail)}")
    print(f"  OOM: {len(oom)}")
    print(f"  Wrong type: {len(wrong)}")
    print(f"  Skipped: {len(skip)}")
    print(f"\nAmong OK generators:")
    print(f"  Connected: {len(connected)}/{len(ok)} ({100*len(connected)/len(ok):.1f}%)")
    print(f"  Zero crosslinks: {len(zero_cl)}/{len(ok)} ({100*len(zero_cl)/len(ok):.1f}%)")
    
    # Diversity stats
    if ok:
        fiber_counts = [r["graph"]["num_fibers"] for r in ok]
        cl_counts = [r["graph"]["num_crosslinks"] for r in ok]
        dims = [r["graph"]["dimension"] for r in ok]
        
        print(f"\nDiversity:")
        print(f"  Fiber range: {min(fiber_counts)}-{max(fiber_counts)}, unique: {len(set(fiber_counts))}")
        print(f"  CL range: {min(cl_counts)}-{max(cl_counts)}, unique: {len(set(cl_counts))}")
        print(f"  2D structures: {sum(1 for d in dims if d == 2)}")
        print(f"  3D structures: {sum(1 for d in dims if d == 3)}")
    
    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_generators": len(gens),
        "results": results,
        "summary": {
            "ok": len(ok),
            "fail": len(fail),
            "oom": len(oom),
            "wrong_type": len(wrong),
            "skipped": len(skip),
            "connected": len(connected),
            "zero_cl": len(zero_cl),
        }
    }
    
    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to {OUT_FILE}")
    
    # List failures
    if fail:
        print(f"\n{'='*85}")
        print(f"FAILURES ({len(fail)})")
        print(f"{'='*85}")
        for r in fail:
            print(f"  {r['name']}: {r.get('error', '?')[:80]}")


if __name__ == "__main__":
    main()
