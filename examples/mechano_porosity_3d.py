"""Estimate void volume before and after axial loading of a 3D rod network."""
import argparse
from contextlib import redirect_stdout
import json
import sys

import numpy as np

if __package__:
    from .mechanoelectric_network_3d import build_sample
else:
    from mechanoelectric_network_3d import build_sample


def _cylinder_union_volume(start, stop, radii, bounds, *, samples, axis, phase):
    """Intersect finite circular cylinders with parallel rays in a box.

    The union length on each ray is analytic; only the two transverse
    coordinates are sampled. All coordinates and radii use the same unit.
    """
    p = np.asarray(start, dtype=float)
    q = np.asarray(stop, dtype=float)
    r = np.asarray(radii, dtype=float)
    b = np.asarray(bounds, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or q.shape != p.shape or r.shape != p.shape[:1]:
        raise ValueError("expected matching (n, 3) endpoints and (n,) radii")
    if b.shape != (2, 3) or np.any(b[1] <= b[0]):
        raise ValueError("bounds must be increasing (lower, upper) triples")
    if not np.isfinite(b).all() or not all(np.isfinite(a).all() for a in (p, q, r)):
        raise ValueError("cylinder coordinates, radii and bounds must be finite")
    if np.any(r <= 0):
        raise ValueError("radii must be positive")
    if (not isinstance(samples, (int, np.integer)) or isinstance(samples, (bool, np.bool_))
            or not 1 <= samples <= 3072):
        raise ValueError("samples must be an integer from 1 to 3072")
    if axis not in (0, 1, 2) or not 0 < phase < 1:
        raise ValueError("axis must be 0, 1 or 2 and phase must be in (0, 1)")
    order = np.roll(np.arange(3), -axis)
    p, q, b = p[:, order], q[:, order], b[:, order]
    lo, hi = b
    d = q - p
    length = np.linalg.norm(d, axis=1)
    if np.any(~np.isfinite(length)) or np.any(length <= 0):
        raise ValueError("cylinders must have positive finite lengths")
    u = d / length[:, None]
    # The finite cylinder lies in these conservative AABBs. Tile only
    # bounds the workspace and prunes untouched rays; it does not change sampling.
    c_lo, c_hi = np.minimum(p, q) - r[:, None], np.maximum(p, q) + r[:, None]
    dy, dz = (hi - lo)[1:] / samples
    iy = lo[1] + (np.arange(samples) + phase) * dy
    iz = lo[2] + (np.arange(samples) + 1 - phase) * dz
    total = 0.
    for j in range(0, samples, 48):
        ys = iy[j:j + 48]
        for k in range(0, samples, 48):
            zs = iz[k:k + 48]
            c = np.flatnonzero((c_lo[:, 1] <= ys[-1]) & (c_hi[:, 1] >= ys[0])
                               & (c_lo[:, 2] <= zs[-1]) & (c_hi[:, 2] >= zs[0]))
            if not len(c):
                continue
            xc, yc, zc = p[c].T
            vx, vy, vz = u[c].T
            yy = ys[:, None, None] - yc
            zz = zs[None, :, None] - zc
            transverse = np.hypot(vy, vz)
            axial_ray = transverse <= 8 * np.finfo(float).eps
            dist = np.where(axial_ray, 1., transverse)
            along = (vy * yy + vz * zz) / dist
            across = (-vz * yy + vy * zz) / dist
            discriminant = r[c]**2 - across**2
            root = np.sqrt(np.maximum(discriminant, 0.))
            a = np.where(axial_ray, lo[0] - xc, (vx * along - root) / dist)
            z = np.where(axial_ray, hi[0] - xc, (vx * along + root) / dist)
            valid = np.where(axial_ray, yy**2 + zz**2 <= r[c]**2, discriminant >= 0.)
            cap_offset = transverse * along
            parallel_cap = np.abs(vx) <= 8 * np.finfo(float).eps
            divisor = np.where(parallel_cap, 1., vx)
            h0, h1 = -cap_offset / divisor, (length[c] - cap_offset) / divisor
            a = np.maximum(a, np.where(parallel_cap, lo[0] - xc, np.minimum(h0, h1)))
            z = np.minimum(z, np.where(parallel_cap, hi[0] - xc, np.maximum(h0, h1)))
            valid &= (~parallel_cap) | ((cap_offset >= 0.) & (cap_offset <= length[c]))
            a = np.maximum(a, lo[0] - xc) + xc
            z = np.minimum(z, hi[0] - xc) + xc
            valid &= z > a
            a, z = np.where(valid, a, hi[0]), np.where(valid, z, hi[0])
            sort = np.argsort(a, axis=-1)
            a, z = np.take_along_axis(a, sort, axis=-1), np.take_along_axis(z, sort, axis=-1)
            rightmost = np.maximum.accumulate(z, axis=-1)
            a[..., 1:] = np.maximum(a[..., 1:], rightmost[..., :-1])
            total += np.maximum(0., z - a).sum(dtype=float)
    return float(total * dy * dz)


def run_example(strain=0.005, *, samples=768, axis=0):
    """Return paired 3D void-volume estimates for fixed-radius straight rods."""
    import fibernet as fn

    if not np.isfinite(strain) or not 0. < strain <= 0.01:
        raise ValueError("strain must be a positive engineering strain no larger than 0.01")
    graph, ids, edges, x, endpoints, _, _, radii = build_sample()
    states = {"reference": x}
    width = 0.006
    grip = sum(x[:, 0] == 0.)
    if grip != 16 or sum(x[:, 0] == width) != grip:
        raise RuntimeError("unexpected sample grip geometry")
    for sign, name in ((-1., "compression"), (1., "tension")):
        result = fn.simulate(graph, backend="fem", nonlinear=False, material_mode="graph",
                             length_unit="m", stretch_ratio=1. + sign * strain,
                             pct=(grip + 0.5) / len(ids))
        coordinates = dict(zip(result.metadata["node_ids"], result.deformed_positions))
        states[name] = np.asarray([coordinates[k] for k in ids], dtype=float)
    # All three states use the same enclosing box, so a volume change is
    # not an artifact of changing the measurement domain.
    box = np.array([[-1.2, -.5, -.5], [7.2, 6.5, 6.5]])
    rv = np.asarray(radii) * 1000.
    box_volume = float(np.prod(box[1] - box[0]))
    evaluations = []
    for phase in (.27, .68):
        volumes = {}
        for name, xyz in states.items():
            p = xyz[endpoints[:, 0]] * 1000.
            q = xyz[endpoints[:, 1]] * 1000.
            if np.any(np.minimum(p, q) - rv[:, None] <= box[0]) or np.any(
                    np.maximum(p, q) + rv[:, None] >= box[1]):
                raise RuntimeError("the box no longer encloses all cylinders")
            solid = _cylinder_union_volume(p, q, rv, box, samples=samples,
                                            axis=axis, phase=phase)
            volumes[name] = box_volume - solid
        v0 = volumes["reference"]
        evaluations.append({
            "phase": phase, "void_mm3": volumes,
            "relative_change": {key: (volumes[key] - v0) / v0
                                for key in ("compression", "tension")},
        })
    return {"nodes": len(ids), "rods": len(edges), "strain_magnitude": float(strain),
            "ray_axis": "xyz"[axis], "samples_per_transverse_axis": samples,
            "box_mm3": box_volume, "estimates": evaluations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strain", type=float, default=.005)
    parser.add_argument("--samples", type=int, default=768)
    parser.add_argument("--axis", choices="xyz", default="x")
    args = parser.parse_args()
    try:
        with redirect_stdout(sys.stderr):
            summary = run_example(args.strain, samples=args.samples, axis="xyz".index(args.axis))
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
