"""Independent analytical and fixed-cap correspondence regression."""
from pathlib import Path
import numpy as np
import pytest
from fibernet.analysis.stokes_cap_comparison import compare_saved_stokes_caps

# Separately specified VTK tetra10 local edge pairs.
TET_EDGES = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))


def prism(permuted=False, dy=0.0):
    """A triangular cap at each end. Only the physical coordinates match."""
    xyz = [(x, y + dy, z) for x in (0., 1.) for y, z in ((0., 0.), (1., 0.), (0., 1.))]
    tets = []
    mids = {}
    for ids in ((0, 1, 2, 5), (0, 1, 4, 5), (0, 3, 4, 5)):
        ids = list(ids)
        if permuted:
            ids = ids[1:] + ids[:1]
        cell = np.asarray([xyz[i] for i in ids])
        if np.linalg.det(cell[1:] - cell[0]) < 0:
            ids[:2] = ids[:2][::-1]
        row = ids.copy()
        for i, j in TET_EDGES:
            ends = tuple(sorted((ids[i], ids[j])))
            if ends not in mids:
                mids[ends] = len(xyz)
                xyz.append(tuple(np.asarray([xyz[k] for k in ends]).mean(axis=0)))
            row.append(mids[ends])
        tets.append(row)
    q = np.asarray(xyz)
    t = np.asarray(tets)
    if permuted:
        order = np.arange(len(q))[::-1]
        t = np.argsort(order)[t[::-1]]
        q = q[order]
    return q, t


def velocity(points, field):
    return np.column_stack([field(points[:, 1], points[:, 2]), np.zeros((len(points), 2))])


def test_closed_form_polynomial_and_numbering(monkeypatch):
    from fibernet.analysis import stokes_cap_comparison as mod
    a, b = prism(), prism(permuted=True)
    old = (*a, velocity(a[0], lambda y, z: np.ones(len(y))))
    new = (*b, velocity(b[0], lambda y, z: 1.02 + .06 * (y*y-z*z)))
    paths = {Path("old"): old, Path("new"): new}
    monkeypatch.setattr(mod, "read_own_quadratic_vtu", paths.__getitem__)
    r = compare_saved_stokes_caps(Path("old"), Path("new"))
    for key, outward_sign in (("x_min", -1), ("x_max", 1)):
        q = r[key]
        assert q["triangles"] == 1 and q["area_m2"] == .5
        assert q["reference_outward_m3_s"] == outward_sign * .5
        np.testing.assert_allclose([q["throughput_ratio"], q["normalized_profile_residual_L2"]],
                                   [1.02, .02], rtol=0, atol=2e-15)
        assert abs(q["residual_signed_flux_over_reference"]) < 1e-15


@pytest.mark.parametrize("case", ["gain", "reverse", "zero", "cancelled", "translated"])
def test_scope_and_gain_control(monkeypatch, case):
    from fibernet.analysis import stokes_cap_comparison as mod
    a = prism()
    b = prism(permuted=True, dy=.125 if case == "translated" else 0.)
    def reference(y, z):
        return y-z if case == "cancelled" else np.ones(len(y))
    def candidate(y, z):
        scale = {"gain": 2., "translated": 1., "zero": 0., "reverse": -1., "cancelled": 1.}[case]
        return np.ones(len(y))*scale
    paths = {"old": (*a, velocity(a[0], reference)), "new": (*b, velocity(b[0], candidate))}
    monkeypatch.setattr(mod, "read_own_quadratic_vtu", paths.__getitem__)
    if case == "gain":
        assert compare_saved_stokes_caps("old", "new")["x_max"]["normalized_profile_residual_L2"] == 0.
    else:
        message = "must match exactly" if case == "translated" else "flows in the same direction"
        with pytest.raises(ValueError, match=message):
            compare_saved_stokes_caps("old", "new")


def test_cancellation_dominated_candidate_is_not_nonzero(monkeypatch):
    from fibernet.analysis import stokes_cap_comparison as mod
    p, t = prism()
    def profile(y, z):
        x = 1-y-z
        # At all vertices: 10. At the three edge midpoints: 1, 2, -3.
        # The exact flux vanishes; rescaling by 10 can generate roundoff.
        return 10*(x*(2*x-1)+y*(2*y-1)+z*(2*z-1)) + 4*x*y + 8*y*z - 12*z*x
    paths = {"old": (p, t, velocity(p, lambda y,z: np.ones(len(y)))),
             "new": (p, t, velocity(p, profile))}
    monkeypatch.setattr(mod, "read_own_quadratic_vtu", paths.__getitem__)
    with pytest.raises(ValueError, match="flows in the same direction"):
        compare_saved_stokes_caps("old", "new")


def test_strict_native_vtu_replay(tmp_path):
    pytest.importorskip("meshio")
    from fibernet.sim.tet4_open_stokes import (
        OpenStokesQuadraticVelocity, write_open_stokes_velocity_vtu,
    )
    from fibernet.analysis.stokes_cap_flux import audit_saved_stokes

    files = []
    for i in (0, 1):
        p, t = prism(permuted=bool(i))
        source = np.zeros((len(p), 2), dtype=np.uint64)
        corners = np.unique(t[:, :4])
        source[corners] = corners[:, None]
        for k, (a, b) in enumerate(TET_EDGES):
            source[t[:, 4+k]] = np.sort(t[:, [a, b]], axis=1)
        # A negative velocity in both runs exercises reverse-direction pairs.
        u = velocity(p, lambda y, z: -(1. + i*(.02+.06*(y*y-z*z))))
        field = OpenStokesQuadraticVelocity(p, t, u, source, np.arange(len(t), dtype=np.uint64))
        files.append(tmp_path / f"state-{i}.vtu")
        write_open_stokes_velocity_vtu(files[-1], field)
    comparison = compare_saved_stokes_caps(*files)

    from json import loads
    from subprocess import run
    from sys import executable

    command = [executable, "-m", "fibernet.analysis.stokes_cap_comparison"]
    result = run([*command, *files], capture_output=True, text=True, check=True, timeout=30)
    assert result.stderr == ""
    assert loads(result.stdout) == comparison

    audit = [audit_saved_stokes(path) for path in files]
    for side in ("x_min", "x_max"):
        q = comparison[side]
        np.testing.assert_allclose([q["reference_outward_m3_s"], q["candidate_outward_m3_s"]],
                                   [r[side]["outward_flow_m3_s"] for r in audit], rtol=1e-15)
        np.testing.assert_allclose(q["normalized_profile_residual_L2"], .02, atol=2e-15)

def test_every_local_cap_face_and_three_vertex_permutation():
    from itertools import permutations
    from fibernet.analysis.stokes_cap_comparison import _trace
    points,cells = prism()
    v = velocity(points,lambda y,z: 1.2+.11*y*y-.03*z+ .04*y*z)
    for side in (False,True):
        ref_plane,ref = _trace(points,cells,v,side)
        for perm in permutations(range(4)):
            edge_slot = {tuple(sorted(ij)): k for k,ij in enumerate(TET_EDGES,4)}
            indices=list(perm)+[edge_slot[tuple(sorted((perm[i],perm[j])))] for i,j in TET_EDGES]
            plane,rows = _trace(points,cells[:,indices],v,side)
            assert plane==ref_plane and rows.keys()==ref.keys()
            for k, (area, speed, mids) in rows.items():
                a0, s0, m0 = ref[k]
                assert area==a0 and mids==m0
                np.testing.assert_array_equal(speed,s0)



def test_cli_does_not_emit_partial_json_on_invalid_input(tmp_path):
    from subprocess import run
    from sys import executable

    bad = tmp_path / "bad.vtu"
    bad.write_text("<broken")
    result = run([executable, "-m", "fibernet.analysis.stokes_cap_comparison",
                  str(bad), str(bad)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert result.stdout == ""
    assert "Invalid VTU XML" in result.stderr
    assert "Traceback" not in result.stderr
