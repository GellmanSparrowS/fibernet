"""Pure structural rewards must remain usable without PyTorch installed."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_public_rewards_without_torch_in_fresh_process(tmp_path):
    program = textwrap.dedent("""
        import importlib.abc
        import math
        import sys

        class BlockTorch(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "torch" or fullname.startswith("torch."):
                    raise ModuleNotFoundError("torch intentionally unavailable", name=fullname)

        assert "torch" not in sys.modules
        sys.meta_path.insert(0, BlockTorch())

        import numpy as np
        from fibernet.ml import (
            StructureState, connectivity_reward, property_target_reward,
        )
        from fibernet.ml import gflownet

        assert not gflownet.HAS_TORCH
        assert "torch" not in sys.modules
        assert connectivity_reward is gflownet.connectivity_reward
        assert property_target_reward is gflownet.property_target_reward

        empty = StructureState()
        assert connectivity_reward(empty) == 0.1
        assert property_target_reward(empty) == 0.1

        # Validation precedes even the empty-state early return.
        for invalid in (0.0, 0.5, 2.0, 10**1000, float("nan"), float("inf"), -float("inf")):
            try:
                property_target_reward(empty, target_connectivity=invalid)
            except ValueError as exc:
                assert "custom reward_fn" in str(exc)
            else:
                raise AssertionError("unsupported connectivity target was ignored")
        for invalid in (True, False, "1.0", None, [1.0], np.array([1.0]), 1+0j, np.bool_(True)):
            try:
                property_target_reward(empty, target_connectivity=invalid)
            except TypeError as exc:
                assert "real scalar" in str(exc)
            else:
                raise AssertionError("invalid connectivity target was accepted")

        positions = [np.array([0., 0.]), np.array([1., 0.]), np.array([2., 0.])]
        chain = StructureState(node_positions=positions, edges=[(0, 1), (1, 2)])
        assert math.isclose(connectivity_reward(chain), 0.7820534770689358)
        assert property_target_reward(chain, target_n_nodes=3, target_n_edges=2) == 10.

        disconnected = StructureState(node_positions=positions, edges=[(0, 1)])
        assert math.isclose(connectivity_reward(disconnected), 1.9779007785878446)
        assert math.isclose(
            property_target_reward(disconnected, target_n_nodes=3, target_n_edges=1),
            20. / 3.,
        )
        # Exact float representation checks preserve historical default outputs.
        for state, expected in ((empty, 0.1), (chain, 10.0), (disconnected, (2. / 3.) * 10)):
            kwargs = {} if state is empty else dict(target_n_nodes=3, target_n_edges=state.n_edges)
            for target in (1.0, 1, np.float64(1.0), np.float32(1.0)):
                actual = property_target_reward(state, target_connectivity=target, **kwargs)
                assert actual.hex() == expected.hex()
            assert property_target_reward(state, **kwargs).hex() == expected.hex()
        # Also pin the non-peak count factors in the default formula.
        expected = math.exp(-0.1 * (3-15)**2) * math.exp(-0.05 * (2-20)**2) * 1.0 * 10
        assert property_target_reward(chain).hex() == expected.hex()
    """)
    root = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ, MPLBACKEND="Agg", MPLCONFIGDIR=str(tmp_path / "mpl"))
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [root, env.get("PYTHONPATH")]))
    completed = subprocess.run(
        [sys.executable, "-c", program], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
