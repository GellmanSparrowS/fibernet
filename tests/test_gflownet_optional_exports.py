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
    """)
    root = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ, MPLBACKEND="Agg", MPLCONFIGDIR=str(tmp_path / "mpl"))
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [root, env.get("PYTHONPATH")]))
    completed = subprocess.run(
        [sys.executable, "-c", program], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
