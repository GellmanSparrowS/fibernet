"""Pure split fixtures; load only dataset.py, avoiding unrelated package imports."""
import builtins
import copy
import importlib.util
from fractions import Fraction
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

_PATH = Path(__file__).resolve().parents[1] / "fibernet" / "ml" / "dataset.py"
_NAME = "_fibernet_dataset_group_split_test"
_SPEC = importlib.util.spec_from_file_location(_NAME, _PATH)
_DATASET = importlib.util.module_from_spec(_SPEC)
sys.modules[_NAME] = _DATASET
_SPEC.loader.exec_module(_DATASET)
DatasetSample, FiberNetDataset = _DATASET.DatasetSample, _DATASET.FiberNetDataset


def fixture(counts=(2,) * 10):
    samples, groups = [], []
    for parent, count in enumerate(counts):
        group = f"parent-{parent:02d}"
        for variant in range(count):
            samples.append(DatasetSample(
                f"{group}-material-{variant}", "fixture", (1, 1), {},
                {"x": float(parent)}, {"y": float(variant)},
                {"parent_source": group, "material_variant": variant,
                 "evaluation_exposure": "consumed_final",
                 "prior_evaluation": {"campaign": "saved-fixture"}}))
            groups.append(group)
    return FiberNetDataset(samples, ["x"], ["y"]), groups


def ids(parts):
    return [[sample.sample_id for sample in part.samples] for part in parts]


def parent_sets(parts):
    return [{sample.metadata["parent_source"] for sample in part.samples}
            for part in parts]


def no_sklearn(name, *args, **kwargs):
    if name == "sklearn" or name.startswith("sklearn."):
        raise AssertionError("Invalid input or row mode imported sklearn")
    return _IMPORT(name, *args, **kwargs)


_IMPORT = builtins.__import__


class GroupSplitTests(unittest.TestCase):
    def test_legacy_row_order_and_positional_arguments(self):
        ds, _ = fixture()
        expected = np.random.RandomState(23).permutation(len(ds.samples))
        expected = [expected[:12], expected[12:16], expected[16:]]
        expected_ids = [[ds.samples[i].sample_id for i in part] for part in expected]
        with mock.patch("builtins.__import__", side_effect=no_sklearn):
            self.assertEqual(ids(ds.split((.6, .2, .2), 23)), expected_ids)
            self.assertEqual(ids(ds.split((.6, .2, .2), 23, method="row")), expected_ids)

    def test_legacy_small_empty_partitions_preserved(self):
        ds, _ = fixture((1,))
        with mock.patch("builtins.__import__", side_effect=no_sklearn):
            self.assertEqual([part.n_samples for part in ds.split()], [0, 0, 1])

    def test_group_partition_coverage_material_variants_and_metadata(self):
        ds, groups = fixture((1, 3, 2, 4, 1, 5, 2, 3, 4, 2))
        before = copy.deepcopy(ds.__dict__)
        groups_before = groups[:]
        parts = ds.split((.6, .2, .2), 23, method="group", groups=groups)
        parents = parent_sets(parts)
        self.assertEqual([len(value) for value in parents], [6, 2, 2])
        for i in range(3):
            self.assertGreater(parts[i].n_samples, 0)
            for j in range(i + 1, 3):
                self.assertFalse(parents[i] & parents[j])
        self.assertEqual(sorted(sum(ids(parts), [])), sorted(ids((ds,))[0]))
        self.assertEqual(ds.__dict__, before)
        self.assertEqual(groups, groups_before)
        for part in parts:
            self.assertEqual(part.feature_names, ds.feature_names)
            self.assertEqual(part.target_names, ds.target_names)
            row_positions = [next(i for i, source in enumerate(ds.samples)
                                  if source is sample) for sample in part.samples]
            self.assertEqual(row_positions, sorted(row_positions))
            for sample in part.samples:
                self.assertEqual(sample.metadata["evaluation_exposure"], "consumed_final")
                self.assertEqual(sample.metadata["prior_evaluation"], {"campaign": "saved-fixture"})

    def test_result_containers_are_new_but_samples_retain_legacy_identity(self):
        ds, groups = fixture()
        parts = ds.split(method="group", groups=groups)
        original_count = ds.n_samples
        self.assertEqual(len({id(part) for part in parts}), 3)
        for part in parts:
            self.assertIsNot(part.samples, ds.samples)
            self.assertTrue(all(any(sample is source for source in ds.samples)
                                for sample in part.samples))
        parts[0].samples.pop()
        self.assertEqual(ds.n_samples, original_count)

    def test_deterministic_seed_and_row_order_independent_group_assignment(self):
        ds, groups = fixture()
        a = ds.split(method="group", groups=groups, random_state=17)
        self.assertEqual(ids(a), ids(ds.split(method="group", groups=groups, random_state=17)))
        self.assertNotEqual(parent_sets(a), parent_sets(ds.split(
            method="group", groups=groups, random_state=41)))
        reversed_ds = FiberNetDataset(list(reversed(ds.samples)), ["x"], ["y"])
        b = reversed_ds.split(method="group", groups=list(reversed(groups)), random_state=17)
        self.assertEqual(parent_sets(a), parent_sets(b))

    def test_decimal_boundaries_use_exact_group_quota_arithmetic(self):
        # Both sequential division and direct multiplication had binary-ceil
        # boundary errors. Genuine above-boundary decimal inputs still ceil up.
        for count, ratios, expected in [
            (10, (.7, .2, .1), [7, 2, 1]),
            (10, (.1, .6, .3), [1, 6, 3]),
            (25, (.44, .28, .28), [11, 7, 7]),
            (10, (.6999999999999999, .2, .1000000000000001), [6, 2, 2]),
        ]:
            with self.subTest(count=count, ratios=ratios):
                ds, groups = fixture((1,) * count)
                parts = ds.split(ratios, 42, method="group", groups=groups)
                self.assertEqual([part.n_samples for part in parts], expected)

    def test_distinct_exact_strings_survive_backend_encoding(self):
        for labels in [
            ["source", "source\0", "other"],
            ["a", "a\0", "b", "b\0", "c", "c\0"],
            ["source", "source\0", "source\0\0", "other", "other ",
             "é", "e\u0301", "x\0y", "\ud800"],
        ]:
            with self.subTest(labels=labels):
                ds, _ = fixture((2,) * len(labels))
                groups = [label for label in labels for _ in range(2)]
                for sample, group in zip(ds.samples, groups):
                    sample.metadata["parent_source"] = group
                parts = ds.split(method="group", groups=groups)
                assigned = parent_sets(parts)
                self.assertEqual(set.union(*assigned), set(labels))
                self.assertFalse(assigned[0] & assigned[1] or assigned[0] & assigned[2]
                                 or assigned[1] & assigned[2])
                self.assertEqual(sum(len(part.samples) for part in parts), len(groups))
                for part, values in zip(parts, assigned):
                    self.assertEqual(part.n_samples, 2 * len(values))
                reverse = FiberNetDataset(list(reversed(ds.samples)), ["x"], ["y"])
                self.assertEqual(assigned, parent_sets(reverse.split(
                    method="group", groups=list(reversed(groups)))))
        ds, _ = fixture((1, 1, 1))
        parts = ds.split(method="group", groups=["source", "source\0", "other"])
        self.assertEqual([part.n_samples for part in parts], [1, 1, 1])

    def test_broken_backend_group_counts_rejected_before_outputs(self):
        from sklearn.model_selection import GroupShuffleSplit
        ds, groups = fixture((1,) * 10)
        outputs = iter([(np.arange(8), np.array([8, 9])),
                        (np.arange(5), np.array([5, 6, 7]))])
        with mock.patch.object(GroupShuffleSplit, "split", side_effect=lambda *a, **k: iter([next(outputs)])):
            with mock.patch.object(FiberNetDataset, "__init__", side_effect=AssertionError("constructed output too early")):
                with self.assertRaisesRegex(ValueError, "requested group counts"):
                    ds.split((.6, .2, .2), method="group", groups=groups)

    def test_seeded_assignment_matches_existing_backend_for_ordinary_labels(self):
        from sklearn.model_selection import GroupShuffleSplit
        ds, groups = fixture((1, 5, 3, 1, 8, 2, 4, 1, 1, 5))
        rng = np.random.RandomState(281)
        development, test = next(GroupShuffleSplit(
            n_splits=1, test_size=2, random_state=rng).split(
                np.arange(len(groups)), groups=groups))
        train, val = next(GroupShuffleSplit(
            n_splits=1, test_size=2, random_state=rng).split(
                development, groups=[groups[i] for i in development]))
        expected = [[ds.samples[i].sample_id for i in part]
                    for part in (development[train], development[val], test)]
        self.assertEqual(ids(ds.split((.6, .2, .2), 281, method="group", groups=groups)), expected)

    def test_decimal_tenths_grid_matches_exact_allocation(self):
        # 28 group counts times 36 positive ratio triplets; no fit/physics.
        for count in range(3, 31):
            ds, groups = fixture((1,) * count)
            for train in range(1, 9):
                for val in range(1, 10 - train):
                    test = 10 - train - val
                    ratios = (train / 10, val / 10, test / 10)
                    test_quota = Fraction(count * test, 10)
                    n_test = -(-test_quota.numerator // test_quota.denominator)
                    val_quota = Fraction((count - n_test) * val, train + val)
                    n_val = -(-val_quota.numerator // val_quota.denominator)
                    expected = [count - n_test - n_val, n_val, n_test]
                    with self.subTest(count=count, ratios=ratios):
                        if min(expected) < 1:
                            with self.assertRaises(ValueError):
                                ds.split(ratios, method="group", groups=groups)
                        else:
                            parts = ds.split(ratios, method="group", groups=groups)
                            self.assertEqual([part.n_samples for part in parts], expected)

    def test_minimum_three_groups_and_numpy_integer_seed(self):
        ds, groups = fixture((2, 2, 2))
        parts = ds.split(method="group", groups=tuple(groups), random_state=np.int64(0))
        self.assertEqual([part.n_samples for part in parts], [2, 2, 2])

    def test_local_random_state_does_not_modify_global_rng(self):
        ds, groups = fixture()
        before = np.random.get_state()
        ds.split(method="group", groups=groups)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_invalid_requests_reject_before_split_dependency_or_outputs(self):
        ds, groups = fixture()
        bad = [
            {"groups": None}, {"groups": "parent"}, {"groups": groups[:-1]},
            {"groups": np.asarray(groups)}, {"groups": [1] * len(groups)},
            {"groups": [""] * len(groups)}, {"groups": [" \t"] * len(groups)},
            {"groups": ["one"] * len(groups)},
            {"groups": ["a", "b"] * (len(groups) // 2)},
            *[{"groups": groups[:3] + [value] + groups[4:]}
              for value in (True, np.bool_(False), float("nan"), float("inf"), [], {}, None)],
            {"ratios": (0, .5, .5)}, {"ratios": (-.1, .5, .6)},
            {"ratios": (.7, .1, .1)}, {"ratios": (True, .1, .1)},
            {"ratios": (float("nan"), .1, .2)},
            {"ratios": (float("inf"), .1, .2)},
            {"ratios": ("0.7", .1, .2)}, {"ratios": (.7, .3)},
            {"ratios": (.01, .01, .98)},
            {"random_state": None}, {"random_state": True}, {"random_state": -1},
            {"random_state": .5}, {"random_state": 2**32},
        ]
        for changes in bad:
            with self.subTest(changes=changes):
                options = dict(method="group", groups=groups)
                options.update(changes)
                before = copy.deepcopy(ds.__dict__)
                with mock.patch("builtins.__import__", side_effect=no_sklearn):
                    with self.assertRaises(ValueError):
                        ds.split(**options)
                self.assertEqual(ds.__dict__, before)
        with mock.patch("builtins.__import__", side_effect=no_sklearn):
            for method in ("unknown", None, True, []):
                with self.subTest(method=method), self.assertRaises(ValueError):
                    ds.split(method=method)
            with self.assertRaisesRegex(ValueError, "requires method='group'"):
                ds.split(groups=groups)

    def test_optional_sklearn_absence_is_explicit(self):
        ds, groups = fixture()
        def missing(name, *args, **kwargs):
            if name == "sklearn.model_selection":
                raise ImportError("fixture missing backend")
            return _IMPORT(name, *args, **kwargs)
        with mock.patch("builtins.__import__", side_effect=missing):
            with self.assertRaisesRegex(ImportError, "optional scikit-learn"):
                ds.split(method="group", groups=groups)
        with mock.patch("builtins.__import__", side_effect=missing):
            self.assertEqual(len(ds.split()), 3)

    def test_broken_backend_overlap_is_rejected_before_output_construction(self):
        from sklearn.model_selection import GroupShuffleSplit
        ds, groups = fixture((2, 2, 2))
        # Exact row coverage, but material variants from each parent are separated.
        outputs = iter([(np.array([0, 1, 2, 3]), np.array([4, 5])),
                        (np.array([0, 2]), np.array([1, 3]))])
        with mock.patch.object(GroupShuffleSplit, "split", side_effect=lambda *a, **k: iter([next(outputs)])):
            with mock.patch.object(FiberNetDataset, "__init__", side_effect=AssertionError("constructed output too early")):
                with self.assertRaisesRegex(ValueError, "Source groups overlap"):
                    ds.split(method="group", groups=groups)

    def test_broken_backend_row_coverage_rejected(self):
        from sklearn.model_selection import GroupShuffleSplit
        ds, groups = fixture((2, 2, 2))
        outputs = iter([(np.array([0, 1, 2, 3]), np.array([4, 4])),
                        (np.array([0, 1]), np.array([2, 3]))])
        with mock.patch.object(GroupShuffleSplit, "split", side_effect=lambda *a, **k: iter([next(outputs)])):
            with self.assertRaisesRegex(ValueError, "exactly partition"):
                ds.split(method="group", groups=groups)


if __name__ == "__main__":
    unittest.main(verbosity=2)

