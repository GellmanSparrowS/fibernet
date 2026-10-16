"""Bounded exception-preserving saves for FiberNetDataset's legacy format."""
import builtins
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import sys
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

import fibernet.ml.dataset as module
from fibernet.ml.dataset import DatasetSample, FiberNetDataset


FILES = ("features.npy", "labels.npy", "metadata.json")


def dataset(prefix="old", offset=0):
    return FiberNetDataset([
        DatasetSample(f"{prefix}-{i}", "audit", (1, 1), {"scale": np.int64(2)},
                      {"x": offset + i}, {"y": offset + 3 * i},
                      {"availability": {"status": "unavailable", "value": None}})
        for i in range(2)
    ], ["x"], ["y"])


def snapshot(path):
    if not path.exists() and not path.is_symlink():
        return None
    result = {}
    for item in [path, *sorted(path.rglob("*"))] if path.is_dir() else [path]:
        key = str(item.relative_to(path))
        if item.is_symlink():
            result[key] = ("symlink", os.readlink(item))
        elif item.is_dir():
            result[key] = ("directory", stat.S_IMODE(item.stat().st_mode))
        else:
            result[key] = (item.read_bytes(), stat.S_IMODE(item.stat().st_mode))
    return result


def destination(tmp_path, state):
    out = tmp_path / "dataset"
    if state != "absent":
        out.mkdir()
    if state in ("checkpoint", "complete", "partial"):
        (out / "checkpoint.json").write_text('{"leave": "checkpoint alone"}\n')
        (out / "notes").mkdir()
        (out / "notes" / "user.txt").write_text("unrelated nested contents\n")
    if state in ("complete", "partial"):
        dataset().save(out)
    if state == "partial":
        (out / "labels.npy").unlink()
        (out / "metadata.json").unlink()
    return out


@pytest.mark.parametrize("state", ["absent", "empty", "checkpoint", "complete", "partial"])
@pytest.mark.parametrize("failure", ["unsupported", "conversion", "json", "selected",
    "features_partial", "labels_partial", "features_after", "labels_after", "metadata_write",
    "publish_features.npy", "publish_labels.npy", "publish_metadata.json"])
def test_save_ordinary_failure_preserves_bytes_modes_and_absence(tmp_path, monkeypatch, state, failure):
    out = destination(tmp_path, state)
    before = snapshot(out)
    ds = dataset("new", 100)
    error = OSError(f"injected {failure}")
    real_save, real_replace, real_open = np.save, os.replace, builtins.open

    def fail(*args, **kwargs):
        raise error

    if failure == "unsupported":
        ds.samples[0].metadata["bad"] = object()
    elif failure == "conversion":
        monkeypatch.setattr(module, "_json_safe", fail)
    elif failure == "json":
        monkeypatch.setattr(module.json, "dumps", fail)
    elif failure == "selected":
        ds.samples[0].features["x"] = None
    elif failure.endswith(("_partial", "_after")):
        name = failure.split("_", 1)[0] + ".npy"

        def fail_save(path, values, *args, **kwargs):
            if Path(path).name == name:
                if failure.endswith("_after"):
                    real_save(path, values, *args, **kwargs)
                else:
                    Path(path).write_bytes(b"injected partial NPY write")
                raise error
            return real_save(path, values, *args, **kwargs)

        monkeypatch.setattr(np, "save", fail_save)
    elif failure == "metadata_write":
        def fail_open(path, mode="r", *args, **kwargs):
            if Path(path).name == "metadata.json" and Path(path).parent.name == "new" and mode == "w":
                with real_open(path, mode, *args, **kwargs) as handle:
                    handle.write('{"partial": ')
                raise error
            return real_open(path, mode, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", fail_open)
    else:
        name = failure.removeprefix("publish_")

        def fail_replace(src, dst, *args, **kwargs):
            if Path(src).parent.name == "new" and Path(dst).name == name:
                raise error
            return real_replace(src, dst, *args, **kwargs)

        monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises((OSError, TypeError, ValueError)):
        ds.save(out)
    assert snapshot(out) == before
    assert not list(tmp_path.glob(".dataset.save-*"))
    if state == "complete":
        loaded = FiberNetDataset.load(out)
        assert loaded.samples[0].sample_id == "old-0"
        np.testing.assert_array_equal(loaded.features, [[0], [1]])
        np.testing.assert_array_equal(loaded.all_labels, [[0], [3]])


def test_backup_failure_precedes_any_publication(tmp_path, monkeypatch):
    out = destination(tmp_path, "complete")
    before = snapshot(out)
    original_copy = shutil.copyfile
    error = OSError("backup copy failed")

    def fail_copy(src, dst, *args, **kwargs):
        if Path(dst).parent.name == "original" and Path(dst).name == "labels.npy":
            Path(dst).write_bytes(b"partial backup")
            raise error
        return original_copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "copyfile", fail_copy)
    with pytest.raises(OSError) as caught:
        dataset("new", 100).save(out)
    assert caught.value is error
    assert snapshot(out) == before


@pytest.mark.parametrize("state", ["absent", "partial", "complete"])
def test_secondary_rollback_failure_retains_originals_and_recovery_descriptor(tmp_path, monkeypatch, state):
    out = destination(tmp_path, state)
    before = snapshot(out)
    real_replace, real_unlink = os.replace, Path.unlink
    publication_error = OSError("metadata publication failed")

    def fail_replace(src, dst, *args, **kwargs):
        if Path(src).parent.name == "new" and Path(src).name == "metadata.json":
            raise publication_error
        if Path(src).name == "restore-features.npy":
            raise OSError("restore failed")
        return real_replace(src, dst, *args, **kwargs)

    def fail_unlink(path, *args, **kwargs):
        if state == "absent" and path == out / "features.npy":
            raise OSError("remove failed")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "replace", fail_replace)
    monkeypatch.setattr(Path, "unlink", fail_unlink)
    with pytest.raises(RuntimeError, match="Do not load.*until recovered") as caught:
        dataset("new", 100).save(out)
    error = caught.value
    assert error.__cause__ is publication_error
    recovery = Path(error.recovery_directory)
    info = json.loads((recovery / "recovery.json").read_text())
    assert info["destination"] == str(out.resolve())
    assert info["destination_existed"] == (state != "absent")
    expected_present = {name: state == "complete" or (state == "partial" and name == "features.npy")
                        for name in FILES}
    assert info["original_present"] == expected_present
    for name in FILES:
        backup = recovery / "original" / name
        assert backup.exists() == expected_present[name]
        if backup.exists():
            assert backup.read_bytes() == before[name][0]
            assert hashlib.sha256(backup.read_bytes()).hexdigest() == info["original_sha256"][name]
    backup_before = snapshot(recovery / "original")
    # Bounded manual-recovery rehearsal; originals are copied, never consumed.
    for name in FILES:
        if expected_present[name]:
            restore = recovery / ("manual-" + name)
            shutil.copyfile(recovery / "original" / name, restore)
            os.chmod(restore, info["original_modes"][name])
            real_replace(restore, out / name)
        elif (out / name).exists():
            real_unlink(out / name)
    if state == "absent":
        out.rmdir()
    assert snapshot(out) == before
    assert snapshot(recovery / "original") == backup_before


@pytest.mark.parametrize("warnings_as_errors", [False, True])
@pytest.mark.parametrize("publication_fails", [False, True])
def test_cleanup_does_not_change_commit_or_original_exception(tmp_path, monkeypatch, caplog,
                                                             warnings_as_errors, publication_fails):
    out = destination(tmp_path, "complete")
    before = snapshot(out)
    error = OSError("original publication failure")
    real_replace = os.replace

    def fail_cleanup(path, *args, **kwargs):
        raise OSError("cleanup failed")

    def replace(src, dst, *args, **kwargs):
        if publication_fails and Path(src).parent.name == "new" and Path(src).name == "metadata.json":
            raise error
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", fail_cleanup)
    monkeypatch.setattr(os, "replace", replace)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("error" if warnings_as_errors else "always")
        if publication_fails:
            with pytest.raises(OSError) as caught:
                dataset("new", 100).save(out)
            assert caught.value is error
            assert snapshot(out) == before
        else:
            assert dataset("new", 100).save(out) is None
            assert FiberNetDataset.load(out).samples[0].sample_id == "new-0"
    assert "cleanup failed" in (caplog.text if warnings_as_errors else str(captured[0].message))
    assert len(list(tmp_path.glob(".dataset.save-*"))) == 1


@pytest.mark.parametrize("state", ["empty", "checkpoint", "partial", "complete"])
def test_repeat_save_preserves_sidecars(tmp_path, state):
    out = destination(tmp_path, state)
    sidecars = {str(p.relative_to(out)): p.read_bytes() for p in out.rglob("*")
                if p.is_file() and p.name not in FILES}
    for offset in (100, 200):
        dataset("new", offset).save(out)
        loaded = FiberNetDataset.load(out)
        np.testing.assert_array_equal(loaded.features, [[offset], [offset + 1]])
        for name, contents in sidecars.items():
            assert (out / name).read_bytes() == contents
    assert not list(tmp_path.glob(".dataset.save-*"))


def test_generate_keeps_existing_checkpoint_directory(tmp_path, monkeypatch):
    out = tmp_path / "generated"
    monkeypatch.setitem(sys.modules, "fibernet.sim.accelerated", SimpleNamespace(TaichiEngine=lambda: None))
    for _ in range(2):
        FiberNetDataset.generate(units=[], save_dir=str(out), verbose=False)
        assert (out / "checkpoint.json").is_file()
        assert FiberNetDataset.load(out).n_samples == 0


@pytest.mark.parametrize("kind", ["ordinary", "empty", "empty_declared", "empty_selection"])
def test_saved_format_matches_legacy_bytes_and_loads(tmp_path, kind):
    ds = dataset()
    ds.samples[0].metadata.update({"array": np.array([1, 2]), "tuple": (1, 2),
                                  "float": np.float32(1.5), "legacy_nan": float("nan")})
    if kind == "empty":
        ds = FiberNetDataset([], [], [])
    elif kind == "empty_declared":
        ds = FiberNetDataset([], ["b", "a"], ["y"])
    elif kind == "empty_selection":
        ds = FiberNetDataset(ds.samples, [], [])
    out = tmp_path / "saved"
    ds.save(out)
    for name, values in [("features.npy", ds.features), ("labels.npy", ds.all_labels)]:
        legacy = io.BytesIO()
        np.save(legacy, values)
        assert (out / name).read_bytes() == legacy.getvalue()
    meta = {"feature_names": ds.feature_names, "target_names": ds.target_names,
        "n_samples": ds.n_samples, "samples": [{"id": s.sample_id, "unit": s.unit,
            "grid": list(s.grid), "params": {k: module._json_safe(v) for k, v in s.params.items()},
            "metadata": {k: module._json_safe(v) for k, v in s.metadata.items()}} for s in ds.samples]}
    # Reproduce the historical text writer, including platform newline/encoding.
    legacy_json = tmp_path / "legacy.json"
    with legacy_json.open("w") as handle:
        json.dump(meta, handle, indent=2)
    assert (out / "metadata.json").read_bytes() == legacy_json.read_bytes()
    loaded = FiberNetDataset.load(out)
    np.testing.assert_array_equal(loaded.features, ds.features)
    np.testing.assert_array_equal(loaded.all_labels, ds.all_labels)


@pytest.mark.parametrize("kind", ["managed_symlink", "dangling_symlink", "managed_directory",
                                  "destination_symlink", "destination_file"])
def test_nonregular_targets_fail_before_changes(tmp_path, kind):
    out = destination(tmp_path, "complete")
    if kind == "destination_symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(out, target_is_directory=True)
        target = alias
    elif kind == "destination_file":
        target = tmp_path / "file"
        target.write_text("keep this file")
    else:
        target = out
        (out / "labels.npy").unlink()
        if kind == "managed_directory":
            (out / "labels.npy").mkdir()
            (out / "labels.npy" / "user.txt").write_text("leave alone")
        else:
            other = tmp_path / "other.npy"
            if kind == "managed_symlink":
                other.write_bytes(b"unrelated target")
            (out / "labels.npy").symlink_to(other)
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="regular file|real directory"):
        dataset("new", 100).save(target)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("special", [stat.S_ISUID, stat.S_ISGID, stat.S_ISVTX])
def test_special_mode_bits_rejected_without_applying_them(tmp_path, monkeypatch, special):
    out = destination(tmp_path, "complete")
    before = snapshot(out)
    real_lstat = Path.lstat

    def special_lstat(path, *args, **kwargs):
        info = real_lstat(path, *args, **kwargs)
        return SimpleNamespace(st_mode=info.st_mode | special) if path == out / "labels.npy" else info

    with monkeypatch.context() as patcher:
        patcher.setattr(Path, "lstat", special_lstat)
        with pytest.raises(ValueError, match="special mode"):
            dataset("new", 100).save(out)
    assert snapshot(out) == before


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")
@pytest.mark.parametrize("rollback", [False, True])
def test_ordinary_modes_preserved_only_by_chmod_of_private_files(tmp_path, monkeypatch, rollback):
    out = destination(tmp_path, "complete")
    modes = dict(zip(FILES, [0o640, 0o444, 0o600]))
    for name, mode in modes.items():
        os.chmod(out / name, mode)  # Disposable fixtures only.
    before = snapshot(out)
    real_chmod, real_replace = os.chmod, os.replace
    touched = []

    def observe_chmod(path, mode, *args, **kwargs):
        assert out not in Path(path).parents
        assert ".dataset.save-" in str(path)
        assert mode & 0o7000 == 0
        touched.append(Path(path))
        return real_chmod(path, mode, *args, **kwargs)

    def replace(src, dst, *args, **kwargs):
        if rollback and Path(src).parent.name == "new" and Path(src).name == "metadata.json":
            raise OSError("publish failed")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "chmod", observe_chmod)
    monkeypatch.setattr(os, "replace", replace)
    if rollback:
        with pytest.raises(OSError, match="publish failed"):
            dataset("new", 100).save(out)
        assert snapshot(out) == before
    else:
        dataset("new", 100).save(out)
    assert touched
    assert {name: stat.S_IMODE((out / name).stat().st_mode) for name in FILES} == modes


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", ["short_grid", "writer", "publication"])
def test_csv_late_failures_preserve_old_file_or_absence(tmp_path, monkeypatch, existing, failure):
    path = tmp_path / "dataset.csv"
    if existing:
        dataset().to_csv(path)
    before = snapshot(path)
    ds = dataset("new", 100)
    if failure == "short_grid":
        ds.samples[1].grid = (1,)
    elif failure == "writer":
        writerow = csv.DictWriter.writerow
        calls = []

        def fail_writer(self, row):
            calls.append(row)
            if len(calls) == 3:
                raise OSError("CSV partial write failed")
            return writerow(self, row)

        monkeypatch.setattr(csv.DictWriter, "writerow", fail_writer)
    else:
        def fail_replace(*args, **kwargs):
            raise OSError("CSV replace failed")

        monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises((OSError, IndexError)):
        ds.to_csv(path)
    assert snapshot(path) == before
    assert not list(tmp_path.glob(".dataset.csv.save-*"))


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")
def test_new_files_are_private_and_csv_format_and_existing_mode_are_preserved(tmp_path):
    ds = dataset()
    out = tmp_path / "saved"
    ds.save(out)
    assert all(stat.S_IMODE((out / name).stat().st_mode) == 0o600 for name in FILES)
    path = tmp_path / "dataset.csv"
    ds.to_csv(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.read_bytes() == (b"sample_id,unit,grid_x,grid_y,feat_x,label_y\r\n"
                                 b"old-0,audit,1,1,0,0\r\nold-1,audit,1,1,1,3\r\n")
    os.chmod(path, 0o640)
    ds.to_csv(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o640


@pytest.mark.parametrize("kind", ["symlink", "directory"])
def test_csv_rejects_nonregular_target(tmp_path, kind):
    path = tmp_path / "dataset.csv"
    if kind == "symlink":
        target = tmp_path / "user.csv"
        target.write_text("unrelated\n")
        path.symlink_to(target)
    else:
        path.mkdir()
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="regular file"):
        dataset().to_csv(path)
    assert snapshot(tmp_path) == before


def test_csv_quoted_multiline_and_unicode_match_legacy_writer(tmp_path):
    ds = dataset()
    ds.samples[0].sample_id = 'sample,"quoted"\nnext line'
    ds.samples[0].unit = 'unit,\r\nµ'
    expected = tmp_path / "legacy.csv"
    # Match the original CSV writer: no newline translation, locale encoding.
    with expected.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "unit", "grid_x", "grid_y", "feat_x", "label_y"])
        writer.writeheader()
        for sample in ds.samples:
            writer.writerow({"sample_id": sample.sample_id, "unit": sample.unit,
                "grid_x": sample.grid[0], "grid_y": sample.grid[1],
                "feat_x": sample.features["x"], "label_y": sample.labels["y"]})
    path = tmp_path / "quoted.csv"
    ds.to_csv(path)
    assert path.read_bytes() == expected.read_bytes()
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["sample_id"] == ds.samples[0].sample_id
    assert rows[0]["unit"] == ds.samples[0].unit


@pytest.mark.parametrize("commit", [False, True])
def test_csv_cleanup_and_warning_filter_do_not_change_outcome(tmp_path, monkeypatch, caplog, commit):
    path = tmp_path / "dataset.csv"
    dataset().to_csv(path)
    before = snapshot(path)
    real_replace, real_unlink = os.replace, Path.unlink
    publication_error = OSError("CSV publication failed")

    def replace(src, dst, *args, **kwargs):
        if not commit:
            raise publication_error
        return real_replace(src, dst, *args, **kwargs)

    def unlink(temporary, *args, **kwargs):
        if temporary.name.startswith(".dataset.csv.save-"):
            raise OSError("CSV cleanup failed")
        return real_unlink(temporary, *args, **kwargs)

    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(Path, "unlink", unlink)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        if commit:
            assert dataset("new", 100).to_csv(path) is None
            assert b"new-0" in path.read_bytes()
        else:
            with pytest.raises(OSError) as caught:
                dataset("new", 100).to_csv(path)
            assert caught.value is publication_error
            assert snapshot(path) == before
    assert "CSV cleanup failed" in caplog.text


def test_failed_private_mode_copy_preserves_destination(tmp_path, monkeypatch):
    out = destination(tmp_path, "complete")
    before = snapshot(out)

    def fail_mode(path, mode):
        assert ".dataset.save-" in str(path)
        raise OSError("private staged mode copy failed")

    monkeypatch.setattr(os, "chmod", fail_mode)
    with pytest.raises(OSError, match="mode copy failed"):
        dataset("new", 100).save(out)
    assert snapshot(out) == before


def test_warning_emission_failure_does_not_replace_original_error(tmp_path, monkeypatch, caplog):
    out = destination(tmp_path, "complete")
    before = snapshot(out)
    original_error = OSError("initial staged write failed")

    def fail_write(*args, **kwargs):
        raise original_error

    def fail_cleanup(*args, **kwargs):
        raise OSError("cleanup failed")

    def fail_warning(*args, **kwargs):
        raise RuntimeError("custom warning handler failed")

    monkeypatch.setattr(np, "save", fail_write)
    monkeypatch.setattr(shutil, "rmtree", fail_cleanup)
    monkeypatch.setattr(warnings, "warn", fail_warning)
    with pytest.raises(OSError) as caught:
        dataset("new", 100).save(out)
    assert caught.value is original_error
    assert snapshot(out) == before
    assert "cleanup failed" in caplog.text


@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
def test_interruption_after_replacement_retains_originals_without_claiming_rollback(tmp_path, monkeypatch,
                                                                                   interrupt_type):
    out = destination(tmp_path, "complete")
    before = snapshot(out)
    real_replace = os.replace
    interruption = interrupt_type("injected publication interruption")

    def interrupted_replace(src, dst, *args, **kwargs):
        result = real_replace(src, dst, *args, **kwargs)
        if Path(src).parent.name == "new" and Path(src).name == "features.npy":
            # The filesystem operation completed before Python recorded it.
            raise interruption
        return result

    monkeypatch.setattr(os, "replace", interrupted_replace)
    with pytest.raises(interrupt_type) as caught:
        dataset("new", 100).save(out)
    assert caught.value is interruption
    assert (out / "features.npy").read_bytes() != before["features.npy"][0]
    workspaces = list(tmp_path.glob(".dataset.save-*"))
    assert len(workspaces) == 1
    info = json.loads((workspaces[0] / "recovery.json").read_text())
    assert all(info["original_present"].values())
    for name in FILES:
        original = workspaces[0] / "original" / name
        assert original.read_bytes() == before[name][0]
        assert hashlib.sha256(original.read_bytes()).hexdigest() == info["original_sha256"][name]


def test_interruption_during_ordinary_rollback_retains_originals(tmp_path, monkeypatch):
    out = destination(tmp_path, "complete")
    before = snapshot(out)
    real_replace = os.replace

    def replace(src, dst, *args, **kwargs):
        if Path(src).parent.name == "new" and Path(src).name == "metadata.json":
            raise OSError("ordinary publication failure")
        if Path(src).name.startswith("restore-"):
            raise KeyboardInterrupt("interrupted rollback")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", replace)
    with pytest.raises(KeyboardInterrupt, match="interrupted rollback"):
        dataset("new", 100).save(out)
    workspaces = list(tmp_path.glob(".dataset.save-*"))
    assert len(workspaces) == 1
    for name in FILES:
        assert (workspaces[0] / "original" / name).read_bytes() == before[name][0]


def test_failed_fresh_save_preserves_destination_absence_but_can_leave_parents(tmp_path, monkeypatch):
    out = tmp_path / "new-parent" / "another-parent" / "dataset"

    def fail_write(*args, **kwargs):
        raise OSError("staged write failed")

    monkeypatch.setattr(np, "save", fail_write)
    with pytest.raises(OSError, match="staged write failed"):
        dataset().save(out)
    assert not out.exists()
    assert out.parent.is_dir()
    assert not list(out.parent.glob(".dataset.save-*"))
