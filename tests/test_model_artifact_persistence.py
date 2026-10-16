"""One-file model publication: preserve bytes/absence through ordinary failures."""
import os
from pathlib import Path
import pickle
import socket
import stat
import tempfile
import warnings

import numpy as np
import pytest

pytest.importorskip("sklearn")
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import fibernet.easy as easy
from fibernet.ml.dataset import DatasetSample, FiberNetDataset


@pytest.fixture(params=["csv", "native"])
def trainer(request, tmp_path):
    rows = [DatasetSample(f"row-{i}", "fixture", (1, 1), {},
                          {"x": float(i), "z": float(i % 5) ** 2},
                          {"y": float(2*i + 3*(i % 5)**2)}, {})
            for i in range(24)]
    dataset = FiberNetDataset(rows, ["x", "z"], ["y"])
    source = dataset
    if request.param == "csv":
        pytest.importorskip("pandas")
        source = tmp_path / "input.csv"
        source.write_text("x,z,y\n" + "".join(
            f"{s.features['x']},{s.features['z']},{s.labels['y']}\n" for s in rows))

    def train(destination, **kwargs):
        options = dict(target="y", features=["z", "x"], model_type="lr",
                       test_size=.25, save_path=destination)
        if request.param == "native":
            options["split"] = "row"
        options.update(kwargs)
        return easy.train_model(source, **options)

    train.kind = request.param
    train.dataset = dataset
    return train


def snapshot(path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    return path.read_bytes(), stat.S_IMODE(info.st_mode), info.st_ino


def stages(path):
    return list(path.parent.glob(f".{path.name}.save-*"))


@pytest.mark.parametrize("exists", [False, True])
@pytest.mark.parametrize("failure", ["serialization", "partial_pickle", "after_pickle",
                                    "partial_write", "close", "fdopen", "create", "chmod", "replace"])
def test_failures_preserve_exact_old_artifact_or_absence(trainer, tmp_path, monkeypatch,
                                                        exists, failure):
    destination = tmp_path / "model.pkl"
    if exists:
        trainer(destination)
        destination.chmod(0o640)
    before = snapshot(destination)
    unrelated = tmp_path / ".model.pkl.save-unrelated"
    unrelated.write_bytes(b"unrelated user file")
    unrelated.chmod(0o604)
    unrelated_before = snapshot(unrelated)
    error = OSError(f"injected {failure}")
    real_dump, real_fdopen = pickle.dump, os.fdopen
    wrote = []
    unclaimed_fds = []

    if failure in ("serialization", "partial_pickle", "after_pickle"):
        def dump(value, handle, *args, **kwargs):
            if failure == "serialization":
                # Real pickle fails on the lambda after writing a large earlier item.
                value = dict(value, large=b"x" * 100_000, unsupported=lambda: None)
                try:
                    return real_dump(value, handle, *args, **kwargs)
                finally:
                    handle.flush()
                    wrote.append(os.fstat(handle.fileno()).st_size)
            if failure == "partial_pickle":
                class PartialWriter:
                    def write(self, data):
                        handle.write(data[:31])
                        handle.flush()
                        wrote.append(os.fstat(handle.fileno()).st_size)
                        raise error
                return real_dump(value, PartialWriter(), *args, **kwargs)
            real_dump(value, handle, *args, **kwargs)
            handle.flush()
            wrote.append(os.fstat(handle.fileno()).st_size)
            raise error
        monkeypatch.setattr(pickle, "dump", dump)
    elif failure in ("partial_write", "close", "fdopen"):
        def fdopen(fd, *args, **kwargs):
            if failure == "fdopen":
                unclaimed_fds.append(fd)
                raise error
            handle = real_fdopen(fd, *args, **kwargs)

            class Writer:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    handle.__exit__(*args)
                    if failure == "close":
                        raise error

                def write(self, data):
                    if failure == "partial_write":
                        handle.write(data[:19])
                        handle.flush()
                        wrote.append(os.fstat(handle.fileno()).st_size)
                        raise error
                    return handle.write(data)
            return Writer()
        monkeypatch.setattr(os, "fdopen", fdopen)
    else:
        def fail(*args, **kwargs):
            raise error
        if failure == "create":
            monkeypatch.setattr(tempfile, "mkstemp", fail)
        else:
            monkeypatch.setattr(os, failure, fail)

    expected = (AttributeError, pickle.PicklingError) if failure == "serialization" else OSError
    with pytest.raises(expected) as caught:
        trainer(destination)
    if failure != "serialization":
        assert caught.value is error
    if failure in ("serialization", "partial_pickle", "after_pickle", "partial_write"):
        assert wrote and wrote[0] > 0
    assert snapshot(destination) == before
    assert snapshot(unrelated) == unrelated_before
    assert stages(destination) == [unrelated]
    for fd in unclaimed_fds:
        with pytest.raises(OSError):
            os.fstat(fd)
    if exists:
        # Only our own trusted fixture is deserialized.
        restored = pickle.loads(destination.read_bytes())
        assert restored["artifact_version"] == 1
        assert restored["features"] == ["z", "x"]


@pytest.mark.parametrize("kind", ["symlink", "dangling", "directory", "fifo", "socket"])
def test_nonregular_targets_reject_before_stage_or_pickle(trainer, tmp_path, monkeypatch, kind):
    destination = tmp_path / "model.pkl"
    other = tmp_path / "other.pkl"
    other.write_bytes(b"leave other bytes")
    other_before = snapshot(other)
    sock = None
    if kind in ("symlink", "dangling"):
        destination.symlink_to(other if kind == "symlink" else tmp_path / "absent.pkl")
    elif kind == "directory":
        destination.mkdir()
        (destination / "notes.txt").write_text("leave nested notes")
    elif kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO creation requires POSIX")
        os.mkfifo(destination)
    else:
        if not hasattr(socket, "AF_UNIX"):
            pytest.skip("Unix sockets unavailable")
        try:
            sock = socket.socket(socket.AF_UNIX)
        except PermissionError:
            pytest.skip("Runtime disallows Unix socket creation")
        sock.bind(str(destination))
    info = destination.lstat()

    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported target entered staging or serialization")
    monkeypatch.setattr(tempfile, "mkstemp", forbidden)
    monkeypatch.setattr(pickle, "dump", forbidden)
    try:
        with pytest.raises(ValueError, match="regular file"):
            trainer(destination)
    finally:
        if sock is not None:
            sock.close()
    assert destination.lstat() == info
    assert snapshot(other) == other_before
    if kind == "directory":
        assert (destination / "notes.txt").read_text() == "leave nested notes"
    if kind == "dangling":
        assert not (tmp_path / "absent.pkl").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
@pytest.mark.parametrize("special", [0o4000, 0o2000, 0o1000, 0o7000])
def test_special_permission_bits_reject_without_mutation(trainer, tmp_path, monkeypatch, special):
    destination = tmp_path / "model.pkl"
    destination.write_bytes(b"preserve special-mode bytes")
    destination.chmod(0o640 | special)
    before = snapshot(destination)

    def forbidden(*args, **kwargs):
        pytest.fail("Special target entered staging, serialization, or metadata mutation")
    monkeypatch.setattr(tempfile, "mkstemp", forbidden)
    monkeypatch.setattr(pickle, "dump", forbidden)
    monkeypatch.setattr(os, "chmod", forbidden)
    with pytest.raises(ValueError, match="special mode bits"):
        trainer(destination)
    assert snapshot(destination) == before


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
@pytest.mark.parametrize("mode", [None, 0o400, 0o640, 0o751, 0o777])
def test_success_private_stage_preserves_only_ordinary_modes(trainer, tmp_path, monkeypatch, mode):
    destination = tmp_path / "model.pkl"
    if mode is not None:
        destination.write_bytes(b"old contents")
        destination.chmod(mode)
    before = snapshot(destination)
    real_dump, real_replace, real_chmod = pickle.dump, os.replace, os.chmod
    handles, chmod_calls = [], []

    def dump(value, handle, *args, **kwargs):
        handles.append(handle)
        assert stat.S_IMODE(os.fstat(handle.fileno()).st_mode) & ~0o600 == 0
        assert snapshot(destination) == before
        return real_dump(value, handle, *args, **kwargs)

    def chmod(path, value, *args, **kwargs):
        assert Path(path) != destination
        assert Path(path) in stages(destination)
        assert handles[0].closed
        chmod_calls.append((Path(path), value))
        return real_chmod(path, value, *args, **kwargs)

    def replace(src, dst, *args, **kwargs):
        assert Path(src).parent == destination.parent
        assert Path(dst) == destination
        assert handles[0].closed
        assert snapshot(destination) == before
        artifact = pickle.loads(Path(src).read_bytes())
        assert artifact["features"] == ["z", "x"]
        assert stat.S_IMODE(Path(src).stat().st_mode) == (0o600 if mode is None else mode)
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(pickle, "dump", dump)
    monkeypatch.setattr(os, "chmod", chmod)
    monkeypatch.setattr(os, "replace", replace)
    trainer(destination)
    after = snapshot(destination)
    assert after[1] == (0o600 if mode is None else mode)
    assert before is None or before[2] != after[2]
    assert len(chmod_calls) == 1
    assert not stages(destination)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_zero_permission_mode_is_preserved_without_reading_original(trainer, tmp_path):
    destination = tmp_path / "model.pkl"
    destination.write_bytes(b"original inaccessible bytes")
    destination.chmod(0o000)
    original_inode = destination.stat().st_ino
    trainer(destination)
    assert stat.S_IMODE(destination.stat().st_mode) == 0o000
    assert destination.stat().st_ino != original_inode
    assert not stages(destination)
    # Test-only access restoration to inspect the new fixture after mode checks.
    destination.chmod(0o600)
    assert pickle.loads(destination.read_bytes())["target"] == "y"


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
@pytest.mark.parametrize("mask", [0o000, 0o077, 0o777])
def test_new_artifact_mode_is_0600_regardless_of_umask(trainer, tmp_path, mask):
    destination = tmp_path / "model.pkl"
    previous = os.umask(mask)
    try:
        trainer(destination)
    finally:
        os.umask(previous)
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert pickle.loads(destination.read_bytes())["target"] == "y"


@pytest.mark.parametrize("model_type,cls,settings", [
    ("lr", LinearRegression, {}),
    ("rf", RandomForestRegressor, {"n_estimators": 100, "random_state": 42}),
    ("gb", GradientBoostingRegressor, {"n_estimators": 100, "random_state": 42}),
])
def test_success_envelope_training_and_predictions_unchanged(trainer, tmp_path, model_type, cls, settings):
    destination = tmp_path / "model.pkl"
    X = np.asarray([[s.features["z"], s.features["x"]] for s in trainer.dataset.samples])
    y = np.asarray([s.labels["y"] for s in trainer.dataset.samples])
    train, test = train_test_split(np.arange(24), test_size=.25, random_state=42)
    reference = cls(**settings).fit(X[train], y[train])
    result = trainer(destination, model_type=model_type)
    artifact = pickle.loads(destination.read_bytes())
    assert set(artifact) == {"model", "features", "metrics", "artifact_version", "target"}
    assert artifact["artifact_version"] == 1
    assert artifact["features"] == ["z", "x"]
    assert artifact["target"] == "y"
    assert artifact["metrics"] == result
    assert type(artifact["model"]) is cls
    assert result == trainer(None, model_type=model_type)
    np.testing.assert_array_equal(artifact["model"].predict(X), reference.predict(X))
    records = [{"x": s.features["x"], "z": s.features["z"]} for s in trainer.dataset.samples]
    prediction = easy.predict_named(artifact, records)
    np.testing.assert_array_equal(prediction["predictions"], reference.predict(X))
    assert prediction["target"] == "y"
    assert not stages(destination)


@pytest.mark.parametrize("path_type", ["path", "str", "bytes", "relative"])
def test_supported_path_forms(trainer, tmp_path, monkeypatch, path_type):
    destination = tmp_path / "model.pkl"
    value = destination
    if path_type == "str":
        value = str(destination)
    elif path_type == "bytes":
        value = os.fsencode(destination)
    elif path_type == "relative":
        # CSV source was captured as an absolute path by the fixture.
        monkeypatch.chdir(tmp_path)
        value = "model.pkl"
    trainer(value)
    assert pickle.loads(destination.read_bytes())["target"] == "y"


def test_missing_parent_is_not_created(trainer, tmp_path):
    parent = tmp_path / "missing" / "nested"
    with pytest.raises(FileNotFoundError):
        trainer(parent / "model.pkl")
    assert not (tmp_path / "missing").exists()


def test_both_training_paths_use_one_helper(trainer, tmp_path, monkeypatch):
    destination = tmp_path / "model.pkl"
    calls = []
    real_save = easy._save_model_artifact

    def save(path, artifact):
        calls.append((path, artifact))
        return real_save(path, artifact)
    monkeypatch.setattr(easy, "_save_model_artifact", save)
    result = trainer(destination)
    assert len(calls) == 1
    assert calls[0][0] == destination
    assert calls[0][1]["metrics"] is result


@pytest.mark.skipif(os.name != "posix", reason="POSIX hardlinks and modes")
def test_hardlink_alias_keeps_old_inode_bytes_and_mode(trainer, tmp_path):
    destination = tmp_path / "model.pkl"
    trainer(destination)
    destination.chmod(0o641)
    alias = tmp_path / "old-model.pkl"
    os.link(destination, alias)
    before = snapshot(alias)
    trainer(destination, model_type="gb")
    assert snapshot(alias) == before
    assert destination.stat().st_ino != alias.stat().st_ino
    assert stat.S_IMODE(destination.stat().st_mode) == 0o641


@pytest.mark.parametrize("publication_fails", [False, True])
@pytest.mark.parametrize("warnings_as_errors", [False, True])
def test_cleanup_failure_never_changes_save_outcome(trainer, tmp_path, monkeypatch, caplog,
                                                    publication_fails, warnings_as_errors):
    destination = tmp_path / "model.pkl"
    trainer(destination)
    before = snapshot(destination)
    error = OSError("original replacement failure")
    real_replace = os.replace
    cleanup_paths = []

    def replace(src, dst):
        if publication_fails:
            raise error
        return real_replace(src, dst)

    def unlink(path, *args, **kwargs):
        cleanup_paths.append(Path(path))
        raise OSError("cleanup failed")
    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(os, "unlink", unlink)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error" if warnings_as_errors else "always", RuntimeWarning)
        if publication_fails:
            with pytest.raises(OSError) as failure:
                trainer(destination, model_type="gb")
            assert failure.value is error
            assert snapshot(destination) == before
        else:
            result = trainer(destination, model_type="gb")
            assert pickle.loads(destination.read_bytes())["metrics"] == result
            assert snapshot(destination) != before
    assert len(cleanup_paths) == 1
    assert cleanup_paths[0].parent == destination.parent
    assert cleanup_paths[0] != destination
    assert len(stages(destination)) == (1 if publication_fails else 0)
    if warnings_as_errors:
        assert "Model artifact cleanup failed" in caplog.text
    else:
        assert any("Model artifact cleanup failed" in str(item.message) for item in caught)
    diagnostic = caplog.text if warnings_as_errors else str(caught[0].message)
    assert str(cleanup_paths[0]) in diagnostic
    assert "save outcome is unchanged" in diagnostic


@pytest.mark.parametrize("publication_fails", [False, True])
def test_cleanup_warning_and_logging_failures_do_not_mask_outcome(trainer, tmp_path, monkeypatch,
                                                                 publication_fails):
    import logging
    destination = tmp_path / "model.pkl"
    trainer(destination)
    before = snapshot(destination)
    error = OSError("original failure")

    def unlink(*args, **kwargs):
        raise OSError("cleanup failed")

    def log(*args, **kwargs):
        raise RuntimeError("broken logging handler")

    def replace(*args, **kwargs):
        raise error
    monkeypatch.setattr(os, "unlink", unlink)
    monkeypatch.setattr(logging.getLogger(easy.__name__), "warning", log)
    if publication_fails:
        monkeypatch.setattr(os, "replace", replace)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        if publication_fails:
            with pytest.raises(OSError) as caught:
                trainer(destination)
            assert caught.value is error
            assert snapshot(destination) == before
        else:
            trainer(destination)
            assert snapshot(destination) != before
