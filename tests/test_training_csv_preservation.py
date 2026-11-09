"""CSV training inputs must survive model publication at aliased paths."""
import builtins
import os
from pathlib import Path
import pickle
import tempfile

import pytest

pytest.importorskip("pandas")
pytest.importorskip("sklearn")
from sklearn.linear_model import LinearRegression
import sklearn.model_selection

import fibernet.easy as easy


def csv_input(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text("x,y\n" + "".join(f"{i},{2*i+1}\n" for i in range(16)), encoding="utf-8")
    return source


def train(source, destination):
    return easy.train_model(source, target="y", features=["x"], model_type="lr",
                            test_size=.25, save_path=destination)


def symlink(link, target, *, directory=False):
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Symlinks unavailable: {exc}")


@pytest.mark.parametrize("alias", ["path", "str", "bytes_destination", "relative",
                                   "dotdot", "input_symlink", "output_symlink",
                                   "input_parent_symlink", "output_parent_symlink"])
def test_csv_alias_rejects_before_split_fit_or_save(tmp_path, monkeypatch, alias):
    original = csv_input(tmp_path)
    source = destination = original
    if alias == "str":
        source = destination = str(original)
    elif alias == "bytes_destination":
        destination = os.fsencode(original)
    elif alias == "relative":
        monkeypatch.chdir(tmp_path)
        destination = "input.csv"
    elif alias == "dotdot":
        (tmp_path / "nested").mkdir()
        destination = tmp_path / "nested" / ".." / "input.csv"
    elif alias in ("input_symlink", "output_symlink"):
        link = tmp_path / "alias.csv"
        symlink(link, original)
        if alias == "input_symlink":
            source = link
        else:
            destination = link
    elif alias.endswith("parent_symlink"):
        link = tmp_path / "alias-directory"
        symlink(link, tmp_path, directory=True)
        if alias == "input_parent_symlink":
            source = link / original.name
        else:
            destination = link / original.name
    before = original.read_bytes(), original.stat().st_ino
    entries = sorted(path.name for path in tmp_path.iterdir())

    def forbidden(*args, **kwargs):
        pytest.fail("Colliding CSV reached splitting, fitting, or model staging")
    monkeypatch.setattr(sklearn.model_selection, "train_test_split", forbidden)
    monkeypatch.setattr(LinearRegression, "fit", forbidden)
    monkeypatch.setattr(tempfile, "mkstemp", forbidden)
    monkeypatch.setattr(easy, "_save_model_artifact", forbidden)
    real_import = builtins.__import__

    def import_without_ml(name, *args, **kwargs):
        if name.split(".")[0] in ("pandas", "sklearn"):
            pytest.fail("Colliding CSV reached optional training imports")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", import_without_ml)
    with pytest.raises(ValueError, match="save_path.*input CSV"):
        train(source, destination)
    assert (original.read_bytes(), original.stat().st_ino) == before
    assert sorted(path.name for path in tmp_path.iterdir()) == entries


@pytest.mark.parametrize("destination_kind", ["new", "existing", "hardlink", "same_basename"])
def test_distinct_destination_preserves_csv_and_training(tmp_path, destination_kind):
    source = csv_input(tmp_path)
    before = source.read_bytes(), source.stat().st_ino
    destination = tmp_path / "model.pkl"
    if destination_kind == "existing":
        destination.write_bytes(b"previous model")
    elif destination_kind == "hardlink":
        try:
            os.link(source, destination)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"Hardlinks unavailable: {exc}")
    elif destination_kind == "same_basename":
        (tmp_path / "models").mkdir()
        destination = tmp_path / "models" / source.name
    reference = train(source, None)
    metrics = train(source, destination)
    assert metrics == reference
    assert (source.read_bytes(), source.stat().st_ino) == before
    artifact = pickle.loads(destination.read_bytes())  # Only this trusted fixture.
    assert artifact["metrics"] == metrics
    assert artifact["features"] == ["x"]
    assert artifact["target"] == "y"
    assert artifact["artifact_version"] == 1
    assert destination.stat().st_ino != source.stat().st_ino
    assert not list(destination.parent.glob(f".{destination.name}.save-*"))


def test_csv_input_symlink_can_save_elsewhere(tmp_path):
    source = csv_input(tmp_path)
    link = tmp_path / "alias.csv"
    symlink(link, source)
    before = source.read_bytes(), source.stat().st_ino
    destination = tmp_path / "model.pkl"
    train(link, destination)
    assert link.is_symlink()
    assert (source.read_bytes(), source.stat().st_ino) == before
    assert pickle.loads(destination.read_bytes())["target"] == "y"


@pytest.mark.parametrize("destination", [None, ""])
def test_omitted_model_destination_keeps_csv_unchanged(tmp_path, destination):
    source = csv_input(tmp_path)
    before = source.read_bytes(), source.stat().st_ino
    assert train(source, destination)["n_samples"] == 16
    assert (source.read_bytes(), source.stat().st_ino) == before
    assert list(tmp_path.iterdir()) == [source]
