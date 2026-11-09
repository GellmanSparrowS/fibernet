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


@pytest.mark.parametrize("alias", ["case", "unicode", "parent_case"])
def test_filesystem_lookup_aliases_reject_when_supported(tmp_path, monkeypatch, alias):
    directory = tmp_path / "Inputs"
    directory.mkdir()
    source = csv_input(directory)
    if alias == "case":
        destination = directory / "INPUT.csv"
    elif alias == "unicode":
        source = source.rename(directory / "caf\u00e9.csv")
        destination = directory / "cafe\u0301.csv"
    else:
        destination = tmp_path / "inputs" / source.name
    if not destination.exists() or not source.samefile(destination):
        pytest.skip(f"Filesystem does not provide {alias} lookup aliases")
    before = source.read_bytes(), source.stat().st_ino

    def forbidden(*args, **kwargs):
        pytest.fail("Filesystem lookup alias reached model training or saving")
    monkeypatch.setattr(LinearRegression, "fit", forbidden)
    monkeypatch.setattr(easy, "_save_model_artifact", forbidden)
    with pytest.raises(ValueError, match="save_path.*input CSV"):
        train(source, destination)
    assert (source.read_bytes(), source.stat().st_ino) == before


@pytest.mark.parametrize("alias", ["case", "unicode", "parent_case"])
def test_lookup_alias_identity_model_rejects_before_training(tmp_path, monkeypatch, alias):
    # Exercise insensitive lookup semantics even on a case-sensitive CI volume.
    # The capability-gated test above additionally covers actual filesystem IO.
    import unicodedata
    directory = tmp_path / "Inputs"
    directory.mkdir()
    source = csv_input(directory)
    if alias == "case":
        destination = directory / "INPUT.csv"
    elif alias == "unicode":
        source = source.rename(directory / "caf\u00e9.csv")
        destination = directory / "cafe\u0301.csv"
    else:
        destination = tmp_path / "inputs" / source.name
    before = source.read_bytes(), source.stat().st_ino
    original = Path.samefile

    def samefile(path, other):
        key = lambda value: unicodedata.normalize("NFD", str(value)).casefold()
        return key(path) == key(other) or original(path, other)
    monkeypatch.setattr(Path, "samefile", samefile)

    def forbidden(*args, **kwargs):
        pytest.fail("Modeled lookup alias reached training or staging")
    monkeypatch.setattr(LinearRegression, "fit", forbidden)
    monkeypatch.setattr(easy, "_save_model_artifact", forbidden)
    with pytest.raises(ValueError, match="save_path.*input CSV"):
        train(source, destination)
    assert (source.read_bytes(), source.stat().st_ino) == before


@pytest.mark.parametrize("alternate", ["INPUT.csv", "cafe\u0301.csv"])
def test_distinct_case_or_unicode_hardlink_entries_remain_supported(tmp_path, alternate):
    source = csv_input(tmp_path)
    if alternate.startswith("cafe"):
        source = source.rename(tmp_path / "caf\u00e9.csv")
    destination = tmp_path / alternate
    if destination.exists():
        pytest.skip("Filesystem folds these spellings to the existing entry")
    try:
        os.link(source, destination)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Hardlinks unavailable: {exc}")
    before = source.read_bytes(), source.stat().st_ino
    train(source, destination)
    assert (source.read_bytes(), source.stat().st_ino) == before
    assert pickle.loads(destination.read_bytes())["target"] == "y"
    assert not source.samefile(destination)


@pytest.mark.parametrize("kind", ["missing", "separate_file", "other_parent_hardlink"])
def test_ordinary_destinations_do_not_scan_directories(tmp_path, monkeypatch, kind):
    source = csv_input(tmp_path)
    destination = tmp_path / "model.pkl"
    if kind == "separate_file":
        destination.write_bytes(b"previous model")
    elif kind == "other_parent_hardlink":
        (tmp_path / "models").mkdir()
        destination = tmp_path / "models" / "model.pkl"
        try:
            os.link(source, destination)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"Hardlinks unavailable: {exc}")

    def forbidden(*args, **kwargs):
        pytest.fail("Unambiguous destination required directory enumeration")
    monkeypatch.setattr(os, "scandir", forbidden)
    assert easy._model_save_overwrites_csv(source, destination) is False


def test_unreadable_ambiguous_directory_fails_before_training(tmp_path, monkeypatch):
    source = csv_input(tmp_path)
    destination = tmp_path / "model.pkl"
    try:
        os.link(source, destination)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Hardlinks unavailable: {exc}")
    before = source.read_bytes(), source.stat().st_ino

    def denied(*args, **kwargs):
        raise PermissionError("directory entries are unreadable")
    def forbidden(*args, **kwargs):
        pytest.fail("Unverified same-inode destination reached training")
    monkeypatch.setattr(os, "scandir", denied)
    monkeypatch.setattr(LinearRegression, "fit", forbidden)
    with pytest.raises(PermissionError, match="directory entries"):
        train(source, destination)
    assert (source.read_bytes(), source.stat().st_ino) == before
    assert destination.read_bytes() == before[0]
