"""prepare-data: the run's dataset, or a reproducible subset of it, as the dataset `prepared` (PRD kaos#1911 X-5)."""
import shutil

import pytest
import yaml
from conftest import ROOT, import_in, make_dataset
from kaos_lib import cli, testing


def files(folder):
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())


def images(folder, split):
    return sorted(p.name for p in (folder / "images" / split).iterdir())


def labels(folder, split):
    return sorted(p.name for p in (folder / "labels" / split).iterdir())


def run_entry(tmp_path, dataset, **data):
    """The step in this process, with the settings `data.sample_size` and `data.seed` and the dataset as its input."""
    entry = import_in("prepare_data", "app.entry")
    settings = {"sample_size": "", "seed": 42, **data}
    with testing.running(tmp_path / "run", kind="transform", params={"data": settings}, inputs={"dataset": dataset},
                         produces=[("dataset", "prepared")]) as run:
        entry.main()
    return run


# ------------------------------------------------------------------------------------- no subset

def test_without_a_subset_every_file_is_kept_unchanged(tmp_path, dataset):
    run = run_entry(tmp_path, dataset)
    assert files(run.output("prepared")) == files(dataset)
    assert (run.output("prepared") / "data.yaml").read_text() == (dataset / "data.yaml").read_text()
    assert run.result["values"]["splits"] == {"test": {"kept": 2, "of": 2}, "train": {"kept": 6, "of": 6},
                                              "val": {"kept": 3, "of": 3}}


def test_without_a_subset_the_files_are_not_copied_when_the_disk_is_shared(tmp_path, dataset):
    """A file of the output is the file of the input (a hard link), so a big dataset costs no second copy."""
    subset = import_in("prepare_data", "app.subset")
    subset.write(dataset, tmp_path / "out", None, 1)
    image = "images/train/train00.jpg"
    assert (tmp_path / "out" / image).stat().st_ino == (dataset / image).stat().st_ino


# ------------------------------------------------------------------------------------- a subset

def test_a_subset_keeps_that_many_images_per_split_with_their_labels(tmp_path, dataset):
    out = run_entry(tmp_path, dataset, sample_size="2", seed=7).output("prepared")
    assert [len(images(out, s)) for s in ("train", "val", "test")] == [2, 2, 2]
    for split in ("train", "val", "test"):
        assert [n[:-4] for n in images(out, split)] == [n[:-4] for n in labels(out, split)], "each image keeps its label"
    assert (out / "data.yaml").read_text() == (dataset / "data.yaml").read_text()
    assert (out / "README.md").exists(), "a file outside images/ and labels/ is kept"


def test_the_same_seed_keeps_the_same_files_and_another_seed_other_ones(tmp_path, dataset):
    first = images(run_entry(tmp_path / "a", dataset, sample_size="3", seed=1).output("prepared"), "train")
    again = images(run_entry(tmp_path / "b", dataset, sample_size="3", seed=1).output("prepared"), "train")
    other = [images(run_entry(tmp_path / f"s{seed}", dataset, sample_size="3", seed=seed).output("prepared"), "train")
             for seed in range(2, 12)]
    assert first == again
    assert any(picked != first for picked in other), "some other seed keeps other files"


def test_a_subset_is_the_same_on_another_machine(tmp_path, dataset):
    """The pick depends on the seed and the file's path only, so it is the same where the files are stored differently."""
    copy = make_dataset(tmp_path / "copy")
    assert images(run_entry(tmp_path / "a", dataset, sample_size="3", seed=5).output("prepared"), "train") == \
        images(run_entry(tmp_path / "b", copy, sample_size="3", seed=5).output("prepared"), "train")


def test_a_bigger_subset_than_the_dataset_keeps_everything(tmp_path, dataset):
    assert files(run_entry(tmp_path, dataset, sample_size="500").output("prepared")) == files(dataset)


def test_a_subset_that_is_a_number_not_a_text_works(tmp_path, dataset):
    assert len(images(run_entry(tmp_path, dataset, sample_size=1).output("prepared"), "train")) == 1


@pytest.mark.parametrize("size", ["0", "-3"])
def test_a_size_that_keeps_nothing_is_refused(tmp_path, dataset, size):
    with pytest.raises(SystemExit, match="sample_size"):
        run_entry(tmp_path, dataset, sample_size=size)


def test_a_folder_that_is_not_a_yolo_dataset_is_refused(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "notes.txt").write_text("x")
    with pytest.raises(SystemExit, match="data.yaml"):
        run_entry(tmp_path, tmp_path / "raw")


# ------------------------------------------------------------------------------------- the step as `kaosctl step run`

def test_kaosctl_step_run_prepares_the_dataset_and_training_finds_it(tmp_path, dataset, monkeypatch):
    """The two steps one after the other on this machine: what prepare-data wrote is the input of model-training."""
    params = tmp_path / "params.yaml"
    params.write_text(yaml.safe_dump({"data": {"sample_size": "2", "seed": 3}}))
    monkeypatch.chdir(tmp_path)
    app = tmp_path / "app"
    (app / "steps").mkdir(parents=True)
    for name in ("prepare_data", "model_training"):  # a copy: `kaosctl step run` makes a Python environment in the step's folder
        shutil.copytree(ROOT / "steps" / name, app / "steps" / name, ignore=shutil.ignore_patterns("__pycache__", ".venv"))
    (app / "pipeline.py").write_text((ROOT / "pipeline.py").read_text())
    code = cli.main(["step", "run", "prepare-data", str(app), "--params-file", str(params), "--input",
                     f"dataset={dataset}"])
    assert code == 0
    prepared = app / ".kaos" / "local" / "staged" / "prepared"
    assert len(images(prepared, "train")) == 2 and (prepared / "data.yaml").is_file()
