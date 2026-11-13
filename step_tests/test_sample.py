"""The small sample a first run trains on (PRD kaos#1911): exact labels, the yolo layout, small, the same every time,
and used only for the project's default dataset when it holds no YOLO dataset."""
import struct
import zlib

import pytest
from conftest import import_in
from kaos_lib import datasets, layouts, testing

BOX, KEYPOINTS = 4, 4


@pytest.fixture(scope="module")
def sample():
    return import_in("prepare_data", "app.sample")


@pytest.fixture
def made(sample, tmp_path):
    out = tmp_path / "sample"
    sample.write(out)
    return out


def png_pixels(path):
    """(width, height, rows) of one of our PNG files, read without a library."""
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, chunks = 8, {}
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        kind, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + length]
        assert struct.unpack(">I", data[pos + 8 + length:pos + 12 + length])[0] == zlib.crc32(kind + body)
        chunks.setdefault(kind, b"")
        chunks[kind] += body
        pos += 12 + length
    width, height = struct.unpack(">II", chunks[b"IHDR"][:8])
    raw = zlib.decompress(chunks[b"IDAT"])
    stride = width * 3 + 1
    assert all(raw[y * stride] == 0 for y in range(height))
    return width, height, [raw[y * stride + 1:(y + 1) * stride] for y in range(height)]


def test_the_sample_has_the_yolo_layout_and_passes_the_layout_check(made):
    assert layouts.load("yolo").validate(made).errors == []
    assert sorted(p.name for p in made.iterdir()) == ["README.md", "data.yaml", "images", "labels"]


def test_the_sample_is_a_handful_of_small_images_under_one_megabyte(sample, made):
    files = [p for p in made.rglob("*") if p.is_file()]
    assert sum(p.stat().st_size for p in files) < 1_000_000
    for split, count in (("train", sample.TRAIN), ("val", sample.VAL)):
        assert len(list((made / "images" / split).glob("*.png"))) == count
        assert len(list((made / "labels" / split).glob("*.txt"))) == count
    assert sample.TRAIN + sample.VAL <= 50


def test_every_image_has_one_label_line_with_a_box_and_four_keypoints(made):
    for label in (made / "labels").rglob("*.txt"):
        (line,) = label.read_text().splitlines()
        values = line.split()
        assert len(values) == 1 + BOX + 3 * KEYPOINTS
        assert (made / "images" / label.parent.name / (label.stem + ".png")).is_file()


def test_the_box_and_the_corners_are_exactly_the_rectangle_in_the_picture(sample, made):
    for image in (made / "images").rglob("*.png"):
        width, height, rows = png_pixels(image)
        background = rows[0][:3]
        xs, ys = set(), set()
        for y, row in enumerate(rows):
            for x in range(width):
                if row[x * 3:x * 3 + 3] != background:
                    xs.add(x)
                    ys.add(y)
        x0, x1, y0, y1 = min(xs), max(xs) + 1, min(ys), max(ys) + 1   # the rectangle covers x0..x1-1, y0..y1-1
        assert len(xs) == x1 - x0 and len(ys) == y1 - y0, "one solid rectangle"
        v = [float(t) for t in (made / "labels" / image.parent.name / (image.stem + ".txt")).read_text().split()]
        assert v[0] == 0
        assert v[1:5] == pytest.approx([(x0 + x1) / 2 / width, (y0 + y1) / 2 / height, (x1 - x0) / width, (y1 - y0) / height], abs=1e-6)
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]   # top left, top right, bottom right, bottom left
        for i, (cx, cy) in enumerate(corners):
            kx, ky, visible = v[5 + 3 * i:8 + 3 * i]
            assert (kx, ky, visible) == pytest.approx((cx / width, cy / height, 2), abs=1e-6)


def test_the_same_files_come_out_every_time(sample, tmp_path):
    sample.write(tmp_path / "a")
    sample.write(tmp_path / "b")
    names = sorted(p.relative_to(tmp_path / "a").as_posix() for p in (tmp_path / "a").rglob("*") if p.is_file())
    assert names == sorted(p.relative_to(tmp_path / "b").as_posix() for p in (tmp_path / "b").rglob("*") if p.is_file())
    assert all((tmp_path / "a" / n).read_bytes() == (tmp_path / "b" / n).read_bytes() for n in names)


def test_the_images_differ_from_one_another(made):
    pictures = {p.read_bytes() for p in (made / "images").rglob("*.png")}
    assert len(pictures) == 20


# ------------------------------------------------------------------------------------- when the sample is used

def test_only_the_default_dataset_without_a_yolo_dataset_is_a_starter(sample, tmp_path, made):
    text = tmp_path / "text"
    (text / "sample").mkdir(parents=True)
    (text / "README.txt").write_text("a text sample")
    assert sample.is_starter("dataset-default", text)
    assert sample.is_starter("dataset-default", tmp_path / "nothing"), "an empty dataset too"
    assert not sample.is_starter("dataset-cats", text), "another dataset is read as it is"
    assert not sample.is_starter("dataset-default", made), "a YOLO dataset in the default dataset is kept"
    half = tmp_path / "half"
    (half / "images").mkdir(parents=True)
    assert not sample.is_starter("dataset-default", half), "a dataset that has begun to be a YOLO one is not replaced"


@pytest.fixture
def as_default(monkeypatch):
    """The input of the step is read from the repository `dataset-default`."""
    monkeypatch.setattr(datasets, "dataset_repo", lambda name: "dataset-default" if name == "dataset" else f"dataset-{name}")


def run_entry(tmp_path, folder, **data):
    entry = import_in("prepare_data", "app.entry")
    with testing.running(tmp_path / "run", kind="transform", params={"data": {"sample_size": "", "seed": 42, **data}},
                         inputs={"dataset": folder}, produces=[("dataset", "prepared")]) as run:
        entry.main()
    return run


def starter_folder(tmp_path):
    folder = tmp_path / "starter"
    (folder / "sample").mkdir(parents=True)
    (folder / "README.txt").write_text("a text sample")
    (folder / "sample" / "measurements.csv").write_text("a,b\n1,2\n")
    return folder


def test_a_first_run_on_the_default_dataset_prepares_the_sample(as_default, tmp_path):
    run = run_entry(tmp_path, starter_folder(tmp_path))
    prepared = run.output("prepared")
    assert layouts.load("yolo").validate(prepared).errors == []
    assert run.result["values"]["splits"] == {"train": {"kept": 16, "of": 16}, "val": {"kept": 4, "of": 4}}
    assert not (prepared / "README.txt").exists(), "the text sample of the project is not part of it"


def test_a_sample_size_applies_to_the_sample_too(as_default, tmp_path):
    run = run_entry(tmp_path, starter_folder(tmp_path), sample_size="3")
    assert run.result["values"]["splits"] == {"train": {"kept": 3, "of": 16}, "val": {"kept": 3, "of": 4}}


def test_a_yolo_dataset_in_the_default_dataset_is_used_as_it_is(as_default, tmp_path, dataset):
    run = run_entry(tmp_path, dataset)
    assert run.result["values"]["splits"]["train"] == {"kept": 6, "of": 6}


def test_another_dataset_that_is_not_yolo_is_refused_and_not_replaced_by_the_sample(tmp_path):
    with pytest.raises(SystemExit, match="data.yaml"):
        run_entry(tmp_path, starter_folder(tmp_path))   # its repository is dataset-dataset here, not the default
