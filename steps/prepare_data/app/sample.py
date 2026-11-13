"""A small labelled sample, made here, for a first run of the pipeline on a new project.

A new project has no labelled images. Its default dataset holds only a text sample that the platform put there, so
a first run with every setting as it is would stop at "the dataset has no data.yaml". `prepare-data` then writes this
sample as the dataset `prepared` instead (see `is_starter`), and the rest of the pipeline trains on it.

The images are not a real dataset and are not copied from one: each is a flat-coloured picture with one lighter
rectangle, drawn here from a fixed seed, so the same files come out on every machine. The label of an image is exact:
the box is the rectangle and the four keypoints are its corners (top left, top right, bottom right, bottom left).
That is the layout the pipeline reads (docs/datasets.md): `data.yaml`, `images/<split>/`, `labels/<split>/`.

Write the same files into a folder to look at them:  python steps/prepare_data/app/sample.py ./sample
"""
import random
import struct
import sys
import zlib
from pathlib import Path

# The project's default dataset: the one the run form starts with (config/dataset.yaml).
STARTER_REPO = "dataset-default"

SIZE = 160           # pixels, width and height
TRAIN, VAL = 16, 4   # images per split
SEED = 2026

DATA_YAML = """path: .
train: images/train
val: images/val
nc: 1
names:
  0: box
kpt_shape: [4, 3]          # 4 keypoints, each as x y visibility
flip_idx: [1, 0, 3, 2]     # top left <-> top right, bottom left <-> bottom right
"""

README = """# Sample dataset

A tiny labelled sample made by the pipeline, so that a first run works before you have data of your own.
Each image is a flat colour with one lighter rectangle. The label of an image is exact: the box is the
rectangle, and the four keypoints are its corners (top left, top right, bottom right, bottom left).
It is not a real dataset: a model trained on it only shows that the pipeline works from end to end.

To use your own images, make a dataset in this layout and choose it as the `dataset` of a run (docs/datasets.md).
"""


def _png(width: int, height: int, rows: list) -> bytes:
    """A PNG file of `rows` (one bytes of RGB per pixel row), with no library."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    raw = b"".join(b"\x00" + row for row in rows)   # filter type 0 on every row
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def _image(rng: random.Random) -> tuple:
    """`(png bytes, label line)` of one picture: a rectangle on a background, and its exact label."""
    background = tuple(rng.randint(20, 100) for _ in range(3))
    colour = tuple(rng.randint(170, 255) for _ in range(3))
    w, h = rng.randint(48, 104), rng.randint(48, 104)
    x0, y0 = rng.randint(8, SIZE - 8 - w), rng.randint(8, SIZE - 8 - h)
    x1, y1 = x0 + w, y0 + h
    outside, inside = bytes(background) * SIZE, bytes(background) * x0 + bytes(colour) * w + bytes(background) * (SIZE - x1)
    rows = [inside if y0 <= y < y1 else outside for y in range(SIZE)]
    box = [(x0 + x1) / 2 / SIZE, (y0 + y1) / 2 / SIZE, w / SIZE, h / SIZE]
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    values = [f"{v:.6f}" for v in box] + [f"{x / SIZE:.6f} {y / SIZE:.6f} 2" for x, y in corners]
    return _png(SIZE, SIZE, rows), "0 " + " ".join(values) + "\n"


def write(out: Path) -> dict:
    """Write the sample into `out` and return the number of images of each split."""
    rng = random.Random(SEED)
    counts = {"train": TRAIN, "val": VAL}
    for split, count in counts.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(count):
            png, label = _image(rng)
            (out / "images" / split / f"{split}{i:02d}.png").write_bytes(png)
            (out / "labels" / split / f"{split}{i:02d}.txt").write_text(label)
    (out / "data.yaml").write_text(DATA_YAML)
    (out / "README.md").write_text(README)
    return counts


def is_starter(repo: str, folder: Path) -> bool:
    """True for the project's default dataset when nothing in it is a YOLO dataset: no `data.yaml`, no `images/`,
    no `labels/`. Any other dataset is read as it is, so a dataset of your own is never replaced."""
    return repo == STARTER_REPO and not any((folder / name).exists() for name in ("data.yaml", "images", "labels"))


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) == 2 else None
    if target is None:
        raise SystemExit("usage: python sample.py <folder to write the sample into>")
    print(write(target))
