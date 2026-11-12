"""Pick a reproducible subset of a YOLO dataset: ``images/<split>/**`` with its labels, per split."""
import hashlib
import os
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _rank(seed: int, relative: str) -> str:
    """The order images are kept in: by a hash of the seed and the file's path. The same seed picks the same
    files on every machine, and adding or removing an image changes the pick by at most that one image."""
    return hashlib.sha256(f"{seed}:{relative}".encode()).hexdigest()


def _link(source: Path, target: Path) -> None:
    """Put ``source`` at ``target`` without copying its bytes when the folders share a disk."""
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def write(dataset: Path, out: Path, size: int | None, seed: int) -> dict:
    """Write ``dataset`` into ``out``, keeping at most ``size`` images per split (every image when ``size`` is None).

    A kept image takes its label (``labels/<same path>.txt``) with it; every file outside ``images/`` and
    ``labels/`` (``data.yaml``, a README) is kept as it is. Paths do not change, so a run that keeps the same
    files writes the same dataset. Returns the number of images kept and found, per split.
    """
    images_dir = dataset / "images"
    splits = sorted(p for p in images_dir.iterdir() if p.is_dir()) if images_dir.is_dir() else []
    kept, found = set(), {}
    for split in splits:
        images = sorted(f.relative_to(dataset).as_posix() for f in split.rglob("*") if f.suffix.lower() in IMAGE_SUFFIXES)
        chosen = images if size is None else sorted(images, key=lambda r: _rank(seed, r))[:size]
        found[split.name] = {"kept": len(chosen), "of": len(images)}
        kept.update(chosen)
    stems = {Path(r).with_suffix("").as_posix() for r in kept}  # images/train/a
    for file in sorted(f for f in dataset.rglob("*") if f.is_file()):
        relative = file.relative_to(dataset)
        if relative.parts[0] == "images" and file.suffix.lower() in IMAGE_SUFFIXES and relative.as_posix() not in kept:
            continue
        if size is not None and relative.parts[0] == "labels" and \
                Path("images", *relative.parts[1:]).with_suffix("").as_posix() not in stems:
            continue
        _link(file, out / relative)
    return found
