"""prepare-data: the run's dataset, or a reproducible subset of it, as the dataset `prepared` (PRD kaos#1911 X-5)."""
from kaos_lib import step

from app import sample, subset


def main() -> None:
    ctx = step.start()
    data = ctx.params["data"]
    size = int(data["sample_size"]) if str(data.get("sample_size") or "").strip() else None  # empty: every image
    if size is not None and size < 1:
        raise SystemExit(f"data.sample_size must be a positive number or empty, got {size}")
    raw, prepared = step.input("dataset"), step.output("prepared")
    if sample.is_starter(ctx.ledger.named("dataset").repo, raw):
        # A new project: its default dataset holds no images. Train on the small sample made here instead.
        print(f"[prepare-data] {sample.STARTER_REPO} holds no YOLO dataset yet, so this run uses the small sample made "
              "by the pipeline (its README.md is in the prepared dataset). Choose a dataset of your own on the run form "
              "to train on your images.", flush=True)
        raw = step.workdir() / "sample"
        sample.write(raw)
    elif not (raw / "data.yaml").is_file():
        raise SystemExit("the dataset has no data.yaml at its root: it is not a YOLO dataset (data.yaml, images/, labels/)")
    found = subset.write(raw, prepared, size, int(data["seed"]))
    step.check("prepared", min_files=2)
    print(f"[prepare-data] kept {found}", flush=True)
    step.result(sample_size=size, seed=int(data["seed"]), splits=found)


if __name__ == "__main__":
    main()
