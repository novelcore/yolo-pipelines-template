"""data-summary step: prints a one-line summary of the dataset config."""

import argparse
import yaml

READS = ["data", "summary"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True)
    args, _ = parser.parse_known_args()
    cfg = yaml.safe_load(args.params)

    data = cfg["data"]
    summary = cfg["summary"]
    prefix = summary["label"]
    print(f"{prefix}: dataset ref='{data['ref']}' source='{data['source']}' seed={data['seed']}")


if __name__ == "__main__":
    main()
