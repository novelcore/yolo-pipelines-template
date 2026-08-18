import argparse
import yaml

# The config sections this step reads.
READS = ["experiment"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True)
    args, _ = parser.parse_known_args()
    cfg = yaml.safe_load(args.params)

    # Print a config parameter — here, the experiment name.
    print(f"👋 Hello world! experiment.name = {cfg['experiment']['name']}")


if __name__ == "__main__":
    main()

