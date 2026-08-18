"""hydra-demo: prove the composed Hydra config reaches a step."""
import argparse
import yaml

READS = ["experiment", "train", "evaluation"]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True)
    args, _ = parser.parse_known_args()
    cfg = yaml.safe_load(args.params)

    print("=== hydra-demo received ===")
    print(f"  experiment.name       = {cfg['experiment']['name']}")
    print(f"  train.epochs          = {cfg['train']['epochs']}")
    print(f"  evaluation.split      = {cfg['evaluation']['split']}")
    print(f"  evaluation.iou_thresh = {cfg['evaluation']['iou_threshold']}")

if __name__ == "__main__":
    main()
