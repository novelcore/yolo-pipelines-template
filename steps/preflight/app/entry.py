"""preflight — a minimal CPU step: read the resolved params, print a summary,
prove a developer can add a step with nothing but reads= + a few lines."""
import argparse, yaml

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    args, _ = ap.parse_known_args()
    cfg = yaml.safe_load(args.params)
    exp = cfg.get("experiment", {})
    print(f"preflight: experiment={exp.get('name','?')} epochs={cfg.get('train',{}).get('epochs','?')}")
    print("preflight: OK — config resolved, ready for downstream steps")

if __name__ == "__main__":
    main()
