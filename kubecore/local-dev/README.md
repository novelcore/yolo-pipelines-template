# kubecore/local-dev - stand-ins for the local preview only

These files let `./run.sh` render and compose on your computer. **You do not own or edit them**: the platform
provides the real values when your pipeline is built.

- `pipeline-context.yaml`: the platform's description of your project (machine classes, endpoints) that the
  preview needs.

The values here are non-functional placeholders (`PLACEHOLDER-*`, `*.internal.invalid`). A local render shows the
right shape; it never uses real endpoints or credentials, and none belong in this repository.
