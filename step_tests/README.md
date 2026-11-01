# step_tests/ - tests of the step code (they use the kubecore package)

The steps import the kubecore PACKAGE (datasets, run outputs). The repo root also holds the
vendored engine, also named `kubecore`, which `tests/` imports. The two must never share a
`sys.path`, so they run as two commands:

```bash
python -m pytest tests                                         # engine (vendored kubecore/ on the path)
pip install "https://github.com/novelcore/hera-pipeline-template/archive/<commit>.zip#subdirectory=packages/kubecore" \
    pytest boto3 psutil pydantic pydantic-settings
pytest step_tests                                              # step code (the package on the path)
```

Use the `pytest` command, not `python -m pytest`, here: `python -m` puts the repo root on
`sys.path` and the vendored engine would shadow the package. `conftest.py` refuses to run if it does.
`<commit>` is the one in the step Dockerfiles (`KUBECORE_PKG_COMMIT`).

`fake_lakefs.py` is the package's in-memory lakeFS, copied as is.
