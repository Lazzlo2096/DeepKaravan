# Concurrent API benchmark

Run explicitly against the running single-container service:

```powershell
uv run --locked --extra api python super_test/api_capacity.py --concurrency 4 --repeats 2
```

The benchmark submits async requests and polls results. It covers all seven
nonempty dense/sparse/colbert combinations, checks order, vector dimensions,
finite values and dense normalization. No additional model process is started.
Use `--batch-size`, `--timeout`, `--base-url`, and `--output` to configure it.
The default report is `test-results/api-capacity.json`. Queue rejection is recorded
as an HTTP error and causes exit code 1. Network and inference failures stop the run.

This replaces the former direct broker benchmark; it exercises API validation,
IPC queues and the model process together. Old benchmark output is not migrated.
