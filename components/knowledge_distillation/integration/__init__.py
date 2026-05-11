"""Framework-specific glue for KD runners.

`deim_kd_engine.py` patches DEIM's train_one_epoch to inject KD terms;
`deim_kd_launch.py` is the torchrun entrypoint that activates the patch then
defers to DEIM's normal training driver.

Lives separately from `runners/` because these modules import vendored DEIM
internals at runtime (must run under the DEIM venv on the training server,
not the uv env).
"""
