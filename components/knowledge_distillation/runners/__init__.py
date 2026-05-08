"""Per-cell training entrypoints (one module per A0–A7 cell).

Each runner is a stub until its cell is scheduled. Stubs raise NotImplementedError
to keep accidental invocation loud. Wrapper shell scripts (``scripts/train_kd_*.sh``)
will be added when the corresponding cell lands.
"""
