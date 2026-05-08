"""Pipeline components beyond the core detector workstream.

Each subpackage owns one optional add-on (KD, future temporal/depth tracks, ...).
Code here is independent of the main training entrypoints in ``main.py`` — runners
import from here only when the corresponding R2/R3 cell is scheduled.
"""
