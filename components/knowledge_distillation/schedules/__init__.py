"""Multi-phase / multi-teacher orchestration — landed per-cell when scheduled.

These are NOT loss functions. They sequence training phases, swap teachers,
toggle intrinsic distillation paths, or load alternate teacher checkpoints.

Planned modules (per v1.2 §3.2 / §2.3 / §5.4 / §5.5):
    kd_phase_runner.py  — Stage-0 (warm start, COCO → R2 hard-target) /
                          Stage-1 (KD-on, ramp via losses/kd_weight_ramp) /
                          Stage-2 (KD-off final tuning, optional last 5-10%
                          epochs) phase orchestrator. The scalar weight ramp
                          itself lives in losses/kd_weight_ramp.py; this
                          module owns the ENABLE/DISABLE phase transitions
                          and any per-phase data-loader / aug-policy swaps.
    mtpd_progressive.py — A5 teacher-sequence orchestrator: phase 1 same-family
                          M (stable features), phase 2 complementary-family M.
                          Synchronous 4-teacher explicitly excluded.
    takd_assistant.py   — A7 capacity-gap bridge: L → M assistant → S student
                          (Mirzadeh 2020). One of the §2.3 "two-of-three"
                          mandatory mitigations.
    eskd_loader.py      — A7 early-stopped teacher checkpoint resolver
                          (Cho 2019). Picks an early epoch's teacher rather
                          than the final one to mitigate capacity gap.
    golsd_toggle.py     — A0 control: disables D-FINE's intrinsic GO-LSD
                          self-distillation (DEIM trainer hook, see §5.5).
                          Required for the DEIM-path baseline to net-separate
                          external KD contribution from intrinsic GO-LSD.

Composition with losses/: a runner combines (a) a schedule from this package
to determine WHICH teacher / WHICH phase / WHICH GO-LSD setting is active,
with (b) a loss-module from losses/ for the per-batch KD signal.
"""
