"""Backbone-injection patches per detector family — single file lands at Phase 1-A.

Per plan §1.4: TSM is **deeply coupled** to the main-track-selected detector,
and Phase 1-A starts AFTER R2 detector selection is final. So this subpackage
holds at most ONE patch file at any time — for the selected detector. Holding
speculative patches for non-selected detectors invites drift.

Naming convention (when the patch lands):
    yolo26_basicblock.py     YOLO26 C3 BasicBlock.forward injection
    yolov13_basicblock.py    YOLOv13 C2f BasicBlock.forward injection
    deim_hg_block.py         DEIM HGNetv2 backbone HG_Block injection.
                             Target class: `HG_Block` in
                             `DEIM/engine/backbone/hgnetv2.py:199` (NOT under
                             `extre_module/`). Injection point: start of
                             `HG_Block.forward` (line 275), on `x` before the
                             `for layer in self.layers:` loop. Channel split
                             1/8 + 1/8 + 6/8 on `in_chs`.

                             Patch scope — UNAMBIGUOUS, three equivalent forms:
                                 config name:    stage2, stage3, stage4
                                 code indexing:  HGNetv2.stages[1], [2], [3]
                                 return mapping: return_idx = [1, 2, 3]
                                                 (HGNetv2.__init__ default;
                                                 matches DEIM-S traffic_light
                                                 config line 32). These three
                                                 are the stages whose features
                                                 are returned to HybridEncoder.

                                 patched:    HG_Block instances inside
                                             stage2 / stage3 / stage4
                                             (HGNetv2.stages[1..3])
                                 NOT patched: StemBlock (line 125; single-conv
                                              entry, no temporal surface)
                                 NOT patched: stage1 (HGNetv2.stages[0]); its
                                              output is not in return_idx and
                                              receptive field is too small
                                              for short-range temporal context

                             Strict OUT-OF-SCOPE on DEIM (do NOT add patch
                             files for these even if requested mid-cycle):
                                 - HybridEncoder (`DEIM/engine/deim/
                                   hybrid_encoder.py`) — transformer + AIFI
                                   self-attention; TSM is conv-only.
                                 - D-FINE decoder transformer layers (pure
                                   attention).

Patch contract (per file):
    - The patch MUST be import-time idempotent — applying twice raises a
      RuntimeError, not a silent re-monkey-patch.
    - The patch MUST preserve the un-shifted forward signature so the same
      checkpoint format works in TSM-on / TSM-off ablation runs.
    - The patch MUST register a top-level `apply()` function and a top-level
      `revert()` function for the runner to scope the monkey-patch around the
      training call.
    - The patch file MUST declare its target detector version (e.g. via a
      module-level `TARGET_DETECTOR = "yolo26"` constant) so the runner can
      validate the import matches the selected detector before patching.

Speculative no-go (§1.6 risk row 4):
    "主检测器选型仍未定，提前启动 TSM" → 返工. Do NOT pre-write patches for
    multiple detectors hoping to commit later. Wait for selection.
"""
