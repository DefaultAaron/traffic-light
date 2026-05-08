"""TSM nn.Module pieces — generic across detector families.

Two modules:
    temporal_shift.py   The TemporalShift op itself: 1/8-channel forward shift
                        with zero-pad boundary; 7/8 channels untouched. Stateless
                        in training (clip is one tensor of shape (B, T, C, H, W)
                        and the shift is along T); stateful in streaming
                        inference (consumes a per-stage previous-frame tensor
                        from FeatureCache and emits the shifted feature).
    feature_cache.py    Per-stage previous-frame feature container for streaming
                        inference. One instance per camera; holds 1/8-channel
                        slice per neck stage (P3 / P4 / P5; cap at P3+P4 if
                        memory pressure). Reset on stream gap > N frames or
                        explicit reset() call.

------------------------------------------------------------------------------
Implementation contract (when modules land — Phase 1-A scheduling)
------------------------------------------------------------------------------
- Both modules are torch.nn.Module subclasses; no custom autograd ops needed.
- Shift op MUST be expressible as torch.split + torch.zeros_like + torch.cat
  so ONNX export reduces to (Slice + Concat). Custom op is a code smell — the
  plan §1.5 Phase 1-C explicitly says "ONNX export 处理 shift 算子: Slice +
  Concat 即可表达 (无需自定义算子)".
- FeatureCache MUST be a plain Python container — NOT registered as a child
  Module of TemporalShift; the cache is per-camera runtime state, not a
  learnable parameter.
- Channel split: c1 = C[:C//8] (forward-shifted), c2 = C[C//8:C//4] (reserved
  for bidirectional training only — zero-pad in inference), c3 = C[C//4:]
  (untouched). The 6/8 untouched residual is the §1.1 spec.
- Boundary handling: zero-pad on the temporal boundary (first frame of clip
  for forward shift). Streaming inference's "first frame after reset" is the
  same boundary — the cache returns zeros until the first frame populates it.

------------------------------------------------------------------------------
Memory budget (§1.5 Phase 1-B / §1.6 risk row 2)
------------------------------------------------------------------------------
Per-camera cache footprint:
    1/8 channel × per-stage HxW × dtype=fp16 (engine precision)

    YOLO26-s @ 640 input typical:
        P3=80x80x32, P4=40x40x64, P5=20x20x128
        1/8 slice: 80x80x4 + 40x40x8 + 20x20x16
                 = 25.6k + 12.8k + 6.4k
                 = 44.8k fp16 elements ≈ 90 KB per camera
    DEIM-D-FINE-S (HGNetv2-B2) @ 640 input typical:
        HG_Stage 1 / 2 / 3 outputs at 80x80 / 40x40 / 20x20 with HGNetv2-B2
        out_chs roughly 256 / 512 / 1024 (size variant dependent; verify at
        Phase 1-A).
        1/8 slice: 80x80x32 + 40x40x64 + 20x20x128
                 = 204.8k + 102.4k + 51.2k
                 = 358.4k fp16 elements ≈ 720 KB per camera
        Still well under §1.6 budget of 5 MB / camera.

Headroom: §1.6 budget is 5 MB / camera. DEIM-S backbones with deeper stages
(higher channel counts than YOLO26-s) sit at ~720 KB; YOLO26-s sits at ~90
KB. Both cached in full stay comfortably under budget. If memory pressure
surfaces post-Phase-1-C, drop P5 / HG_Stage 3 first (deepest stage; least
helpful for small-target recall) — see §1.6 mitigation row 2.
"""
