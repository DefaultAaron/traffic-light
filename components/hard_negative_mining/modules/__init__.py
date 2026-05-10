"""Hard-negative mining + human-verification modules (a-stage scaffold).

Two distinct concerns live here:

  ``miner``    — Run R1 baseline on demo videos / R2 self-collected
                 hard scenes; collect FP frames where the baseline
                 produced a non-real-light detection at confidence ≥
                 0.25. Output: a CANDIDATE manifest of (image, source,
                 baseline detection metadata) rows. NOT the frozen
                 eval manifest — that is a separate downstream artifact
                 in ``data/eval_manifest.py``.
  ``verifier`` — Plan §4.5 risk-mitigation requires human verification
                 of mined candidates BEFORE they enter the bg/ training
                 directory. The plan prose: "挖帧后必须人工核验，建议
                 每 200 帧抽检 ≥ 20" (10% sample). This subpackage
                 owns the typed sample-protocol contract.

Sister-file precedent: ``components/copy_paste_balance/modules/``.
Both modules are a-stage scaffolds; b-stage fills bodies.
"""
