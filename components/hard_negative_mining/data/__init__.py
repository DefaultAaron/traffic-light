"""Frozen evaluation manifest for the §四 hard-negative-mining ablation.

Sister precedent: ``components/copy_paste_balance/data/`` (locked iter-11
2026-05-09). This subpackage owns the FROZEN-MANIFEST contract that
plan §4.7 calls out as the anti-gaming safeguard:

  * The evaluation manifest (``runs/_hard_negative_eval_manifest.json``)
    contains image SHA256 + label source + evaluation thresholds (conf
    ≥ 0.25, NMS IoU = 0.5). It is shared with §3.7 (copy-paste-balance
    rare-related FP denominator).
  * Once frozen, the manifest MUST NOT be modified — neither the image
    set nor the thresholds — for the duration of the ablation. Plan
    §4.7 prose is verbatim: "评估时不得改 manifest 或阈值——保护 FP
    下降率 denominator 不被换 frame 子集 game".

The dataclass here is a-stage scaffold: schema lives, parser stub.
"""
