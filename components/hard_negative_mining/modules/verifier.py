"""Human verification protocol for mined hard-negative candidates.

Plan §4.5 risk + mitigation (verbatim):

    "误把'真实但难分的灯'标记为硬负 → 召回掉. 挖帧后必须人工核验,
    建议每 200 帧抽检 ≥ 20."

Concretely: for every mining batch of N candidates, at least 10% must
be human-verified before any candidate enters the training bg/
directory. The verifier records per-candidate verdicts in a typed
report; un-sampled candidates are passed through with the assumption
they share the sample's verified-per-batch-error rate. If the sample
verification rate falls below the configured floor, the entire batch
is REJECTED — the mining pass must be re-run or the threshold
adjusted.

This module is a-stage scaffold: typed contract + run() stub.

Sister-file precedent (internal B2 audit trail; b-stage authors can
ignore): ``components/copy_paste_balance/modules/`` — a-stage
scaffold with __post_init__ contract and NotImplementedError body.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import ClassVar

from components.hard_negative_mining._internals import (
    PLAN_LOCKED_MIN_SAMPLE_FRACTION,
    PLAN_TPM_RATE_CEILING,
)


class VerificationVerdict(str, Enum):
    """Per-candidate human-review outcome.

    Three terminal states; matches plan §4.5 risk-mitigation taxonomy:
      * ``true_negative`` — confirmed FP; safe to use as bg/ training image.
      * ``true_positive_missed`` — actually a real light the baseline
        missed (would-be label-noise if used as bg/); MUST be excluded
        AND flagged for label correction in the upstream R2 dataset.
      * ``ambiguous`` — reviewer cannot decide (occlusion, motion blur,
        edge of frame); excluded from bg/ to be safe.

    Re-exported from the package ``__init__`` because b-stage's
    verification report deserializer needs to map the report JSON's
    ``verdict`` field to this enum without reaching into the deep path.
    The dataclasses in this module do NOT carry a verdict-typed field
    directly — verdicts live in the per-row report payload b-stage
    builds inside ``run_verification``.
    """

    TRUE_NEGATIVE = "true_negative"
    TRUE_POSITIVE_MISSED = "true_positive_missed"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class VerificationSampleProtocol:
    """Plan §4.5 sampling protocol — frozen contract.

    Plan prose: "建议每 200 帧抽检 ≥ 20" — 10% sampling floor. The
    protocol is intentionally a fraction (not an absolute count) so
    smaller batches still satisfy the safety floor.

    The protocol is enforced at the batch boundary: if the verifier
    cannot achieve at least ``min_sample_fraction`` of candidates
    reviewed (for any reason — operator availability, schedule slip,
    etc.), the batch is REJECTED rather than partially passed. Plan
    §4.5 prose names this "must verify" — partial verification with
    silent pass-through is precisely the risk being mitigated.
    """

    min_sample_fraction: float            # plan §4.5: 0.10 (10%)
    max_true_positive_missed_rate: float  # within-sample ceiling

    _LOCKED_MIN_SAMPLE_FRACTION: ClassVar[float] = PLAN_LOCKED_MIN_SAMPLE_FRACTION
    _MAX_TPM_RATE_CEILING: ClassVar[float] = PLAN_TPM_RATE_CEILING

    def __post_init__(self) -> None:
        # Bool-exclusion + finite + lock equality on the sample fraction
        # (plan §4.5 hard-pin; deviation requires re-running the §四 loop).
        if not isinstance(self.min_sample_fraction, float) or isinstance(
            self.min_sample_fraction, bool
        ):
            raise ValueError(
                f"min_sample_fraction must be float; got "
                f"{type(self.min_sample_fraction).__name__}="
                f"{self.min_sample_fraction!r}"
            )
        if not math.isfinite(self.min_sample_fraction):
            raise ValueError(
                f"min_sample_fraction must be finite; got "
                f"{self.min_sample_fraction!r}"
            )
        if self.min_sample_fraction != self._LOCKED_MIN_SAMPLE_FRACTION:
            raise ValueError(
                f"min_sample_fraction must equal "
                f"{self._LOCKED_MIN_SAMPLE_FRACTION} (plan §4.5 hard-pin: "
                f"'每 200 帧抽检 ≥ 20' = 10%); got "
                f"{self.min_sample_fraction}"
            )
        # max_true_positive_missed_rate: ceiling configurable but bounded.
        # Plan §4.5 doesn't cap the "acceptable" TP-missed rate explicitly;
        # we cap at PLAN_TPM_RATE_CEILING (0.20) defensively — beyond that
        # the mining pass is producing too much noise and should be
        # re-tuned, not waved through.
        if not isinstance(self.max_true_positive_missed_rate, float) or isinstance(
            self.max_true_positive_missed_rate, bool
        ):
            raise ValueError(
                f"max_true_positive_missed_rate must be float; got "
                f"{type(self.max_true_positive_missed_rate).__name__}="
                f"{self.max_true_positive_missed_rate!r}"
            )
        if not math.isfinite(self.max_true_positive_missed_rate):
            raise ValueError(
                f"max_true_positive_missed_rate must be finite; got "
                f"{self.max_true_positive_missed_rate!r}"
            )
        if not (0.0 <= self.max_true_positive_missed_rate <= self._MAX_TPM_RATE_CEILING):
            raise ValueError(
                f"max_true_positive_missed_rate must be in "
                f"[0, {self._MAX_TPM_RATE_CEILING}]; got "
                f"{self.max_true_positive_missed_rate} (a higher ceiling "
                f"signals a noisy mining pass; re-tune rather than widen)"
            )


@dataclass(frozen=True)
class VerificationConfig:
    """Configuration for one verification batch.

    Reads ``candidates_json`` (output of
    ``modules/miner.mine_candidates``); writes a typed verification
    report to ``output_report_json``. The report is INPUT to the bg/
    directory builder — only candidates with ``TRUE_NEGATIVE`` verdicts
    (within a batch that passed the sample-floor + TPM-rate gates) are
    eligible for inclusion.
    """

    candidates_json: Path
    output_report_json: Path
    protocol: VerificationSampleProtocol

    def __post_init__(self) -> None:
        if not isinstance(self.candidates_json, Path):
            raise ValueError(
                f"candidates_json must be Path; got "
                f"{type(self.candidates_json).__name__}="
                f"{self.candidates_json!r}"
            )
        if not isinstance(self.output_report_json, Path):
            raise ValueError(
                f"output_report_json must be Path; got "
                f"{type(self.output_report_json).__name__}="
                f"{self.output_report_json!r}"
            )
        # B2 review I4 2026-05-10: reject path collision between input
        # and output. Equal paths would clobber the candidates JSON
        # mid-read on filesystems that allow concurrent open-for-write.
        if self.candidates_json == self.output_report_json:
            raise ValueError(
                f"candidates_json and output_report_json must differ; "
                f"got {self.candidates_json} for both (writing the report "
                f"over the input would clobber the candidates manifest)"
            )
        if not isinstance(self.protocol, VerificationSampleProtocol):
            raise ValueError(
                f"protocol must be VerificationSampleProtocol; got "
                f"{type(self.protocol).__name__}"
            )


def run_verification(config: VerificationConfig) -> Path:
    """Drive the human-verification protocol for one mining batch.

    Process (b-stage):
      1. Load candidates JSON.
      2. Stratified-random sample ≥ ``protocol.min_sample_fraction`` rows;
         present each to the human reviewer.
      3. Record per-candidate ``VerificationVerdict``.
      4. Compute within-sample stats: TP-missed rate, ambiguous rate.
      5. If TP-missed rate exceeds ``protocol.max_true_positive_missed_rate``,
         mark the entire batch as REJECTED in the report.
      6. Atomic-rename ``output_report_json.tmp`` → ``output_report_json``.

    Output schema (b-stage):

        {
          "schema_version": "1",
          "candidates_json_sha256": str,
          "protocol": {
            "min_sample_fraction": 0.10,
            "max_true_positive_missed_rate": float
          },
          "sample_size": int,
          "candidate_count": int,
          "sample_verdicts": {
            "true_negative": int,
            "true_positive_missed": int,
            "ambiguous": int
          },
          "true_positive_missed_rate": float,    # within sample
          "batch_status": "accepted" | "rejected",
          "rejection_reason": str | null,
          "verified_at": str                     # ISO-8601
        }

    Args:
        config: ``VerificationConfig`` resolved from CLI / YAML.

    Returns:
        Path to the written verification report
        (== ``config.output_report_json``).

    Raises:
        FileNotFoundError: ``config.candidates_json`` missing.
        ValueError: malformed candidates JSON.
        NotImplementedError: a-stage scaffold.
    """
    raise NotImplementedError("b-stage")
