"""YAML → typed-dataclass loader for the HMM ablation.

C3 NEW-MAJOR 5 (2026-05-09): the YAML adapter contract documented on
``TransitionConfig`` (tuple coercion + enum coercion) was floating
without a load-bearing implementation surface. This module exposes the
loader so the YAML→dataclass conversion is part of the locked a-stage
public API, not a hidden contract that b-stage invents inside the
runner.

The loader returns a single ``HmmYamlConfig`` bundling every YAML knob
(transition + observation + inference + runner). Per-subsystem
dataclasses (``TransitionConfig``, ``ObservationModel`` constructor
args, etc.) are built FROM ``HmmYamlConfig`` — never directly from raw
YAML — so the coercion contract has exactly one load-bearing entry
point.

Scaffold (a-stage): function signature only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from components.hmm_smoother.data.transition_counts import GapPolicy
from components.hmm_smoother.modules.inference import HmmInferenceMode
from components.hmm_smoother.modules.observation import ObservationMode
from components.hmm_smoother.modules.transition import (
    IllegalTransitionPolicy,
    TransitionConfig,
)


@dataclass(frozen=True)
class HmmYamlConfig:
    """Typed representation of ``configs/temporal_hmm.yaml``.

    Every field corresponds 1:1 to a YAML key (no renames, no
    derivations). String enums and list-of-list pairs are coerced into
    their typed counterparts here so downstream consumers never see raw
    YAML scalars.

    Cross-knob invariants enforced in ``__post_init__`` (pure value
    checks only — no I/O, no filesystem):
      * ``num_classes`` is set (the YAML default ``null`` is forbidden
        at this layer; the runner is responsible for refusing to load
        a YAML with ``num_classes: null``).
      * ``laplace_alpha`` is one of {0.01, 0.1, 1.0} — plan §2.2
        sensitivity-sweep set; YAML deviation is a contract violation
        (C3 iter-1 NEW-MINOR 6 2026-05-09: hard-fail rather than
        silent widening of the sweep).
      * ``observation_epsilon`` is in (0, 1 / num_classes) per the
        ``ObservationModel`` constructor invariant.
      * Every (src, dst) in ``illegal_transition_set`` is in
        ``[0, num_classes)``.

    Filesystem-state checks (C3 iter-2 NEW-MINOR 5 2026-05-09):
    ``transition_matrix_path`` existence / readability is NOT checked
    here. A frozen-dataclass ``__post_init__`` doing I/O makes
    construction non-deterministic (config valid at one tick, invalid
    at the next, depending on whether a parallel job is rewriting the
    artifact). Path existence is the runner's responsibility — it
    runs in ``run_ablation()`` (or ``TransitionMatrix.from_npy()``)
    where missing-file errors can produce diagnostic
    ``decision: "executor_error"`` rows with the path in ``notes``,
    rather than aborting config construction with a stack trace.
    """

    num_classes: int
    transition_matrix_path: Path | None
    laplace_alpha: float
    illegal_transition_set: tuple[tuple[int, int], ...]
    illegal_transition_policy: IllegalTransitionPolicy
    viterbi_window: int | None
    inference_mode: HmmInferenceMode
    min_track_length: int
    map_regression_tolerance_pp: float
    gap_policy: GapPolicy
    observation_mode: ObservationMode
    observation_epsilon: float

    _ALLOWED_LAPLACE_ALPHAS: ClassVar[tuple[float, ...]] = (0.01, 0.1, 1.0)

    def __post_init__(self) -> None:
        # Pure value checks only — NO filesystem I/O (C3 iter-2 NEW-MINOR 5).
        # Codex stop-gate 2026-05-09: this used to be docstring-only; now
        # implemented so direct construction can't bypass the contract.
        # num_classes contract: explicit int > 0. The YAML ships `num_classes: null`
        # as a forcing function (loader must override before construction); the three
        # checks below cover (1) the None bypass, (2) wrong-type bypass (str, float,
        # bool — note bool is an int subclass in Python so it needs an explicit
        # exclusion), and (3) the range check. Codex stop-gate 2026-05-09: the prior
        # `<= 0` check raised TypeError (not ValueError) when num_classes was None,
        # so the documented "null is rejected" contract was not actually enforced.
        # C3 stop-hook iter-2 NEW-MINOR 2026-05-09: transition_matrix_path
        # type validation. Filesystem state (existence / readability) is
        # the runner's responsibility — but Path-vs-str type validation is
        # pure boundary work and belongs here. Without this, a loader that
        # forgets `Path(raw)` coercion ships a raw YAML str downstream and
        # the failure surfaces later as a confusing AttributeError on a
        # path method, not a structured ValueError at config-load.
        if self.transition_matrix_path is not None and not isinstance(self.transition_matrix_path, Path):
            raise ValueError(
                f"transition_matrix_path must be Path or None; got "
                f"{type(self.transition_matrix_path).__name__}={self.transition_matrix_path!r} "
                f"(loader must coerce YAML string → pathlib.Path before construction)"
            )
        if self.num_classes is None:
            raise ValueError(
                "num_classes must be set explicitly; got None "
                "(YAML default `null` is rejected — the loader / runner MUST provide "
                "a concrete int matching the detector's class count before construction)"
            )
        if not isinstance(self.num_classes, int) or isinstance(self.num_classes, bool):
            raise ValueError(
                f"num_classes must be int; got "
                f"{type(self.num_classes).__name__}={self.num_classes!r}"
            )
        if self.num_classes <= 0:
            raise ValueError(
                f"num_classes must be > 0; got {self.num_classes}"
            )
        # B2 stop-hook delta I2 2026-05-09: float-with-bool-exclusion before
        # the membership test, because `True in (0.01, 0.1, 1.0)` returns True
        # (Python coerces bool→int→float in equality), so `laplace_alpha=True`
        # would silently alias to 1.0 and corrupt the sensitivity-sweep
        # artifact. YAML serializes 0.1 / 1.0 as float; integer 1 is rejected
        # by design (loader / runner must use the float form to match the
        # plan §2.2 sweep set).
        if not isinstance(self.laplace_alpha, float) or isinstance(self.laplace_alpha, bool):
            raise ValueError(
                f"laplace_alpha must be float; got "
                f"{type(self.laplace_alpha).__name__}={self.laplace_alpha!r}"
            )
        if self.laplace_alpha not in self._ALLOWED_LAPLACE_ALPHAS:
            raise ValueError(
                f"laplace_alpha must be one of "
                f"{self._ALLOWED_LAPLACE_ALPHAS}; got {self.laplace_alpha}. "
                f"Plan §2.2 mandates this fixed sweep set; deviation is a "
                f"contract violation (C3 iter-1 NEW-MINOR 6)."
            )
        if not isinstance(self.illegal_transition_policy, IllegalTransitionPolicy):
            raise ValueError(
                f"illegal_transition_policy must be IllegalTransitionPolicy enum; "
                f"got {type(self.illegal_transition_policy).__name__} "
                f"(loader must coerce YAML string before construction)"
            )
        if not isinstance(self.inference_mode, HmmInferenceMode):
            raise ValueError(
                f"inference_mode must be HmmInferenceMode enum; "
                f"got {type(self.inference_mode).__name__}"
            )
        if not isinstance(self.observation_mode, ObservationMode):
            raise ValueError(
                f"observation_mode must be ObservationMode enum; "
                f"got {type(self.observation_mode).__name__}"
            )
        if not isinstance(self.gap_policy, GapPolicy):
            raise ValueError(
                f"gap_policy must be GapPolicy enum; "
                f"got {type(self.gap_policy).__name__}"
            )
        # C3 stop-hook delta NEW-MINOR (observation_epsilon) 2026-05-09:
        # type-with-bool-exclusion before the range check — same hazard
        # class as B2 I2/I3, applied for symmetry across numeric fields.
        if not isinstance(self.observation_epsilon, float) or isinstance(self.observation_epsilon, bool):
            raise ValueError(
                f"observation_epsilon must be float; got "
                f"{type(self.observation_epsilon).__name__}={self.observation_epsilon!r}"
            )
        if not (0.0 < self.observation_epsilon < 1.0 / self.num_classes):
            raise ValueError(
                f"observation_epsilon must be in (0, 1/num_classes={1.0 / self.num_classes:g}); "
                f"got {self.observation_epsilon}"
            )
        # B2 stop-hook delta I3 2026-05-09: int-with-bool-exclusion. Without
        # this, `min_track_length=True` would pass the `>= 2` check as 1
        # (False) with a confusing "got True" diagnostic.
        if not isinstance(self.min_track_length, int) or isinstance(self.min_track_length, bool):
            raise ValueError(
                f"min_track_length must be int; got "
                f"{type(self.min_track_length).__name__}={self.min_track_length!r}"
            )
        if self.min_track_length < 2:
            raise ValueError(
                f"min_track_length must be >= 2 (need at least 2 frames "
                f"to form a transition pair); got {self.min_track_length}"
            )
        # C3 stop-hook delta NEW-MAJOR (map_regression_tolerance_pp) 2026-05-09:
        # type-with-bool-exclusion before the range check. Without this,
        # `map_regression_tolerance_pp=True` would pass `True < 0.0` as False
        # and silently widen the deploy gate's mAP-no-regression floor from
        # the intended 0.2 pp to 1.0 pp — directly corrupting deploy/drop
        # decision correctness for any borderline-regression candidate.
        if not isinstance(self.map_regression_tolerance_pp, float) or isinstance(self.map_regression_tolerance_pp, bool):
            raise ValueError(
                f"map_regression_tolerance_pp must be float; got "
                f"{type(self.map_regression_tolerance_pp).__name__}={self.map_regression_tolerance_pp!r}"
            )
        # C3 stop-hook iter-2 NEW-MAJOR 2026-05-09: NaN/+inf bypass the
        # `< 0.0` range check (NaN < 0.0 is False, +inf < 0.0 is False) and
        # would corrupt the deploy-no-regression decision threshold. Real
        # decision-rule corruption hazard, same severity class as the bool
        # leak — finite-check before the range check.
        if not math.isfinite(self.map_regression_tolerance_pp):
            raise ValueError(
                f"map_regression_tolerance_pp must be finite; got "
                f"{self.map_regression_tolerance_pp!r}"
            )
        if self.map_regression_tolerance_pp < 0.0:
            raise ValueError(
                f"map_regression_tolerance_pp must be >= 0; "
                f"got {self.map_regression_tolerance_pp}"
            )
        # B2 stop-hook delta I3 2026-05-09: viterbi_window=True silently passes
        # `True < 1` (False) and aliases to 1, which would chunk every track at
        # length 1 and break Viterbi entirely. Same hazard as the other numeric
        # int fields — bool exclusion before the range check.
        if self.viterbi_window is not None:
            if not isinstance(self.viterbi_window, int) or isinstance(self.viterbi_window, bool):
                raise ValueError(
                    f"viterbi_window must be None or int; got "
                    f"{type(self.viterbi_window).__name__}={self.viterbi_window!r}"
                )
            if self.viterbi_window < 1:
                raise ValueError(
                    f"viterbi_window must be None or >= 1; got {self.viterbi_window} "
                    f"(C3 iter-3 NEW-MINOR 4: zero/negative window breaks chunking)"
                )
        # B2 stop-hook delta S1 2026-05-09: this cell-bounds check mirrors
        # TransitionConfig.__post_init__ — re-checked here so HmmYamlConfig
        # stands alone for tests / direct construction without going through
        # transition_config().
        # C3 stop-hook delta NEW-MINOR (illegal_transition_set) 2026-05-09:
        # validate the declared tuple-of-int-pairs shape BEFORE the bounds
        # check. Without this, a list-of-lists (loader botch) or bool class
        # IDs (which silently alias to 0/1) would slip through and corrupt
        # downstream consumers. Mask-cell IDs MUST be plain int.
        if not isinstance(self.illegal_transition_set, tuple):
            raise ValueError(
                f"illegal_transition_set must be tuple; got "
                f"{type(self.illegal_transition_set).__name__}={self.illegal_transition_set!r} "
                f"(loader must coerce YAML list → tuple before construction)"
            )
        for pair in self.illegal_transition_set:
            if not (isinstance(pair, tuple) and len(pair) == 2):
                raise ValueError(
                    f"illegal_transition_set entries must be 2-tuples; got "
                    f"{type(pair).__name__}={pair!r}"
                )
            src, dst = pair
            for cell_label, cell in (("src", src), ("dst", dst)):
                if not isinstance(cell, int) or isinstance(cell, bool):
                    raise ValueError(
                        f"illegal_transition cell {cell_label} must be int; got "
                        f"{type(cell).__name__}={cell!r} in pair {pair!r}"
                    )
            if not (0 <= src < self.num_classes and 0 <= dst < self.num_classes):
                raise ValueError(
                    f"illegal_transition cell ({src},{dst}) out of range "
                    f"[0, {self.num_classes})"
                )

    def transition_config(self) -> TransitionConfig:
        """Project this YAML config onto a ``TransitionConfig``.

        b-stage construction (C3 iter-2 NEW-MINOR 6 2026-05-09 — spell
        the call out so a future contributor cannot accidentally drop
        a field and fall back to ``TransitionConfig`` defaults):

            return TransitionConfig(
                num_classes=self.num_classes,
                laplace_alpha=self.laplace_alpha,
                illegal_transition_set=self.illegal_transition_set,
                illegal_transition_policy=self.illegal_transition_policy,
            )

        ALL FOUR fields MUST be passed; the projection is total — no
        ``TransitionConfig`` field is allowed to fall back to its own
        default, because the YAML is the runtime source of truth and
        defaults silently shipping different values is exactly the
        config-drift class this dataclass exists to prevent.

        Raises:
            NotImplementedError: a-stage scaffold.
        """
        raise NotImplementedError("b-stage")


def load_hmm_yaml(path: str | Path) -> HmmYamlConfig:
    """Parse ``configs/temporal_hmm.yaml`` into a typed config bundle.

    Coercion contract (b-stage MUST honor ALL of these):
      * ``illegal_transition_set`` (YAML list of two-element lists) →
        ``tuple(tuple(int, int) for pair in raw)``. Frozen-dataclass
        construction does NOT perform this coercion; the loader does.
      * ``illegal_transition_policy`` (YAML string) →
        ``IllegalTransitionPolicy(raw)``. Unknown values raise
        ``ValueError``.
      * ``inference_mode`` / ``observation_mode`` / ``gap_policy``
        (YAML strings) → respective enum types via
        ``EnumType(raw)``.
      * ``transition_matrix_path`` (YAML string or null) → ``Path`` or
        ``None``. Empty string is normalized to ``None``.
      * ``viterbi_window`` (C3 iter-3 NEW-MINOR 4 2026-05-09): YAML
        ``null`` maps to ``None``; an explicit integer must be
        ``>= 1`` (zero or negative raises ``ValueError``); a missing
        key raises ``ValueError`` rather than defaulting silently
        (zero-window chunking is the kind of unit bug that breaks
        long-track Viterbi without surfacing as a config error).
      * ``num_classes: null`` raises ``ValueError`` immediately — the
        runner is responsible for setting an explicit value before
        load.

    Args:
        path: filesystem path to a ``temporal_hmm.yaml`` file.

    Returns:
        ``HmmYamlConfig`` — fully validated, ready for downstream use.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: schema / range / enum violations as described above.
        NotImplementedError: a-stage scaffold.
    """
    raise NotImplementedError("b-stage")
