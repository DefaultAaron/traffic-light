"""Internal shared helpers for ``components.copy_paste_balance``.

Mirrors ``components/hard_negative_mining/_internals.py`` (B2 iter-3
review): cross-module helpers live here as the public API so the gate
and runner don't reach across module boundaries into each other's
private (``_``-prefixed) symbols.

Module-private (leading underscore on the module path): NOT re-exported
from the package ``__init__``. Submodules import directly via the deep
path.
"""

from __future__ import annotations


def is_hex_sha256(value: object) -> bool:
    """True iff ``value`` is a 64-char lowercase hex string.

    Matches the regex ``^[0-9a-f]{64}$`` baked into the JSON Schema at
    ``components/copy_paste_balance/gates/_copy_paste_decision_schema.json``.
    Single source of truth — the gate's ``ArmMetrics.__post_init__`` and
    the runner's ``_load_eval`` per-key validation both consume this
    implementation rather than re-deriving the predicate.
    """
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()
