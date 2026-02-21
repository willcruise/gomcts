"""Deprecated shim for backwards compatibility.

The canonical implementation lives in `gomcts.utils.go_utils`.
"""

from gomcts.utils.go_utils import action_count, action_to_rc, pass_index, rc_to_action

__all__ = ["pass_index", "action_count", "rc_to_action", "action_to_rc"]

