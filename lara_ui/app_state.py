from __future__ import annotations

import hashlib
import json
from typing import Any, MutableMapping


DEFAULT_STATE: dict[str, Any] = {
    "loaded_log": None,
    "loaded_net": None,
    "loaded_checkpoint": None,
    "checkpoint_path": None,
    "checkpoint_metadata": {},
    "variant_index": [],
    "selected_variants": [],
    "run_configuration": {},
    "run_fingerprint": None,
    "run_status": "not_started",
    "variant_results": {},
    "execution_feed": [],
    "cancel_requested": False,
    "selected_result_id": None,
    "replay_snapshots": {},
    "run_started_at": None,
}


def initialize_state(state: MutableMapping[str, Any]) -> None:
    for key, value in DEFAULT_STATE.items():
        if key not in state:
            if isinstance(value, dict):
                value = dict(value)
            elif isinstance(value, list):
                value = list(value)
            state[key] = value


def configuration_fingerprint(configuration: dict[str, Any]) -> str:
    encoded = json.dumps(configuration, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def apply_configuration(
    state: MutableMapping[str, Any], configuration: dict[str, Any]
) -> bool:
    """Store configuration and invalidate computed results if it changed."""

    fingerprint = configuration_fingerprint(configuration)
    changed = state.get("run_fingerprint") not in {None, fingerprint}
    state["run_configuration"] = configuration
    state["run_fingerprint"] = fingerprint
    if changed:
        state["variant_results"] = {}
        state["execution_feed"] = []
        state["replay_snapshots"] = {}
        state["run_status"] = "invalidated"
        state["selected_result_id"] = None
    return changed


def clear_run(state: MutableMapping[str, Any]) -> None:
    state["variant_results"] = {}
    state["execution_feed"] = []
    state["replay_snapshots"] = {}
    state["run_status"] = "not_started"
    state["cancel_requested"] = False
    state["selected_result_id"] = None
