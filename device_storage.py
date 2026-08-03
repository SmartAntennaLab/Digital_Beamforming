"""Streamlit Custom Component v2 bridge to browser-local settings storage."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import streamlit as st

from device_settings import DEVICE_STORAGE_KEY


_COMPONENT_HTML = """
<span aria-hidden="true" hidden></span>
"""
_COMPONENT_JS = """
export default function(component) {
  const { data, parentElement, setStateValue } = component
  const host = parentElement.host ?? parentElement
  const command = data?.command ?? {}
  const storageKey = String(data?.storage_key ?? "")
  const commandId = Number(command.id ?? 0)
  const action = String(command.action ?? "load")
  const signature = `${commandId}:${action}`

  if (!storageKey || host.__digitalBeamformingStorageCommand === signature) {
    return
  }
  host.__digitalBeamformingStorageCommand = signature

  try {
    if (action === "save") {
      const payload = command.payload ?? {}
      window.localStorage.setItem(storageKey, JSON.stringify(payload))
      setStateValue("loaded_settings", payload)
    } else if (action === "clear") {
      window.localStorage.removeItem(storageKey)
      setStateValue("loaded_settings", {})
    } else {
      const serialized = window.localStorage.getItem(storageKey)
      let payload = {}
      if (serialized) {
        payload = JSON.parse(serialized)
      }
      setStateValue("loaded_settings", payload)
    }
    setStateValue("status", {id: commandId, action, ok: true, error: ""})
  } catch (error) {
    setStateValue("status", {
      id: commandId,
      action,
      ok: false,
      error: String(error?.message ?? error),
    })
  }
}
"""


def _register_device_storage_component():
    return st.components.v2.component(
        "digital_beamforming_device_storage",
        html=_COMPONENT_HTML,
        js=_COMPONENT_JS,
    )


_DEVICE_STORAGE_COMPONENT = _register_device_storage_component()


def mount_device_storage(
    command: Mapping[str, Any],
    *,
    key: str = "_device_storage_component",
):
    """Mount the invisible browser-storage bridge and return its component state."""

    global _DEVICE_STORAGE_COMPONENT
    mount_arguments = {
        "key": key,
        "data": {"storage_key": DEVICE_STORAGE_KEY, "command": dict(command)},
        "width": 1,
        "height": 0,
        "on_loaded_settings_change": lambda: None,
        "on_status_change": lambda: None,
    }
    try:
        return _DEVICE_STORAGE_COMPONENT(**mount_arguments)
    except ValueError as error:
        if "is not registered" not in str(error):
            raise
        # Independent AppTest/runtime instances can reset Streamlit's component
        # registry while Python keeps this module cached. Re-register only then.
        _DEVICE_STORAGE_COMPONENT = _register_device_storage_component()
        return _DEVICE_STORAGE_COMPONENT(**mount_arguments)
