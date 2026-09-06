"""Headless stand-ins for the Tkinter ``MainWindow``.

``ModelData`` (UVR.py) and the ``Seperate*`` engines (separate.py) read all of
their settings from a global ``root`` object that, in the desktop app, is the
Tkinter main window.  Every read is either ``root.<name>_var.get()`` or a call
to a helper method on ``root``.

Rather than re-implement ~400 lines of ``ModelData`` config logic, we build a
tiny object that quacks like ``MainWindow`` for exactly the attributes the
separation path touches, then inject it as ``UVR.root``.  This keeps us bug-for-
bug compatible with the GUI and resilient to upstream tweaks of ``ModelData``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

# Repo root (this file lives in <repo>/api/). Used for locating model dirs
# without importing the heavyweight UVR module just to read a few paths.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(REPO_ROOT, "models")
VR_MODELS_DIR = os.path.join(MODELS_DIR, "VR_Models")
MDX_MODELS_DIR = os.path.join(MODELS_DIR, "MDX_Net_Models")
DEMUCS_MODELS_DIR = os.path.join(MODELS_DIR, "Demucs_Models")
DEMUCS_NEWER_REPO_DIR = os.path.join(DEMUCS_MODELS_DIR, "v3_v4_repo")

VR_HASH_JSON = os.path.join(VR_MODELS_DIR, "model_data", "model_data.json")
MDX_HASH_JSON = os.path.join(MDX_MODELS_DIR, "model_data", "model_data.json")
MDX_MODEL_NAME_SELECT = os.path.join(MDX_MODELS_DIR, "model_data", "model_name_mapper.json")
DEMUCS_MODEL_NAME_SELECT = os.path.join(DEMUCS_MODELS_DIR, "model_data", "model_name_mapper.json")


class _Var:
    """Mimic a Tkinter ``*Var`` — the engines only ever call ``.get()``/``.set()``."""

    __slots__ = ("_value",)

    def __init__(self, value: Any):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


def _load_json(path: str) -> dict:
    import json

    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return {}


# --- Default settings -------------------------------------------------------
# Sourced from gui_data.constants.DEFAULT_DATA -- the same dict MainWindow
# uses to seed every *_var it creates. Importing it directly (instead of a
# hand-curated subset) means new settings ModelData/the engines start reading
# are covered automatically instead of surfacing one at a time as
# AttributeErrors when upstream adds a var (e.g. 'backend_mode', added
# alongside the MPS inference backend). A few entries in DEFAULT_DATA are
# plain attributes on MainWindow rather than *_var Vars (input_paths, lastDir,
# wav_type_set, model_hash_table, ...); those are set explicitly below and
# excluded from the *_var loop via _NON_VAR_KEYS.
def _default_settings() -> dict[str, Any]:
    from gui_data.constants import DEFAULT_DATA  # noqa: PLC0415

    defaults = {k: v for k, v in DEFAULT_DATA.items() if k not in _NON_VAR_KEYS}
    return {**_hardcoded_var_defaults(), **defaults}


def _hardcoded_var_defaults() -> dict[str, Any]:
    """A handful of *_var attributes MainWindow.__init__ sets directly to a
    constant rather than sourcing from DEFAULT_DATA (e.g. ``mdxnet_stems_var =
    tk.StringVar(value=ALL_STEMS)`` -- note the key mismatch with DEFAULT_DATA's
    own unrelated ``mdx_stems``/``demucs_stems`` entries). ModelData reads these
    directly, so they need to be present even though DEFAULT_DATA doesn't
    carry them. Sourced from the same constants UVR.py uses, so they stay in
    sync if upstream changes the hardcoded value."""
    from gui_data.constants import (  # noqa: PLC0415
        ALL_STEMS,
        CHOOSE_ENSEMBLE_OPTION,
        CHOOSE_STEM_PAIR,
        MAX_MIN,
    )

    return {
        "demucs_stems": ALL_STEMS,
        "mdxnet_stems": ALL_STEMS,
        "chosen_ensemble": CHOOSE_ENSEMBLE_OPTION,
        "ensemble_main_stem": CHOOSE_STEM_PAIR,
        "ensemble_type": MAX_MIN,
    }


_NON_VAR_KEYS = {
    "wav_type_set", "user_code", "export_path", "input_paths", "lastDir",
    "fileOneEntry", "fileOneEntry_Full", "fileTwoEntry", "fileTwoEntry_Full",
    "DualBatch_inputPaths", "model_hash_table", "help_hints_var",
}


@dataclass
class HeadlessRoot:
    """A duck-typed replacement for the Tkinter ``MainWindow``.

    Only the attributes/methods that ``ModelData`` and the ``Seperate*`` engines
    actually touch are implemented.  Settings provided via ``overrides`` win over
    :func:`_default_settings` (sourced from ``gui_data.constants.DEFAULT_DATA``).
    """

    overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        merged = {**_default_settings(), **self.overrides}
        for name, value in merged.items():
            setattr(self, f"{name}_var", _Var(value))

        # ``wav_type_set`` is a plain attribute on MainWindow, not a Var.
        self.wav_type_set = "PCM_16"

        # Hash / name mappers (model_data lookups). Loaded from the bundled JSON
        # caches; the registry refreshes these from the remote data links.
        self.vr_hash_MAPPER = _load_json(VR_HASH_JSON)
        self.mdx_hash_MAPPER = _load_json(MDX_HASH_JSON)
        self.mdx_name_select_MAPPER = _load_json(MDX_MODEL_NAME_SELECT)
        self.demucs_name_select_MAPPER = _load_json(DEMUCS_MODEL_NAME_SELECT)

        # Populated only if ``ModelData`` hits the "unrecognized model" popup
        # path; in headless mode we never recognize via popup, so leave empty.
        self.vr_model_params = None
        self.mdx_model_params = None

    # --- helper methods ModelData calls on root -----------------------------
    def check_only_selection_stem(self, _checktype) -> bool:
        # Controls inst-only / vocal-only splitter behaviour; off in headless.
        return False

    def return_ensemble_stems(self, is_primary=False):
        # Only reached in ensemble mode, which headless single-model never uses.
        return None, None

    def process_determine_secondary_model(self, *_args, **_kwargs):
        # No secondary-model chaining in single-model headless separation.
        return None, None

    def process_determine_demucs_pre_proc_model(self, *_args, **_kwargs):
        return None

    def process_determine_vocal_split_model(self):
        return None

    # Unrecognized-model popups: in the GUI these prompt the user. Headless has
    # no UI, so signal "unknown model" by returning None (=> model_status False).
    def pop_up_vr_param(self, _model_hash):
        self.vr_model_params = None

    def pop_up_mdx_model(self, _model_hash, _model_path):
        self.mdx_model_params = None
