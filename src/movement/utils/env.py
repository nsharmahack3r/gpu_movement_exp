"""Environment helpers: load ``.env`` and read dataset paths."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def repo_root() -> Path:
    """Absolute path to the repo root (the directory containing ``pyproject.toml``).

    Walks up from this file so the location works regardless of where the
    package is installed from.
    """
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"Could not locate the repo root from {here}")


def load_env() -> None:
    """Load ``.env`` from the repo root if present (idempotent)."""
    load_dotenv(repo_root() / ".env")


def raw_dataset_path() -> Path:
    """Absolute path to the raw dataset directory (from ``RAW_DATASET_PATH``)."""
    load_env()
    value = os.environ.get("RAW_DATASET_PATH")
    if not value:
        raise RuntimeError(
            "RAW_DATASET_PATH is not set. Create a .env file at the repo root "
            "with RAW_DATASET_PATH=<dir> and PROCESSED_DATASET_PATH=<dir>."
        )
    path = Path(value)
    if not path.is_dir():
        raise FileNotFoundError(f"RAW_DATASET_PATH does not exist: {path}")
    return path


def processed_dataset_path() -> Path:
    """Absolute path to the processed dataset directory (from ``PROCESSED_DATASET_PATH``)."""
    load_env()
    value = os.environ.get("PROCESSED_DATASET_PATH")
    if not value:
        raise RuntimeError(
            "PROCESSED_DATASET_PATH is not set. Create a .env file at the repo root."
        )
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def gee_dataset_path() -> Path:
    """Absolute path for GEE-derived covariate outputs (from ``GEE_DATASET_PATH``).

    The wide per-dataset covariate CSV is written here: one row per fix, every
    completed source's columns joined on ``fix_id``. Distinct from
    ``results/covariates``, which holds the per-source Parquet archive.
    """
    load_env()
    value = os.environ.get("GEE_DATASET_PATH")
    if not value:
        raise RuntimeError(
            "GEE_DATASET_PATH is not set. Add it to .env, e.g. "
            "GEE_DATASET_PATH=<dir> (the wide per-dataset covariate CSV is written "
            "there; the per-source Parquet archive still goes to results/covariates/)."
        )
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


# --- Earth Engine / GCS / Drive --------------------------------------------- #
# Auth modes. The service-account variables are optional in `interactive` mode,
# but GEE_PROJECT is required in BOTH: Earth Engine refuses to initialise without
# a registered Cloud project when the credentials are not a service account
# (`ee.Initialize` raises NO_PROJECT_EXCEPTION).
GEE_AUTH_MODES = ("auto", "service_account", "interactive")
GEE_SERVICE_ACCOUNT_VARS = ("GEE_SERVICE_ACCOUNT", "GEE_KEY_FILE")
GEE_ALWAYS_REQUIRED_VARS = ("GEE_PROJECT",)

# Where batch exports are delivered, and the variable naming each destination.
# Earth Engine computes remotely and cannot write to a local path, so a batch
# export must land in one of Google's stores; `drive` needs no billing account.
GEE_TARGET_VARS = {
    "cloud_storage": "GCS_BUCKET",
    "drive": "GEE_DRIVE_FOLDER",
}
GEE_TARGET_WHY = {
    "cloud_storage": "Batch exports are written to Cloud Storage, so a writable bucket is required.",
    "drive": ("Batch exports are written to a Google Drive folder, so the folder "
              "name is required (Earth Engine creates the folder on the first export)."),
}

_AUTH_WHY = (
    "Earth Engine needs a *registered* Cloud project; it is required for both "
    "service-account and interactive credentials."
)


def _set(name: str) -> str:
    return os.environ.get(name, "").strip()


def _require(name: str, *, why: str = _AUTH_WHY) -> str:
    value = _set(name)
    if not value:
        raise RuntimeError(f"{name} is not set. {why} Add it to .env.")
    return value


def service_account_configured() -> bool:
    """True when both service-account variables are present."""
    load_env()
    return all(_set(var) for var in GEE_SERVICE_ACCOUNT_VARS)


def resolve_auth_mode(explicit: str | None = None) -> str:
    """Resolve ``'service_account'`` or ``'interactive'``.

    Precedence: an explicit ``--auth`` value, then ``GEE_AUTH`` from ``.env``,
    then auto-detection (service account when it is configured, else interactive).

    Pinning ``service_account`` does **not** degrade to interactive when the
    variables are missing; it fails loudly, so a run whose provenance must be
    reproducible from config alone cannot silently pick up a personal login.
    """
    load_env()
    requested = (explicit or "").strip().lower() or _set("GEE_AUTH").lower() or "auto"
    if requested not in GEE_AUTH_MODES:
        raise RuntimeError(
            f"Unknown Earth Engine auth mode {requested!r}; expected one of "
            f"{GEE_AUTH_MODES} (from --auth or GEE_AUTH)."
        )
    if requested == "service_account":
        missing = [var for var in GEE_SERVICE_ACCOUNT_VARS if not _set(var)]
        if missing:
            verb = "is" if len(missing) == 1 else "are"
            raise RuntimeError(
                f"Auth mode 'service_account' was requested but {' and '.join(missing)} "
                f"{verb} not set in .env. This pin exists so a run cannot silently fall "
                f"back to an interactive personal login. Set the variable(s), or allow "
                f"the fallback with --auth interactive / GEE_AUTH=interactive."
            )
        return "service_account"
    if requested == "interactive":
        return "interactive"
    return "service_account" if service_account_configured() else "interactive"


def _service_account_credentials(key_file: str) -> Any:
    try:
        from google.oauth2 import service_account
    except ImportError as exc:
        raise RuntimeError(
            "google-auth is required for Earth Engine. Install the GEE extra: "
            "`uv sync --extra gee`."
        ) from exc
    return service_account.Credentials.from_service_account_file(key_file)


def _interactive_credential_kwargs() -> dict[str, Any]:
    """google-auth kwargs built from the cached interactive Earth Engine login.

    ``ee.oauth`` stores only a refresh token and the scopes; the OAuth client id
    and secret are the Earth Engine client compiled into the library, so the
    assembled arguments are requested from the library rather than parsed from the
    file here. The stored scopes include ``devstorage.full_control``, which is what
    lets the same login read the export destination back in ``join``.

    ``token`` is passed as ``None`` explicitly: ``google.oauth2.credentials.
    Credentials`` requires it positionally and ``ee.oauth.get_credentials_arguments``
    does not return it. This mirrors ``ee.data.get_persistent_credentials``, which
    does ``Credentials(None, **args)``; the refresh afterwards fills the access
    token in. A future ``ee`` version that *does* return ``token`` wins over our
    ``None``, because it is merged second.
    """
    try:
        from ee import oauth as ee_oauth
    except ImportError as exc:
        raise RuntimeError(
            "earthengine-api is required for interactive Earth Engine auth. Install "
            "the GEE extra: `uv sync --extra gee`."
        ) from exc
    path = ee_oauth.get_credentials_path()
    if not os.path.exists(path):
        raise RuntimeError(
            f"No interactive Earth Engine credentials found at {path}. Run "
            f"`uv run earthengine authenticate` first, or set "
            f"{' and '.join(GEE_SERVICE_ACCOUNT_VARS)} in .env for service-account auth."
        )
    args = ee_oauth.get_credentials_arguments()
    if not args.get("refresh_token"):
        raise RuntimeError(
            f"The interactive Earth Engine credentials at {path} contain no "
            f"refresh_token. Re-run `uv run earthengine authenticate`."
        )
    return {"token": None, **args}


def _interactive_credentials() -> Any:
    """Cached interactive login, refreshed and ready for Earth Engine and Drive."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
    except ImportError as exc:
        raise RuntimeError(
            "google-auth is required for interactive Earth Engine auth. Install the "
            "GEE extra: `uv sync --extra gee`."
        ) from exc
    credentials = Credentials(**_interactive_credential_kwargs())
    credentials.refresh(Request())
    return credentials


def gee_auth(explicit: str | None = None, destination: str = "cloud_storage") -> dict[str, Any]:
    """Everything needed to talk to Earth Engine and the export destination.

    ``destination`` selects which target variable is required: ``GCS_BUCKET`` for
    ``cloud_storage`` or ``GEE_DRIVE_FOLDER`` for ``drive``. Returns ``mode``,
    ``project``, ``destination``, ``target`` (bucket name or folder name),
    ``credentials`` (a google-auth object usable for ``ee.Initialize``, the
    Storage client and the Drive client), ``service_account`` and ``key_file``.
    """
    load_env()
    if destination not in GEE_TARGET_VARS:
        raise RuntimeError(
            f"Unknown export destination {destination!r}; expected one of "
            f"{sorted(GEE_TARGET_VARS)} (set export.destination in "
            f"configs/covariates/sources.yaml)."
        )
    mode = resolve_auth_mode(explicit)
    project = _require("GEE_PROJECT")
    target_name = GEE_TARGET_VARS[destination]
    target = _require(target_name, why=GEE_TARGET_WHY[destination])
    if mode == "service_account":
        key_file = _require("GEE_KEY_FILE")
        if not Path(key_file).is_file():
            raise FileNotFoundError(
                f"GEE_KEY_FILE does not point at a readable file: {key_file}"
            )
        return {
            "mode": mode,
            "project": project,
            "destination": destination,
            "target": target,
            "credentials": _service_account_credentials(key_file),
            "service_account": _require("GEE_SERVICE_ACCOUNT"),
            "key_file": key_file,
        }
    return {
        "mode": mode,
        "project": project,
        "destination": destination,
        "target": target,
        "credentials": _interactive_credentials(),
        "service_account": None,
        "key_file": None,
    }
