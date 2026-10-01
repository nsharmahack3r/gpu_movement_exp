"""Google Drive access for the export path.

Batch exports can land in a Drive folder instead of a Cloud Storage bucket
(``export.destination: drive``). This module reads them back.

It talks to the Drive REST v3 API directly with the same google-auth credentials
used for Earth Engine — the stored login already carries the ``drive`` scope, so
no extra dependency and no separate credential is needed. Every HTTP call goes
through an injectable ``transport`` so the whole module is testable offline.

Note that Earth Engine *creates* the destination folder when it runs the first
``Export.table.toDrive``, so this module only ever needs to find an existing
folder, never to create one.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

DRIVE_API = "https://www.googleapis.com/drive/v3/files"
FOLDER_MIME = "application/vnd.google-apps.folder"

# transport(method, url, headers, body) -> response bytes
Transport = Callable[[str, str, dict[str, str], bytes | None], bytes]


class DriveError(RuntimeError):
    """A Drive API call failed. Never swallowed: the reason is in the message."""


def _default_transport(method: str, url: str, headers: dict[str, str],
                       body: bytes | None) -> bytes:
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()
    except urllib.error.HTTPError as exc:  # pragma: no cover - needs network
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise DriveError(f"Drive {method} {url} failed: HTTP {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:  # pragma: no cover - needs network
        raise DriveError(f"Drive {method} {url} failed: {exc.reason}") from exc


@dataclass(frozen=True)
class DriveFile:
    id: str
    name: str


class DriveClient:
    """Minimal Drive v3 client: find a folder, list its children, download one."""

    def __init__(self, credentials: Any, transport: Transport | None = None) -> None:
        self._credentials = credentials
        self._transport = transport or _default_transport

    # -- plumbing -----------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        """Authorisation header, refreshing the token when it has expired."""
        if not getattr(self._credentials, "token", None) or (
            getattr(self._credentials, "expired", False)
        ):
            from google.auth.transport.requests import Request

            self._credentials.refresh(Request())
        return {"Authorization": f"Bearer {self._credentials.token}"}

    def _json(self, url: str, params: dict[str, str]) -> dict:
        query = urllib.parse.urlencode(params)
        raw = self._transport("GET", f"{url}?{query}", self._headers(), None)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DriveError(f"Drive returned non-JSON for {url}: {raw[:200]!r}") from exc

    # -- API ----------------------------------------------------------------
    def find_folder(self, name: str) -> str:
        """Folder id for ``name``, or fail loudly with the configured name."""
        escaped = name.replace("'", "\\'")
        payload = self._json(DRIVE_API, {
            "q": (f"name = '{escaped}' and mimeType = '{FOLDER_MIME}' "
                  f"and trashed = false"),
            "fields": "files(id,name)",
            "pageSize": "10",
        })
        files = payload.get("files") or []
        if not files:
            raise DriveError(
                f"No Drive folder named {name!r} was found. Earth Engine creates it "
                f"on the first successful `submit`; check GEE_DRIVE_FOLDER matches "
                f"the folder the tasks wrote to."
            )
        if len(files) > 1:
            logger.warning("Multiple Drive folders named %r; using the first (%s).",
                           name, files[0]["id"])
        return files[0]["id"]

    def list_children(self, folder_id: str, contains: str) -> list[DriveFile]:
        """Files in a folder whose name contains ``contains``.

        ``contains`` rather than an equality match, because a table export may be
        sharded into several files with suffixed names. The chunk ids are hashes,
        so the substring is unambiguous.
        """
        escaped = contains.replace("'", "\\'")
        payload = self._json(DRIVE_API, {
            "q": (f"'{folder_id}' in parents and name contains '{escaped}' "
                  f"and trashed = false"),
            "fields": "files(id,name)",
            "pageSize": "100",
        })
        return [DriveFile(id=f["id"], name=f["name"]) for f in payload.get("files") or []]

    def download(self, file_id: str) -> bytes:
        """File content (``alt=media``)."""
        return self._transport(
            "GET", f"{DRIVE_API}/{file_id}?alt=media", self._headers(), None
        )

    def fetch(self, folder: str, contains: str) -> list[tuple[str, bytes]]:
        """``(name, content)`` for every file in ``folder`` matching ``contains``."""
        folder_id = self.find_folder(folder)
        files = self.list_children(folder_id, contains)
        if not files:
            raise DriveError(
                f"No file matching {contains!r} in Drive folder {folder!r} — the "
                f"chunk may not have finished, or it wrote somewhere else."
            )
        return [(f.name, self.download(f.id)) for f in files]


def drive_client(auth: dict, transport: Transport | None = None) -> DriveClient:
    """A :class:`DriveClient` using the resolved Earth Engine credentials."""
    return DriveClient(auth["credentials"], transport=transport)
