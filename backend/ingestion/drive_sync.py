"""Google Drive Auto-Sync module for retrieving knowledge base PDFs."""

from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

logger = logging.getLogger(__name__)


class DriveSyncer:
    """Synchronizes PDFs from a Google Drive folder to a local directory."""

    def __init__(self, data_path: str | Path = "data") -> None:
        self._data_path = Path(data_path)
        self._state_path = self._data_path / ".drive_sync_state.json"
        
        # We must authenticate exclusively using environment variables.
        # Ensure we never log or expose any credential values.
        sa_json_str = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
        self._folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "").strip()

        if not sa_json_str:
            raise ValueError(
                "Missing GOOGLE_SERVICE_ACCOUNT_JSON environment variable. "
                "Cannot sync with Google Drive."
            )
        if not self._folder_id:
            raise ValueError(
                "Missing GOOGLE_DRIVE_FOLDER_ID environment variable. "
                "Cannot sync with Google Drive."
            )

        try:
            # Parse the JSON string from GOOGLE_SERVICE_ACCOUNT_JSON directly in memory.
            creds_info = json.loads(sa_json_str)
            self._creds = service_account.Credentials.from_service_account_info(
                creds_info, scopes=["https://www.googleapis.com/auth/drive.readonly"]
            )
            # Never hold onto the raw JSON string or dict longer than necessary.
            del creds_info
            del sa_json_str
            
            self._service = build("drive", "v3", credentials=self._creds, cache_discovery=False)
        except Exception as exc:
            logger.error("Failed to authenticate with Google Drive using provided JSON credentials.")
            raise RuntimeError("Invalid GOOGLE_SERVICE_ACCOUNT_JSON structure or auth failure.") from exc

    def _load_state(self) -> dict[str, str]:
        """Load local sync state mapping filename to modifiedTime."""
        if self._state_path.exists():
            try:
                with open(self._state_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                logger.warning("Could not read drive sync state: %s. Rebuilding.", exc)
        return {}

    def _save_state(self, state: dict[str, str]) -> None:
        """Save local sync state."""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception as exc:
            logger.error("Failed to save drive sync state: %s", exc)

    def sync_pdfs(self) -> list[str]:
        """Download new or modified PDFs from Drive to the local data directory.
        
        Returns:
            A list of filenames that were newly downloaded or updated.
        """
        logger.info("Connecting to Google Drive to check for new PDFs...")
        self._data_path.mkdir(parents=True, exist_ok=True)
        state = self._load_state()
        new_or_modified_files = []

        try:
            results = self._service.files().list(
                q=f"'{self._folder_id}' in parents and mimeType='application/pdf' and trashed=false",
                fields="files(id, name, modifiedTime)",
                pageSize=1000
            ).execute()
            
            items = results.get("files", [])
            logger.info("Found %d PDFs in Google Drive folder.", len(items))

            for item in items:
                file_id = item["id"]
                file_name = item["name"]
                modified_time = item.get("modifiedTime", "")
                
                local_file_path = self._data_path / file_name
                
                # Check if it's new or modified compared to our saved state
                is_new = file_name not in state
                is_modified = file_name in state and state[file_name] != modified_time
                is_missing_locally = not local_file_path.exists()

                if is_new or is_modified or is_missing_locally:
                    logger.info("Downloading PDF: %s", file_name)
                    request = self._service.files().get_media(fileId=file_id)
                    fh = io.BytesIO()
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while done is False:
                        status, done = downloader.next_chunk()
                    
                    with open(local_file_path, "wb") as f:
                        f.write(fh.getvalue())
                    
                    state[file_name] = modified_time
                    new_or_modified_files.append(file_name)
                else:
                    logger.debug("Skipping unmodified PDF: %s", file_name)
                    
            self._save_state(state)
            if new_or_modified_files:
                logger.info("Successfully downloaded %d new/modified PDFs.", len(new_or_modified_files))
            else:
                logger.info("All local PDFs are up to date with Google Drive.")
                
            return new_or_modified_files
            
        except Exception as exc:
            logger.error("Failed to sync with Google Drive: %s", exc)
            # We don't want to crash the whole server if Drive is temporarily down,
            # but we will return whatever we have.
            return []
