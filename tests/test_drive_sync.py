import os
import json
import pytest
from unittest.mock import patch, MagicMock

from backend.ingestion.drive_sync import DriveSyncer


def test_drive_syncer_missing_env_vars(monkeypatch):
    """Test that DriveSyncer raises ValueError if env vars are missing."""
    monkeypatch.delenv("GOOGLE_SERVICE_ACCOUNT_JSON", raising=False)
    monkeypatch.delenv("GOOGLE_DRIVE_FOLDER_ID", raising=False)
    
    with pytest.raises(ValueError) as exc:
        DriveSyncer()
    assert "Missing GOOGLE_SERVICE_ACCOUNT_JSON" in str(exc.value)

def test_drive_syncer_missing_folder_id(monkeypatch):
    """Test that DriveSyncer raises ValueError if folder ID is missing."""
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON", "{}")
    monkeypatch.delenv("GOOGLE_DRIVE_FOLDER_ID", raising=False)
    
    with pytest.raises(ValueError) as exc:
        DriveSyncer()
    assert "Missing GOOGLE_DRIVE_FOLDER_ID" in str(exc.value)

@patch("backend.ingestion.drive_sync.build")
@patch("backend.ingestion.drive_sync.service_account.Credentials.from_service_account_info")
def test_drive_syncer_successful_init(mock_from_info, mock_build, monkeypatch):
    """Test that DriveSyncer initializes properly and calls from_service_account_info."""
    fake_creds = {"type": "service_account", "project_id": "test"}
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON", json.dumps(fake_creds))
    monkeypatch.setenv("GOOGLE_DRIVE_FOLDER_ID", "12345abcdef")
    
    mock_from_info.return_value = MagicMock()
    mock_build.return_value = MagicMock()
    
    syncer = DriveSyncer()
    
    mock_from_info.assert_called_once_with(fake_creds, scopes=["https://www.googleapis.com/auth/drive.readonly"])
    mock_build.assert_called_once_with("drive", "v3", credentials=mock_from_info.return_value, cache_discovery=False)
    
    # Ensure folder id was captured
    assert syncer._folder_id == "12345abcdef"

@patch("backend.ingestion.drive_sync.build")
@patch("backend.ingestion.drive_sync.service_account.Credentials.from_service_account_info")
def test_drive_syncer_invalid_json(mock_from_info, mock_build, monkeypatch):
    """Test that DriveSyncer raises RuntimeError if the JSON is malformed."""
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON", "INVALID_JSON_STRING")
    monkeypatch.setenv("GOOGLE_DRIVE_FOLDER_ID", "12345abcdef")
    
    with pytest.raises(RuntimeError) as exc:
        DriveSyncer()
    assert "Invalid GOOGLE_SERVICE_ACCOUNT_JSON structure or auth failure" in str(exc.value)
