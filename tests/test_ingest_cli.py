import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_ingest(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "backend.ingest", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_ingest_cli_rebuilds_index_from_markdown(tmp_path: Path) -> None:
    docs_path = tmp_path / "documents"
    docs_path.mkdir()
    (docs_path / "placements.md").write_text(
        "# Placement guide\n\nPractice DSA, DBMS, operating systems, and project explanations.",
        encoding="utf-8",
    )
    index_path = tmp_path / "vector_index.json"

    result = run_ingest(
        "--documents-path",
        str(docs_path),
        "--vector-db-path",
        str(index_path),
        "--chunk-size",
        "50",
        "--chunk-overlap",
        "5",
    )

    assert result.returncode == 0, result.stderr
    assert "[OK] Vector index written" in result.stdout
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert len(payload["documents"]) == 1
    indexed = payload["documents"][0]
    assert indexed["metadata"]["filename"] == "placements.md"
    assert "operating systems" in indexed["text"]


def test_ingest_cli_returns_error_for_missing_documents_dir(tmp_path: Path) -> None:
    result = run_ingest(
        "--documents-path",
        str(tmp_path / "missing"),
        "--vector-db-path",
        str(tmp_path / "vector_index.json"),
    )

    assert result.returncode == 1
    assert "Documents directory" in result.stderr
