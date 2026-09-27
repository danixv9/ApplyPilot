import sqlite3
from pathlib import Path

import pytest

from applypilot import config
from applypilot import database


@pytest.fixture(autouse=True)
def reset_connection_cache() -> None:
    if hasattr(database._local, "connections"):
        database._local.connections.clear()
    yield
    if hasattr(database._local, "connections"):
        database._local.connections.clear()


def test_resolve_app_dir_respects_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    override = tmp_path / "my-applypilot"
    monkeypatch.setenv("APPLYPILOT_DIR", str(override))
    resolved = config.resolve_app_dir()
    assert resolved == override


def test_resolve_app_dir_windows_fallback_when_default_not_writable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    default_dir = tmp_path / "home" / ".applypilot"
    fallback_dir = tmp_path / "localappdata" / "ApplyPilot"

    monkeypatch.delenv("APPLYPILOT_DIR", raising=False)
    monkeypatch.setattr(config, "_default_app_dir", lambda: default_dir)
    monkeypatch.setattr(config, "_windows_fallback_app_dir", lambda: fallback_dir)
    monkeypatch.setattr(config.platform, "system", lambda: "Windows")
    monkeypatch.setattr(config, "is_writable_dir", lambda path: path == fallback_dir)

    resolved = config.resolve_app_dir()
    assert resolved == fallback_dir


class _FakeSqliteConn:
    def __init__(self) -> None:
        self.row_factory = None
        self.calls: list[str] = []

    def execute(self, statement: str):  # pragma: no cover - trivial forwarding
        self.calls.append(statement)
        if statement == "PRAGMA journal_mode=WAL":
            raise sqlite3.OperationalError("WAL unsupported")
        return self

    def close(self) -> None:
        return


def test_get_connection_falls_back_when_wal_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "applypilot.db"
    fake_conn = _FakeSqliteConn()

    monkeypatch.setattr(database, "is_writable_dir", lambda _: True)
    monkeypatch.setattr(database.sqlite3, "connect", lambda *_args, **_kwargs: fake_conn)

    conn = database.get_connection(db_path)
    assert conn is fake_conn
    assert "PRAGMA journal_mode=WAL" in fake_conn.calls
    assert "PRAGMA journal_mode=DELETE" in fake_conn.calls


def test_get_connection_raises_when_directory_not_writable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "applypilot.db"

    monkeypatch.setattr(database, "is_writable_dir", lambda _: False)

    with pytest.raises(database.StorageInitError, match="not writable"):
        database.get_connection(db_path)
