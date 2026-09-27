"""ApplyPilot database layer: schema, migrations, stats, and connection helpers.

Single source of truth for the jobs table schema. All columns from every
pipeline stage are created up front so any stage can run independently
without migration ordering issues.
"""

import sqlite3
import threading
import logging
from datetime import datetime, timezone
from pathlib import Path

from applypilot.config import DB_PATH, is_writable_dir

# Thread-local connection storage — each thread gets its own connection
# (required for SQLite thread safety with parallel workers)
_local = threading.local()
log = logging.getLogger(__name__)


class StorageInitError(RuntimeError):
    """Raised when ApplyPilot storage cannot be initialized safely."""


def _ensure_writable_parent(path: Path) -> None:
    """Ensure the database parent directory is writable."""
    parent = path.parent
    if is_writable_dir(parent):
        return
    raise StorageInitError(
        f"Data directory is not writable: {parent}"
    )


def _configure_journal_mode(conn: sqlite3.Connection, db_path: Path) -> None:
    """Prefer WAL mode; fall back to DELETE mode if WAL is unavailable."""
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        return
    except sqlite3.OperationalError as wal_error:
        try:
            conn.execute("PRAGMA journal_mode=DELETE")
            log.warning(
                "WAL unavailable for %s (%s). Falling back to DELETE journal mode.",
                db_path,
                wal_error,
            )
            return
        except sqlite3.OperationalError as delete_error:
            raise StorageInitError(
                f"Unable to set SQLite journal mode for {db_path}"
            ) from delete_error


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Get a thread-local cached SQLite connection with WAL mode enabled.

    Each thread gets its own connection (required for SQLite thread safety).
    Connections are cached and reused within the same thread.

    Args:
        db_path: Override the default DB_PATH. Useful for testing.

    Returns:
        sqlite3.Connection configured with WAL mode and row factory.
    """
    db_file = Path(db_path or DB_PATH)
    path = str(db_file)

    if not hasattr(_local, 'connections'):
        _local.connections = {}

    conn = _local.connections.get(path)
    if conn is not None:
        try:
            conn.execute("SELECT 1")
            return conn
        except sqlite3.ProgrammingError:
            pass

    _ensure_writable_parent(db_file)

    try:
        conn = sqlite3.connect(path, timeout=30)
    except sqlite3.OperationalError as exc:
        raise StorageInitError(
            f"Unable to open database file: {db_file}"
        ) from exc

    try:
        _configure_journal_mode(conn, db_file)
    except StorageInitError:
        conn.close()
        raise

    conn.execute("PRAGMA busy_timeout=10000")
    conn.row_factory = sqlite3.Row
    _local.connections[path] = conn
    return conn


def close_connection(db_path: Path | str | None = None) -> None:
    """Close the cached connection for the current thread."""
    path = str(db_path or DB_PATH)
    if hasattr(_local, 'connections'):
        conn = _local.connections.pop(path, None)
        if conn is not None:
            conn.close()


def init_db(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Create the full jobs table with all columns from every pipeline stage.

    This is idempotent -- safe to call on every startup. Uses CREATE TABLE IF NOT EXISTS
    so it won't destroy existing data.

    Schema columns by stage:
      - Discovery:  url, title, salary, description, location, site, strategy, discovered_at
      - Enrichment: full_description, application_url, detail_scraped_at, detail_error
      - Scoring:    fit_score, score_reasoning, scored_at
      - Tailoring:  tailored_resume_path, tailored_at, tailor_attempts
      - Cover:      cover_letter_path, cover_letter_at, cover_attempts
      - Apply:      applied_at, apply_status, apply_error, apply_attempts,
                   agent_id, last_attempted_at, apply_duration_ms, apply_task_id,
                   verification_confidence

    Args:
        db_path: Override the default DB_PATH.

    Returns:
        sqlite3.Connection with the schema initialized.
    """
    path = db_path or DB_PATH

    # Ensure parent directory exists
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise StorageInitError(
            f"Failed to create database directory: {Path(path).parent}"
        ) from exc

    conn = get_connection(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            -- Discovery stage (smart_extract / job_search)
            url                   TEXT PRIMARY KEY,
            title                 TEXT,
            salary                TEXT,
            description           TEXT,
            location              TEXT,
            site                  TEXT,
            strategy              TEXT,
            discovered_at         TEXT,

            -- Enrichment stage (detail_scraper)
            full_description      TEXT,
            application_url       TEXT,
            detail_scraped_at     TEXT,
            detail_error          TEXT,

            -- Scoring stage (job_scorer)
            fit_score             INTEGER,
            score_reasoning       TEXT,
            scored_at             TEXT,

            -- Tailoring stage (resume tailor)
            tailored_resume_path  TEXT,
            tailored_at           TEXT,
            tailor_attempts       INTEGER DEFAULT 0,

            -- Cover letter stage
            cover_letter_path     TEXT,
            cover_letter_at       TEXT,
            cover_attempts        INTEGER DEFAULT 0,

            -- Application stage
            applied_at            TEXT,
            apply_status          TEXT,
            apply_error           TEXT,
            apply_attempts        INTEGER DEFAULT 0,
            agent_id              TEXT,
            last_attempted_at     TEXT,
            apply_duration_ms     INTEGER,
            apply_task_id         TEXT,
            verification_confidence TEXT
        )
    """)
    conn.commit()

    # Run migrations for any columns added after initial schema
    ensure_columns(conn)

    return conn


# Complete column registry: column_name -> SQL type with optional default.
# This is the single source of truth. Adding a column here is all that's needed
# for it to appear in both new databases and migrated ones.
_ALL_COLUMNS: dict[str, str] = {
    # Discovery
    "url": "TEXT PRIMARY KEY",
    "title": "TEXT",
    "salary": "TEXT",
    "description": "TEXT",
    "location": "TEXT",
    "site": "TEXT",
    "strategy": "TEXT",
    "discovered_at": "TEXT",
    # Enrichment
    "full_description": "TEXT",
    "application_url": "TEXT",
    "detail_scraped_at": "TEXT",
    "detail_error": "TEXT",
    # Scoring
    "fit_score": "INTEGER",
    "score_reasoning": "TEXT",
    "scored_at": "TEXT",
    # Tailoring
    "tailored_resume_path": "TEXT",
    "tailored_at": "TEXT",
    "tailor_attempts": "INTEGER DEFAULT 0",
    # Cover letter
    "cover_letter_path": "TEXT",
    "cover_letter_at": "TEXT",
    "cover_attempts": "INTEGER DEFAULT 0",
    # Application
    "applied_at": "TEXT",
    "apply_status": "TEXT",
    "apply_error": "TEXT",
    "apply_attempts": "INTEGER DEFAULT 0",
    "agent_id": "TEXT",
    "last_attempted_at": "TEXT",
    "apply_duration_ms": "INTEGER",
    "apply_task_id": "TEXT",
    "verification_confidence": "TEXT",
}


def ensure_columns(conn: sqlite3.Connection | None = None) -> list[str]:
    """Add any missing columns to the jobs table (forward migration).

    Reads the current table schema via PRAGMA table_info and compares against
    the full column registry. Any missing columns are added with ALTER TABLE.

    This makes it safe to upgrade the database from any previous version --
    columns are only added, never removed or renamed.

    Args:
        conn: Database connection. Uses get_connection() if None.

    Returns:
        List of column names that were added (empty if schema was already current).
    """
    if conn is None:
        conn = get_connection()

    existing = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    added = []

    for col, dtype in _ALL_COLUMNS.items():
        if col not in existing:
            # PRIMARY KEY columns can't be added via ALTER TABLE, but url
            # is always created with the table itself so this is safe
            if "PRIMARY KEY" in dtype:
                continue
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {dtype}")
            added.append(col)

    if added:
        conn.commit()

    return added


def get_stats(conn: sqlite3.Connection | None = None) -> dict:
    """Return job counts by pipeline stage.

    Provides a snapshot of how many jobs are at each stage, useful for
    dashboard display and pipeline progress tracking.

    Args:
        conn: Database connection. Uses get_connection() if None.

    Returns:
        Dictionary with keys:
            total, by_site, pending_detail, with_description,
            scored, unscored, tailored, untailored_eligible,
            with_cover_letter, applied, score_distribution
    """
    if conn is None:
        conn = get_connection()

    stats: dict = {}

    # Total jobs
    stats["total"] = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]

    # By site breakdown
    rows = conn.execute(
        "SELECT site, COUNT(*) as cnt FROM jobs GROUP BY site ORDER BY cnt DESC"
    ).fetchall()
    stats["by_site"] = [(row[0], row[1]) for row in rows]

    # Enrichment stage
    stats["pending_detail"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE detail_scraped_at IS NULL"
    ).fetchone()[0]

    stats["with_description"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE full_description IS NOT NULL"
    ).fetchone()[0]

    stats["detail_errors"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE detail_error IS NOT NULL"
    ).fetchone()[0]

    # Scoring stage
    stats["scored"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE fit_score IS NOT NULL"
    ).fetchone()[0]

    stats["unscored"] = conn.execute(
        "SELECT COUNT(*) FROM jobs "
        "WHERE full_description IS NOT NULL AND fit_score IS NULL"
    ).fetchone()[0]

    # Score distribution
    dist_rows = conn.execute(
        "SELECT fit_score, COUNT(*) as cnt FROM jobs "
        "WHERE fit_score IS NOT NULL "
        "GROUP BY fit_score ORDER BY fit_score DESC"
    ).fetchall()
    stats["score_distribution"] = [(row[0], row[1]) for row in dist_rows]

    # Tailoring stage
    stats["tailored"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE tailored_resume_path IS NOT NULL"
    ).fetchone()[0]

    stats["untailored_eligible"] = conn.execute(
        "SELECT COUNT(*) FROM jobs "
        "WHERE fit_score >= 7 "
        "AND COALESCE(NULLIF(full_description, ''), NULLIF(description, '')) IS NOT NULL "
        "AND tailored_resume_path IS NULL"
    ).fetchone()[0]

    stats["tailor_exhausted"] = conn.execute(
        "SELECT COUNT(*) FROM jobs "
        "WHERE COALESCE(tailor_attempts, 0) >= 5 "
        "AND tailored_resume_path IS NULL"
    ).fetchone()[0]

    # Cover letter stage
    stats["with_cover_letter"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE cover_letter_path IS NOT NULL"
    ).fetchone()[0]

    stats["cover_exhausted"] = conn.execute(
        "SELECT COUNT(*) FROM jobs "
        "WHERE COALESCE(cover_attempts, 0) >= 5 "
        "AND (cover_letter_path IS NULL OR cover_letter_path = '')"
    ).fetchone()[0]

    # Application stage
    stats["applied"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE applied_at IS NOT NULL"
    ).fetchone()[0]

    stats["apply_errors"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE apply_error IS NOT NULL"
    ).fetchone()[0]

    stats["ready_to_apply"] = conn.execute(
        "SELECT COUNT(*) FROM jobs "
        "WHERE tailored_resume_path IS NOT NULL "
        "AND applied_at IS NULL "
        "AND application_url IS NOT NULL"
    ).fetchone()[0]

    return stats


def store_jobs(conn: sqlite3.Connection, jobs: list[dict],
               site: str, strategy: str) -> tuple[int, int]:
    """Store discovered jobs, skipping duplicates by URL.

    Args:
        conn: Database connection.
        jobs: List of job dicts with keys: url, title, salary, description, location.
        site: Source site name (e.g. "RemoteOK", "Dice").
        strategy: Extraction strategy used (e.g. "json_ld", "api_response", "css_selectors").

    Returns:
        Tuple of (new_count, duplicate_count).
    """
    now = datetime.now(timezone.utc).isoformat()
    new = 0
    existing = 0

    for job in jobs:
        url = job.get("url")
        if not url:
            continue
        try:
            conn.execute(
                "INSERT INTO jobs (url, title, salary, description, location, site, strategy, discovered_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (url, job.get("title"), job.get("salary"), job.get("description"),
                 job.get("location"), site, strategy, now),
            )
            new += 1
        except sqlite3.IntegrityError:
            existing += 1

    conn.commit()
    return new, existing


def get_jobs_by_stage(conn: sqlite3.Connection | None = None,
                      stage: str = "discovered",
                      min_score: int | None = None,
                      limit: int = 100) -> list[dict]:
    """Fetch jobs filtered by pipeline stage.

    Args:
        conn: Database connection. Uses get_connection() if None.
        stage: One of "discovered", "enriched", "scored", "tailored", "applied".
        min_score: Minimum fit_score filter (only relevant for scored+ stages).
        limit: Maximum number of rows to return.

    Returns:
        List of job dicts.
    """
    if conn is None:
        conn = get_connection()

    conditions = {
        "discovered": "1=1",
        "pending_detail": "detail_scraped_at IS NULL",
        "enriched": "full_description IS NOT NULL",
        "pending_score": "full_description IS NOT NULL AND fit_score IS NULL",
        "scored": "fit_score IS NOT NULL",
        "pending_tailor": (
            "fit_score >= ? "
            "AND COALESCE(NULLIF(full_description, ''), NULLIF(description, '')) IS NOT NULL "
            "AND tailored_resume_path IS NULL AND COALESCE(tailor_attempts, 0) < 5"
        ),
        "tailored": "tailored_resume_path IS NOT NULL",
        "pending_apply": (
            "tailored_resume_path IS NOT NULL AND applied_at IS NULL "
            "AND application_url IS NOT NULL"
        ),
        "applied": "applied_at IS NOT NULL",
    }

    where = conditions.get(stage, "1=1")
    params: list = []

    if "?" in where and min_score is not None:
        params.append(min_score)
    elif "?" in where:
        params.append(7)  # default min_score

    if min_score is not None and "fit_score" not in where and stage in ("scored", "tailored", "applied"):
        where += " AND fit_score >= ?"
        params.append(min_score)

    order_by = "fit_score DESC NULLS LAST, discovered_at DESC"
    if stage == "pending_tailor":
        # Prioritize ATS domains that the Playwright apply engine can submit.
        order_by = (
            "CASE WHEN ("
            "lower(coalesce(application_url, url)) LIKE '%jobs.ashbyhq.com%' "
            "OR lower(coalesce(application_url, url)) LIKE '%jobs.lever.co%' "
            "OR lower(coalesce(application_url, url)) LIKE '%lever.co%' "
            "OR lower(coalesce(application_url, url)) LIKE '%greenhouse.io%'"
            ") THEN 0 ELSE 1 END, "
            "fit_score DESC NULLS LAST, discovered_at DESC"
        )

    query = f"SELECT * FROM jobs WHERE {where} ORDER BY {order_by}"
    if limit > 0:
        query += " LIMIT ?"
        params.append(limit)

    rows = conn.execute(query, params).fetchall()

    # Convert sqlite3.Row objects to dicts
    if rows:
        columns = rows[0].keys()
        return [dict(zip(columns, row)) for row in rows]
    return []
