"""Clear local research memory and dashboard runs for a clean debug run."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "outputs" / "debug_runs"


def clear_database(path: Path) -> int:
    if not path.exists():
        return 0
    connection = sqlite3.connect(path)
    try:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        connection.execute("PRAGMA foreign_keys = OFF")
        for table in tables:
            escaped = table.replace('"', '""')
            connection.execute(f'DELETE FROM "{escaped}"')
        connection.commit()
        connection.execute("VACUUM")
        return len(tables)
    finally:
        connection.close()


def clear_runs() -> int:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    resolved_root = RUNS_DIR.resolve()
    count = 0
    for child in RUNS_DIR.iterdir():
        resolved_child = child.resolve()
        if resolved_child.parent != resolved_root:
            raise RuntimeError(f"Refusing to remove unsafe path: {resolved_child}")
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
        count += 1
    return count


if __name__ == "__main__":
    knowledge_tables = clear_database(DATA_DIR / "knowledge_base.db")
    session_tables = clear_database(DATA_DIR / "session_memory.db")
    runs = clear_runs()
    print(
        f"cleared knowledge_tables={knowledge_tables} "
        f"session_tables={session_tables} dashboard_runs={runs}"
    )
