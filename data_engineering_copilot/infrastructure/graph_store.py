from __future__ import annotations

import pathlib
import sqlite3


class GraphStore:
    def __init__(self, db_path: str | pathlib.Path = "data/graph_store.db") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            pathlib.Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            type TEXT NOT NULL
        )
        """)
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER NOT NULL,
            target_id INTEGER NOT NULL,
            relation_type TEXT NOT NULL,
            FOREIGN KEY (source_id) REFERENCES nodes (id) ON DELETE CASCADE,
            FOREIGN KEY (target_id) REFERENCES nodes (id) ON DELETE CASCADE,
            UNIQUE (source_id, target_id, relation_type)
        )
        """)
        self.conn.commit()

    def add_node(self, name: str, node_type: str) -> int:
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO nodes (name, type) VALUES (?, ?)", (name.strip().lower(), node_type.strip())
        )
        cursor.execute("SELECT id FROM nodes WHERE name = ?", (name.strip().lower(),))
        row = cursor.fetchone()
        self.conn.commit()
        return row[0] if row else -1

    def add_edge(self, source_name: str, target_name: str, relation_type: str) -> None:
        source_id = self.add_node(source_name, "concept")
        target_id = self.add_node(target_name, "concept")
        self.conn.execute(
            "INSERT OR IGNORE INTO edges (source_id, target_id, relation_type) VALUES (?, ?, ?)",
            (source_id, target_id, relation_type.strip()),
        )
        self.conn.commit()

    def get_neighbors(self, node_name: str, depth: int = 1) -> list[tuple[str, str, str]]:
        """Returns direct neighbors as list of tuples (source_name, relation_type, target_name)."""
        results: list[tuple[str, str, str]] = []
        name_clean = node_name.strip().lower()
        cursor = self.conn.cursor()
        cursor.execute(
            """
        SELECT n1.name, e.relation_type, n2.name
        FROM edges e
        JOIN nodes n1 ON e.source_id = n1.id
        JOIN nodes n2 ON e.target_id = n2.id
        WHERE n1.name = ? OR n2.name = ?
        """,
            (name_clean, name_clean),
        )
        for row in cursor.fetchall():
            results.append((row[0], row[1], row[2]))
        return results

    def close(self) -> None:
        self.conn.close()
