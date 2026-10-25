"""SQLite storage for graph triples, benchmark questions, and labels."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path

from .errors import TraceError
from .schema import Example, Triple


class GraphStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    @contextmanager
    def connection(self, write: bool = False):
        if write:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path, timeout=60)
        else:
            if not self.path.is_file():
                raise TraceError(
                    "graph_missing", "Build the graph with trace build-graph first", 503
                )
            conn = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=60)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            if write:
                conn.commit()
        except BaseException:
            if write:
                conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self, scope: str = "benchmark_union"):
        with self.connection(write=True) as c:
            c.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS entities(
                    id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    urls TEXT NOT NULL DEFAULT '[]'
                );
                CREATE INDEX IF NOT EXISTS entities_label ON entities(label COLLATE NOCASE);
                CREATE TABLE IF NOT EXISTS triples(
                    id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    UNIQUE(subject,predicate,object)
                );
                CREATE INDEX IF NOT EXISTS triples_subject ON triples(subject);
                CREATE INDEX IF NOT EXISTS triples_object ON triples(object);
                CREATE TABLE IF NOT EXISTS questions(
                    id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    split TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS predicates(
                    id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    label_source TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)
            c.execute("INSERT OR IGNORE INTO metadata VALUES ('scope',?)", (scope,))

    def add_triples(self, triples: Iterable[Triple]):
        with self.connection(write=True) as c:
            for t in triples:
                c.executemany(
                    "INSERT OR IGNORE INTO entities(id,label) VALUES (?,?)",
                    [(t.subject, t.subject), (t.object, t.object)],
                )
                c.execute("INSERT OR IGNORE INTO triples VALUES (?,?,?,?)", (t.id, *t.key))

    def add_examples(self, examples: Iterable[Example]):
        with self.connection(write=True) as c:
            for ex in examples:
                old = c.execute("SELECT split FROM questions WHERE id=?", (ex.id,)).fetchone()
                if old and old["split"] != ex.split:
                    raise ValueError(f"Question {ex.id} occurs in different splits")
                c.execute(
                    "INSERT OR REPLACE INTO questions VALUES (?,?,?)",
                    (ex.id, ex.model_dump_json(), ex.split),
                )
                for t in ex.triples:
                    c.executemany(
                        "INSERT OR IGNORE INTO entities(id,label) VALUES (?,?)",
                        [(t.subject, t.subject), (t.object, t.object)],
                    )
                    c.execute("INSERT OR IGNORE INTO triples VALUES (?,?,?,?)", (t.id, *t.key))

    def question(self, question_id: str) -> Example:
        with self.connection() as c:
            row = c.execute("SELECT payload FROM questions WHERE id=?", (question_id,)).fetchone()
        if row is None:
            raise TraceError("question_not_found", f"Unknown question: {question_id}", 404)
        return Example.model_validate_json(row["payload"])

    def resolve(self, value: str) -> str | None:
        with self.connection() as c:
            exact = c.execute("SELECT id FROM entities WHERE id=?", (value,)).fetchone()
            if exact:
                return exact["id"]
            rows = c.execute(
                "SELECT id FROM entities WHERE label=? COLLATE NOCASE LIMIT 2", (value,)
            ).fetchall()
        if len(rows) == 1:
            return rows[0]["id"]
        if len(rows) > 1:
            raise TraceError(
                "ambiguous_entity", "Multiple entities have this label; supply the exact graph ID"
            )
        return None

    def entity(self, entity_id: str) -> dict:
        with self.connection() as c:
            row = c.execute("SELECT * FROM entities WHERE id=?", (entity_id,)).fetchone()
        return {
            "id": entity_id,
            "label": row["label"] if row else entity_id,
            "source_urls": json.loads(row["urls"]) if row else [],
        }

    def entities(self, entity_ids: list[str]) -> list[dict]:
        found = {}
        with self.connection() as c:
            for start in range(0, len(entity_ids), 400):
                batch = entity_ids[start : start + 400]
                placeholders = ",".join("?" for _ in batch)
                for row in c.execute(f"SELECT * FROM entities WHERE id IN ({placeholders})", batch):
                    found[row["id"]] = row
        return [
            {
                "id": key,
                "label": found[key]["label"] if key in found else key,
                "source_urls": json.loads(found[key]["urls"]) if key in found else [],
            }
            for key in entity_ids
        ]

    def scope(self) -> str:
        with self.connection() as c:
            return c.execute("SELECT value FROM metadata WHERE key='scope'").fetchone()[0]

    def find_entities(self, query: str, limit: int = 20) -> list[dict]:
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with self.connection() as c:
            rows = c.execute(
                "SELECT id,label FROM entities WHERE id=? OR label LIKE ? ESCAPE '\\' ORDER BY id LIMIT ?",
                (query, escaped + "%", limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def incident(self, anchors: Iterable[str], limit: int | None = None) -> list[Triple]:
        anchors = sorted(set(anchors))
        rows: dict[str, Triple] = {}
        with self.connection() as c:
            for anchor in anchors:
                # Query each endpoint separately to use both indexes.
                for col in ("subject", "object"):
                    sql = (
                        f"SELECT id,subject,predicate,object FROM triples WHERE {col}=? ORDER BY id"
                    )
                    args: tuple = (anchor,)
                    if limit is not None:
                        sql += " LIMIT ?"
                        args += (limit + 1,)
                    for r in c.execute(sql, args):
                        rows[r["id"]] = Triple(
                            subject=r["subject"], predicate=r["predicate"], object=r["object"]
                        )
                        if limit is not None and len(rows) > limit:
                            raise TraceError(
                                "graph_limit",
                                f"Anchor neighborhood exceeds {limit} triples; increase TRACE_CANDIDATE_MAX_TRIPLES",
                                422,
                            )
        return sorted(rows.values(), key=lambda t: t.key)

    def all_triples(self, limit: int) -> list[Triple]:
        """Read the supplied KG; raise if its size exceeds the retrieval limit."""
        if limit < 1:
            raise ValueError("Triple limit must be positive")
        with self.connection() as c:
            rows = c.execute(
                "SELECT subject,predicate,object FROM triples ORDER BY subject,predicate,object LIMIT ?",
                (limit + 1,),
            ).fetchall()
        if len(rows) > limit:
            raise TraceError(
                "candidate_limit",
                "Supplied KG exceeds the retrieval limit; increase TRACE_CANDIDATE_MAX_TRIPLES, "
                "use an annotated question, or explicitly request a bounded candidate neighborhood",
                422,
            )
        return [
            Triple(subject=r["subject"], predicate=r["predicate"], object=r["object"]) for r in rows
        ]

    def neighborhood(self, topic: str, hops: int, limit: int) -> list[Triple]:
        frontier = {topic}
        seen = {topic}
        result = {}
        for _ in range(hops):
            next_frontier = set()
            for t in self.incident(frontier, limit):
                result[t.id] = t
                next_frontier.update((t.subject, t.object))
                if len(result) > limit:
                    raise TraceError(
                        "candidate_limit",
                        "Candidate neighborhood exceeds configured limit; use an annotated question or smaller radius",
                        422,
                    )
            frontier = next_frontier - seen
            seen.update(next_frontier)
            if not frontier:
                break
        return sorted(result.values(), key=lambda t: t.key)

    def predicate_labels(self) -> dict[str, str]:
        with self.connection() as c:
            return {r["id"]: r["label"] for r in c.execute("SELECT id,label FROM predicates")}

    def predicate_label_sources(self) -> dict[str, str]:
        """Return label provenance for cache upgrades."""
        with self.connection() as c:
            return {
                r["id"]: r["label_source"]
                for r in c.execute("SELECT id,label_source FROM predicates")
            }

    def predicates(self) -> list[str]:
        with self.connection() as c:
            return [
                r[0] for r in c.execute("SELECT DISTINCT predicate FROM triples ORDER BY predicate")
            ]

    def save_predicate_label(self, predicate: str, label: str, source: str):
        with self.connection(write=True) as c:
            c.execute(
                "INSERT OR REPLACE INTO predicates VALUES (?,?,?)", (predicate, label, source)
            )

    def stats(self) -> dict:
        with self.connection() as c:
            return {
                "entities": c.execute("SELECT count(*) FROM entities").fetchone()[0],
                "triples": c.execute("SELECT count(*) FROM triples").fetchone()[0],
                "questions": c.execute("SELECT count(*) FROM questions").fetchone()[0],
                "scope": c.execute("SELECT value FROM metadata WHERE key='scope'").fetchone()[0],
            }

    def fingerprint(self) -> str:
        """Hash a consistent snapshot, independent of SQLite page layout and WAL state.

        Canonicalize JSON object keys; retain topic and provenance list order.
        """
        digest = hashlib.sha256(b"trace-graph-fingerprint-v1\n")
        tables = (
            ("entities", "id,label,urls", "id", {2}),
            ("triples", "id,subject,predicate,object", "id", set()),
            ("questions", "id,payload,split", "id", {1}),
            ("predicates", "id,label,label_source", "id", set()),
            ("metadata", "key,value", "key", set()),
        )
        with self.connection() as c:
            c.execute("BEGIN")
            for table, columns, order, json_columns in tables:
                digest.update((table + "\n").encode("utf-8"))
                for row in c.execute(f"SELECT {columns} FROM {table} ORDER BY {order}"):
                    values = [
                        json.loads(value) if i in json_columns else value
                        for i, value in enumerate(row)
                    ]
                    encoded = json.dumps(
                        values, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    )
                    digest.update((encoded + "\n").encode("utf-8"))
        return digest.hexdigest()
