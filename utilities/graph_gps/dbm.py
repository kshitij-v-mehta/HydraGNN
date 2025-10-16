import sqlite3
from queue import Queue


_conn: sqlite3.Connection = None
_cursor: sqlite3.Cursor = None
_cache = []


def _create_tables():
    global _conn, _cursor

    _cursor.execute("""
                    CREATE TABLE IF NOT EXISTS graph_data
                    (id INTEGER primary key AUTOINCREMENT,
                     set_type INTEGER NOT NULL,
                     pyg_obj BLOB NOT NULL,
                     status INTEGER NOT NULL DEFAULT 0
                    )""")
    _conn.commit()


def _open_connection(db_path: str):
    global _conn, _cursor

    _conn = sqlite3.connect(db_path)
    _cursor = _conn.cursor()

    _create_tables()


def write_to_db(db_path: str, set_type: int, pyg: bytes, cache_size: int):
    global _conn, _cursor, _cache

    if _conn is None:
        _open_connection(db_path)

    _cache.append( (set_type, sqlite3.Binary(pyg)) )
    if len(_cache) >= cache_size:
        _flush_cache()


def _flush_cache():
    global _conn, _cursor, _cache
    _cursor.executemany("INSERT or REPLACE INTO graph_data (set_type, pyg_obj) VALUES (?, ?)", _cache)
    _conn.commit()
    _cache = []


def close_db():
    global _conn
    _flush_cache()
    _conn.close()
