import sqlite3
import sys
import os, glob, time, pickle
import os.path

from mpi4py import MPI
import torch


class DB:
    def __init__(self, db_name, cache_size: int = 1000):
        self.db_name = db_name
        self.conn = None
        self.cur = None

        # Separate cursors to read and udpate the table
        self.read_cursor = None
        self.update_cursor = None

        # A write cache used during creation and initial population of the db
        self.create_cache = []
        self.cache_size = cache_size

        # A cache to hold rows while reading data in
        self.read_cache = []
        self.read_cache_size = 1000

        # Count of how many transactions are run before commit must be called
        self.pending_commits = 0
        self.max_pending_commits = 1

        self.set_type_codes = {'trainset': 0, 'valset': 1, 'testset': 2}

        self._connect()

    def _connect(self):
        self.conn = sqlite3.connect(self.db_name)
        self.cur = self.conn.cursor()
        self.read_cursor = self.conn.cursor()
        self.update_cursor = self.conn.cursor()

    def create_tables(self):
        self.cur.execute("""
                         CREATE TABLE IF NOT EXISTS graph_data (
                         id INTEGER primary key AUTOINCREMENT,
                         set_type INTEGER NOT NULL,
                         original_pyg BLOB NOT NULL,
                         transformed_pyg BLOB)""")
        self.conn.commit()

    def add(self, set_type, blob):
        type_code = self.set_type_codes[set_type]

        self.create_cache.append((type_code, sqlite3.Binary(blob)))
        if len(self.create_cache) > self.cache_size:
            self._flush_caches()

    def get_unprocessed(self):
        if not self.read_cache:
            self.read_cursor.execute('SELECT * FROM graph_data where transformed_pyg is NULL')
            self.read_cache = self.read_cursor.fetchmany(self.read_cache_size)
            if not self.read_cache:
                return None
        return self.read_cache.pop(0)

    def update_pyg_transformed(self, rowid, pyg_transformed):
        self.update_cursor.execute("UPDATE graph_data set transformed_pyg = ? where rowid = ?",
                                   (sqlite3.Binary(pyg_transformed), rowid))
        self.pending_commits += 1

        if self.pending_commits >= self.max_pending_commits:
            self.conn.commit()
            self.pending_commits = 0

    def _flush_caches(self):
        if len(self.create_cache) > 0:
            self.cur.executemany("INSERT or REPLACE INTO graph_data (set_type, original_pyg) VALUES (?, ?)",
                                 self.create_cache)
        self.conn.commit()
        self.create_cache = []

    def close(self):
        self._flush_caches()
        self.conn.close()
