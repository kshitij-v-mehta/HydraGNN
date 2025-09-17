import sqlite3
import sys
import os, glob, time, pickle
import os.path

from mpi4py import MPI
import torch


class DB:
    def __init__(self, db_name):
        self.conn = None
        self.cur = None

        self.set_type_codes = {'trainset': 0, 'valset': 1, 'testset': 2}

        self.create_sqlite(db_name)

    def create_sqlite(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.cur = self.conn.cursor()
        self.cur.execute("""
                         CREATE TABLE IF NOT EXISTS graph_data (
                         id INTEGER primary key AUTOINCREMENT,
                         set_type INTEGER NOT NULL,
                         original_pyg BLOB NOT NULL,
                         transformed_pyg BLOB)""")
        self.conn.commit()

    def add(self, set_type, blob):
        type_code = self.set_type_codes[set_type]
        self.cur.execute("INSERT or REPLACE INTO graph_data (set_type, original_pyg) VALUES (?, ?)",
                         (type_code, blob))
        self.conn.commit()

    def __del__(self):
        self.conn.commit()
        self.conn.close()
