import sqlite3


class DB:
    def __init__(self, rank):
        self.conn = None
        self.cur = None
        self.rank = rank

        self.create_sqlite(rank)

    def create_sqlite(self, rank):
        self.conn = sqlite3.connect(f"odac23_{rank}.db")
        self.cur = self.conn.cursor()
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS graph_batches (
              path_id TEXT primary key,
              payload BLOB
            )
            """)
        self.conn.commit()

    def write(self, path_id, blob):
        self.cur.execute("INSERT INTO graph_batches (path_id, payload) VALUES (?, ?)", (path_id, blob))
        self.conn.commit()

    def get_all_filenames_from_db(self):
        self.cur.execute("SELECT path_id FROM graph_batches")
        names_tuples = self.cur.fetchall()

        # Convert to a flat list
        filenames = [name[0] for name in names_tuples]
        return filenames

    def __del__(self):
        self.conn.commit()
        self.conn.close()
