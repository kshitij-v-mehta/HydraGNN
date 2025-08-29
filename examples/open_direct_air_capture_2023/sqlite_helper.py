import sqlite3


class DB:
    def __init__(self, db_name):
        self.conn = None
        self.cur = None

        self.create_sqlite(db_name)

    def create_sqlite(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.cur = self.conn.cursor()
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS graph_batches (
              path_id TEXT primary key,
              payload BLOB
            )
            """)
        self.conn.commit()

    def write(self, path_id, blob):
        self.cur.execute("INSERT or REPLACE INTO graph_batches (path_id, payload) VALUES (?, ?)", (path_id, blob))
        self.conn.commit()

    def get_all_filenames(self):
        self.cur.execute("SELECT path_id FROM graph_batches")
        names_tuples = self.cur.fetchall()

        # Convert to a flat list
        filenames = [name[0] for name in names_tuples]
        return filenames

    def __del__(self):
        self.conn.commit()
        self.conn.close()
