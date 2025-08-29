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

    def get_all(self, data_type):
        self.cur.execute("select payload from graph_batches where path_id like ?", (f"{data_type}%",))
        rows = self.cur.fetchall()
        blobs = [row[0] for row in rows]
        return blobs

    def remove_all(self, data_type):
        # Remove all rows where path_id starts with data_type

        self.cur.execute("DELETE from graph_batches where path_id like ?", (f"{data_type}%",))
        self.conn.commit()


    def __del__(self):
        self.conn.commit()
        self.conn.close()

