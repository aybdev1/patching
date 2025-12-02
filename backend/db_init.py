# db_init.py
import sqlite3




import sqlite3

def init_db(db_path="patch_manager.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS hosts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hostname TEXT UNIQUE,
            last_seen TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS patches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kb TEXT UNIQUE,
            title TEXT,
            severity TEXT,
            released TIMESTAMP,
            description TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS host_patches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            host_id INTEGER,
            patch_id INTEGER,
            installed BOOLEAN,
            detected_on TIMESTAMP,
            priority_score REAL,
            FOREIGN KEY(host_id) REFERENCES hosts(id),
            FOREIGN KEY(patch_id) REFERENCES patches(id)
        )
    """)
    conn.commit()
    conn.close()
    print("DB initialized at", db_path)






def init_db222(path="patch_manager.db"):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS hosts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        hostname TEXT UNIQUE,
        last_seen TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS patches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        kb TEXT,
        title TEXT,
        severity TEXT,
        released DATE,
        description TEXT
    );
    CREATE TABLE IF NOT EXISTS host_patches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        host_id INTEGER,
        patch_id INTEGER,
        installed BOOLEAN,
        installed_on DATE,
        detected_on DATE,
        priority_score REAL,
        FOREIGN KEY(host_id) REFERENCES hosts(id),
        FOREIGN KEY(patch_id) REFERENCES patches(id)
    );
    """)
    conn.commit()
    conn.close()
    print("DB initialized at", path)

if __name__ == "__main__":
    init_db()
