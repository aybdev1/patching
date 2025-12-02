import sqlite3
import os
import json

DB_PATH = os.path.join(os.getcwd(), "patchmate.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS targets (
        id INTEGER PRIMARY KEY,
        name TEXT UNIQUE,
        path TEXT
    )
    ''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS patch_history (
        id INTEGER PRIMARY KEY,
        target_name TEXT,
        patch_id TEXT,
        patch_meta TEXT,
        applied_at TEXT
    )
    ''')
    conn.commit()
    conn.close()

def add_target(name, path):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT OR IGNORE INTO targets (name, path) VALUES (?, ?)', (name, path))
    conn.commit()
    conn.close()

def get_targets():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT name, path FROM targets')
    rows = c.fetchall()
    conn.close()
    return [{"name": r[0], "path": r[1]} for r in rows]

def record_patch_application(target_name, patch_id, patch_meta):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    import datetime
    c.execute('INSERT INTO patch_history (target_name, patch_id, patch_meta, applied_at) VALUES (?, ?, ?, ?)',
              (target_name, patch_id, json.dumps(patch_meta), datetime.datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def get_patch_history(target_name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT patch_id, patch_meta, applied_at FROM patch_history WHERE target_name=? ORDER BY applied_at DESC', (target_name,))
    rows = c.fetchall()
    conn.close()
    out = []
    for pid, meta, at in rows:
        try:
            meta_j = json.loads(meta)
        except Exception:
            meta_j = {"raw": meta}
        out.append({"patch_id": pid, "meta": meta_j, "applied_at": at})
    return out
