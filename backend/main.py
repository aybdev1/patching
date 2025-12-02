# main.py
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import sqlite3
from datetime import datetime
import os
from ai_prioritizer import prioritize, heuristic_score
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

import subprocess
from fastapi.responses import JSONResponse



from cpu import router as cpu_router

from vulnerability import router as vulnerability_router 


import subprocess
import shlex
import tempfile
import os
import uuid
from fastapi import HTTPException

from datetime import datetime, timedelta
import psutil
# Heure actuelle + 1 minute

app = FastAPI()

app.include_router(cpu_router)
app.include_router(vulnerability_router)

DB_PATH = os.path.join(os.path.dirname(__file__), "patch_manager.db")

class HostReport(BaseModel):
    hostname: str
    patches: list  # list of patch dicts



def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

@app.on_event("startup")
def startup():
    # initialize DB if missing
    if not os.path.exists(DB_PATH):
        from db_init import init_db
        init_db(DB_PATH)

@app.post("/api/scan")
async def ingest_scan(report: HostReport):
    conn = get_conn()
    cur = conn.cursor()
    # host upsert
    cur.execute("INSERT OR IGNORE INTO hosts (hostname, last_seen) VALUES (?,?)",
                (report.hostname, datetime.utcnow()))
    cur.execute("UPDATE hosts SET last_seen=? WHERE hostname=?",
                (datetime.utcnow(), report.hostname))
    conn.commit()
    # ensure patches
    for p in report.patches:
        kb = p.get("kb", "")
        title = p.get("title", "")
        severity = p.get("severity", "")
        released = p.get("released", None)
        description = p.get("description", "")
        # insert patch if not exists
        cur.execute("SELECT id FROM patches WHERE kb=?", (kb,))
        row = cur.fetchone()
        if not row:
            cur.execute("INSERT INTO patches (kb,title,severity,released,description) VALUES (?,?,?,?,?)",
                        (kb,title,severity,released,description))
            patch_id = cur.lastrowid
        else:
            patch_id = row["id"]
        # find host id
        cur.execute("SELECT id FROM hosts WHERE hostname=?", (report.hostname,))
        host_id = cur.fetchone()["id"]
        # insert host_patches
        cur.execute("""INSERT INTO host_patches (host_id,patch_id,installed,detected_on,priority_score)
                       VALUES (?,?,?,?,?)""",
                    (host_id, patch_id, False, datetime.utcnow(), 0.0))
    conn.commit()
    conn.close()
    return {"status":"ok"}


@app.get("/api/get_hostname")
def get_hostname():
    import socket
    return {"hostname": socket.gethostname()}


@app.get("/api/patches/{hostname}")
def list_patches(hostname: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM hosts WHERE hostname=?", (hostname,))
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Host not found")
    host_id = row["id"]
    cur.execute("""SELECT hp.id as hp_id, p.kb, p.title, p.severity, p.released, p.description, hp.installed
                   FROM host_patches hp JOIN patches p ON hp.patch_id=p.id
                   WHERE hp.host_id=?""", (host_id,))
    patches = []
    for r in cur.fetchall():
        p = dict(r)
        # extend with extra fields for prioritizer
        p_extra = {
            "kb": p["kb"],
            "title": p["title"],
            "severity": p["severity"],
            "released": p["released"],
            # placeholders - in a real system you'd enrich with known exploit data & asset criticality
            "exploit_known": False,
            "asset_criticality": 3
        }
        #print(heuristic_score(p_extra))
        p["priority_score"] = heuristic_score(p_extra)
        patches.append(p)
    # sort
    patches = sorted(patches, key=lambda x: x["priority_score"], reverse=True)
    conn.close()
    return {"host": hostname, "patches": patches}


@app.post("/api/run_scan")
def run_scan():
    ps_script = r"C:\Users\Adminn\Desktop\fax-app\codepatch\scripts\scan_updatesUI.ps1"
    
    try:
        # Run PowerShell script and capture output
        result = subprocess.run(
            ["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", ps_script],
            capture_output=True, text=True, check=True
        )
        return JSONResponse({
            "status": "success",
            "stdout": result.stdout,
            "stderr": result.stderr
        })
    except subprocess.CalledProcessError as e:
        return JSONResponse({
            "status": "error",
            "stdout": e.stdout,
            "stderr": e.stderr
        }, status_code=500)

 

@app.get("/api/cpu_threads")
def get_cpu_threads():
    try:
        cores_info = []
        # Get per-core CPU times as percentages
        for i, times in enumerate(psutil.cpu_times_percent(percpu=True)):
            cores_info.append({
                "core": i,
                "user": times.user,
                "system": times.system,
                "idle": times.idle
            })
        return {"cores": cores_info}

    except Exception as e:
        return {"error": str(e)}


@app.post("/api/trigger_install")
async def trigger_install(data: dict):
    hostname = data.get("hostname")
    kb = data.get("kb")
    if not hostname or not kb:
        raise HTTPException(400, "hostname and kb required")

    # Correct path to your install_update.ps1 script
    script_path = r"C:\Users\Adminn\Desktop\fax-app\codepatch\scripts\install_all_updates.ps1"

    # Build instruction with actual KB
    cmd = f"powershell.exe -ExecutionPolicy Bypass -File \"{script_path}\" -KB {kb}"

    return {"status":"ready", "instruction": f"Run this on host {hostname} as admin: {cmd}"}


@app.delete("/api/patches/{hostname}")
def delete_patches(hostname: str):
    conn = get_conn()
    cur = conn.cursor()
    # Find host ID
    cur.execute("SELECT id FROM hosts WHERE hostname=?", (hostname,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Host not found")
    host_id = row["id"]
    # Delete host_patches for that host
    cur.execute("DELETE FROM host_patches WHERE host_id=?", (host_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted", "host": hostname}










@app.post("/api/trigger_install2")
async def trigger_install2(data: dict):
    """
    data: { "hostname": "...", "kb": "KBxxxx" }
    For local-only demo, we return the command to run on the host.
    In production you'd use WinRM/PSRemoting to run the install script remotely.
    """
    hostname = data.get("hostname")
    kb = data.get("kb")
    if not hostname or not kb:
        raise HTTPException(400, "hostname and kb required")
    # For safety: do not execute remote commands from this demo.
    script = f"powershell.exe -ExecutionPolicy Bypass -File C:\\path\\to\\scripts\\install_update.ps1 -KB {kb}"
    return {"status":"ready", "instruction": f"Run this on host {hostname} as admin: {script}"}

# Serve static UI

@app.get("/api/hostname")
def get_hostname():
    import socket
    return {"hostname": socket.gethostname()}
    


app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

@app.get("/")
def root():
    return HTMLResponse(open(os.path.join(os.path.dirname(__file__), "static","index.html"), "r", encoding="utf-8").read())




 




