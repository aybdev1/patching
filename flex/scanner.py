import os
import json
from typing import List, Dict

def read_version_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

def scan_targets(targets_root) -> List[Dict]:
    """
    Scans directories inside targets_root. Each target is a folder with:
      - version.txt (optional)
      - component files
    Returns list of targets with their metadata.
    """
    results = []
    if not os.path.exists(targets_root):
        return results
    for name in os.listdir(targets_root):
        target_path = os.path.join(targets_root, name)
        if not os.path.isdir(target_path):
            continue
        version = read_version_file(os.path.join(target_path, "version.txt"))
        fingerprint = compute_fingerprint(target_path)
        results.append({
            "name": name,
            "path": target_path,
            "version": version,
            "fingerprint": fingerprint
        })
    return results

def compute_fingerprint(target_path):
    # Simple fingerprint: list of files and sizes (can be extended)
    items = []
    for root, _, files in os.walk(target_path):
        for f in files:
            p = os.path.join(root, f)
            try:
                items.append((os.path.relpath(p, target_path).replace("\\","/"), os.path.getsize(p)))
            except Exception:
                continue
    # deterministic sort
    items.sort()
    return items

def index_patch_source(patch_source_dir):
    """
    Indexes patches stored as ZIP folders or directories.
    Each patch must contain a meta.json with:
      { "id": "patch-1", "component": "target1",
        "from_version": "1.0", "to_version": "1.1",
        "priority": 5, "severity": "high",
        "description": "fixes X" }
    """
    import zipfile
    patches = []
    if not os.path.exists(patch_source_dir):
        os.makedirs(patch_source_dir)
        return patches
    for entry in os.listdir(patch_source_dir):
        full = os.path.join(patch_source_dir, entry)
        meta = None
        if os.path.isdir(full):
            meta_path = os.path.join(full, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta['path'] = full
        elif entry.lower().endswith(".zip"):
            # inspect zip
            try:
                with zipfile.ZipFile(full, 'r') as z:
                    if "meta.json" in z.namelist():
                        content = z.read("meta.json").decode("utf-8")
                        meta = json.loads(content)
                        meta['path'] = full
            except Exception:
                continue
        if meta:
            # ensure id
            if 'id' not in meta:
                meta['id'] = os.path.splitext(entry)[0]
            patches.append({'id': meta['id'], 'meta': meta, **meta})
    return patches
