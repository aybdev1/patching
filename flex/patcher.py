import os
import shutil
import zipfile
import datetime
import json

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def backup_target(target_path):
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    backup_dir = os.path.join(os.getcwd(), "patches", f"backup_{os.path.basename(target_path)}_{ts}")
    ensure_dir(backup_dir)
    shutil.copytree(target_path, os.path.join(backup_dir, os.path.basename(target_path)))
    return backup_dir

def apply_patch_to_target(patch, target_path):
    """
    patch: dict returned from index_patch_source: contains 'path' and 'meta' dict
    target_path: path to target folder
    Steps:
      - Backup target
      - Extract patch contents and copy into target (overwrite)
      - Update version.txt if meta provides to_version
    """
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Target path {target_path} not found")
    backup = backup_target(target_path)
    patch_path = patch.get("path")
    if not patch_path or not os.path.exists(patch_path):
        raise FileNotFoundError("Patch package not found")
    # If patch is zip, extract to temp folder
    tmp_extract = os.path.join(os.getcwd(), "patches", "tmp_extract")
    if os.path.exists(tmp_extract):
        shutil.rmtree(tmp_extract)
    os.makedirs(tmp_extract, exist_ok=True)
    if os.path.isdir(patch_path):
        # copy contents
        for name in os.listdir(patch_path):
            if name == "meta.json":
                continue
            src = os.path.join(patch_path, name)
            dest = os.path.join(tmp_extract, name)
            if os.path.isdir(src):
                shutil.copytree(src, dest)
            else:
                shutil.copy2(src, dest)
    elif patch_path.lower().endswith(".zip"):
        with zipfile.ZipFile(patch_path, 'r') as z:
            # extract everything except meta.json (we can also include)
            for member in z.namelist():
                if member.endswith("/"):
                    continue
                if os.path.basename(member) == "meta.json":
                    continue
                z.extract(member, tmp_extract)
    # Now copy from tmp_extract into target_path, overwriting
    for root, dirs, files in os.walk(tmp_extract):
        rel = os.path.relpath(root, tmp_extract)
        target_root = os.path.join(target_path, rel) if rel != "." else target_path
        if not os.path.exists(target_root):
            os.makedirs(target_root, exist_ok=True)
        for f in files:
            srcf = os.path.join(root, f)
            destf = os.path.join(target_root, f)
            shutil.copy2(srcf, destf)
    # Update version.txt if provided
    meta = patch.get('meta', {})
    to_version = meta.get('to_version')
    if to_version:
        with open(os.path.join(target_path, "version.txt"), "w", encoding="utf-8") as vf:
            vf.write(str(to_version))
    # clean tmp
    shutil.rmtree(tmp_extract)
    return {"backup": backup, "target": target_path, "to_version": to_version}
