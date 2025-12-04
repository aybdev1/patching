from flask import Flask, render_template, jsonify, request
from scanner import scan_targets, index_patch_source
from db import init_db, get_targets, add_target, record_patch_application, get_patch_history
from patcher import apply_patch_to_target
from ai_engine import load_or_train_model, score_patch
import os

app = Flask(__name__)
app.config['PATCH_SOURCE_DIR'] = os.path.join(os.getcwd(), "patch_source")
app.config['TARGETS_DIR'] = os.path.join(os.getcwd(), "targets")

# Initialize DB and AI model
init_db()
model = load_or_train_model()

# Index available patches at startup
available_patches = index_patch_source(app.config['PATCH_SOURCE_DIR'])


@app.route("/")
def index():
    targets = get_targets()
    # Scan targets live to show current status
    scanned = scan_targets(app.config['TARGETS_DIR'])
    # Annotate patches with AI score
    patches_with_scores = []
    for p in available_patches:
        p_copy = p.copy()
        p_copy['ai_score'] = float(score_patch(model, p))
        patches_with_scores.append(p_copy)
    return render_template("index.html", targets=scanned, patches=patches_with_scores)


@app.route("/api/scan", methods=["GET"])
def api_scan():
    scanned = scan_targets(app.config['TARGETS_DIR'])
    return jsonify(scanned)


@app.route("/api/patches", methods=["GET"])
def api_patches():
    patches_with_scores = []
    for p in available_patches:
        p_copy = p.copy()
        p_copy['ai_score'] = float(score_patch(model, p))
        patches_with_scores.append(p_copy)
    return jsonify(patches_with_scores)


@app.route("/api/apply_patch", methods=["POST"])
def api_apply_patch():
    data = request.json
    patch_id = data.get("patch_id")
    target_name = data.get("target_name")
    if not patch_id or not target_name:
        return jsonify({"error": "patch_id and target_name required"}), 400
    patch = next((p for p in available_patches if p["id"] == patch_id), None)
    if patch is None:
        return jsonify({"error": "patch not found"}), 404
    # Apply
    try:
        result = apply_patch_to_target(patch, os.path.join(app.config['TARGETS_DIR'], target_name))
        record_patch_application(target_name, patch['id'], patch['meta'])
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/history/<target_name>", methods=["GET"])
def api_history(target_name):
    history = get_patch_history(target_name)
    return jsonify(history)


#error from the app and waiting for detection from the piupline
if __name__ == "__main__":
    app.run(debug=True)
