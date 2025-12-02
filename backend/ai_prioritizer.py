# ai_prioritizer.py
# Simple prioritizer: can be replaced by a trained sklearn model.
from datetime import datetime
import math

def heuristic_score(patch):
    """
    Compute a priority score 0..1 based on fields:
    patch is dict with keys: severity (Critical/Important/Moderate/Low),
    released (YYYY-MM-DD or None), exploit_known (bool), asset_criticality (1..5)
    """
    sev_map = {"Critical": 1.0, "Important": 0.8, "Moderate": 0.5, "Low": 0.2}
    sev = sev_map.get(patch.get("severity"), 0.4)
    # age multiplier: older patches (unapplied) higher priority
    released = patch.get("released")
    age_days = 0
    if released:
        try:
            released_dt = datetime.fromisoformat(released)
            age_days = (datetime.utcnow() - released_dt).days
        except Exception:
            age_days = 0
    age_factor = min(1.0, math.log1p(age_days + 1) / 5.0)  # saturates around some days
    exploit = 1.0 if patch.get("exploit_known") else 0.0
    asset = float(patch.get("asset_criticality", 3)) / 5.0
    # Weighted sum
    score = 0.5 * sev + 0.25 * asset + 0.15 * age_factor + 0.1 * exploit
    # clamp
    return max(0.0, min(1.0, score))

def prioritize(patch_list):
    for p in patch_list:
        p["priority_score"] = heuristic_score(p)
    return sorted(patch_list, key=lambda x: x["priority_score"], reverse=True)
