"""
Dashboard Flask - live + gallery + analytics
"""
import sys
import logging
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

import config
from src.database import BirdDatabase

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder=str(BASE_DIR / "src/dashboard/templates"))
CORS(app)

db = BirdDatabase()


@app.route("/")
def index():
    stats = db.get_stats()
    recent = db.get_recent_visits(limit=10)
    return render_template("index.html", stats=stats, recent=recent)


@app.route("/gallery")
def gallery():
    species = request.args.get("species", "all")
    page = int(request.args.get("page", 0))
    limit = 24
    visits = db.get_gallery(species_filter=species, limit=limit, offset=page * limit)
    stats = db.get_stats()
    species_list = [s["species"] for s in stats["species_frequency"]]
    return render_template(
        "gallery.html", visits=visits, species_list=species_list, current_species=species, page=page
    )


@app.route("/analytics")
def analytics_page():
    stats = db.get_stats()
    return render_template("analytics.html", stats=stats)


@app.route("/settings")
def settings_page():
    return render_template(
        "settings.html",
        config={
            "detection_threshold": config.DETECTION_THRESHOLD,
            "species_threshold": config.SPECIES_CONFIDENCE_THRESHOLD,
            "grace_period": config.GRACE_PERIOD_SEC,
            "motion_threshold": config.MOTION_THRESHOLD,
            "camera_exposure_value": config.CAMERA_EXPOSURE_VALUE,
            "camera_ae_metering_mode": config.CAMERA_AE_METERING_MODE,
            "camera_ae_exposure_mode": config.CAMERA_AE_EXPOSURE_MODE,
            "camera_af_mode": config.CAMERA_AF_MODE,
            "camera_lens_position": config.CAMERA_LENS_POSITION,
            "sharpness_blur_threshold": config.SHARPNESS_BLUR_THRESHOLD,
            "sharpness_good_ref": config.SHARPNESS_GOOD_REF,
            "max_bbox_fraction": config.MAX_BBOX_FRACTION,
        },
    )


@app.route("/api/visits")
def api_visits():
    limit = int(request.args.get("limit", config.MAX_RECENT_VISITS))
    return jsonify(db.get_recent_visits(limit=limit))


@app.route("/api/visits/<int:visit_id>")
def api_visit_detail(visit_id):
    visit = db.get_visit_by_id(visit_id)
    if not visit:
        return jsonify({"error": "not found"}), 404
    return jsonify(visit)


@app.route("/api/stats")
def api_stats():
    return jsonify(db.get_stats())


@app.route("/api/gallery")
def api_gallery():
    species = request.args.get("species", "all")
    limit = int(request.args.get("limit", 24))
    offset = int(request.args.get("offset", 0))
    return jsonify(db.get_gallery(species_filter=species, limit=limit, offset=offset))


@app.route("/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory(str(BASE_DIR / "assets"), filename)


@app.route("/api/live/frame")
def api_live_frame():
    best_dir = BASE_DIR / "assets" / "best"
    if best_dir.exists():
        files = sorted(best_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            return send_from_directory(str(best_dir), files[0].name)
    latest = BASE_DIR / "assets" / "latest.jpg"
    if latest.exists():
        return send_from_directory(str(BASE_DIR / "assets"), "latest.jpg")
    return jsonify({"error": "no frame yet"}), 404


@app.route("/api/settings")
def api_settings():
    return jsonify(
        {
            "detection_threshold": config.DETECTION_THRESHOLD,
            "species_confidence_threshold": config.SPECIES_CONFIDENCE_THRESHOLD,
            "grace_period": config.GRACE_PERIOD_SEC,
            "motion_threshold": config.MOTION_THRESHOLD,
            "camera_exposure_value": config.CAMERA_EXPOSURE_VALUE,
            "camera_ae_metering_mode": config.CAMERA_AE_METERING_MODE,
            "camera_ae_exposure_mode": config.CAMERA_AE_EXPOSURE_MODE,
            "camera_af_mode": config.CAMERA_AF_MODE,
            "camera_lens_position": config.CAMERA_LENS_POSITION,
            "sharpness_blur_threshold": config.SHARPNESS_BLUR_THRESHOLD,
            "sharpness_good_ref": config.SHARPNESS_GOOD_REF,
            "max_bbox_fraction": config.MAX_BBOX_FRACTION,
            "overrides": getattr(config, "OVERRIDES_LOADED", {}),
        }
    )


def _coerce_settings(payload: dict):
    """Map lower/UPPER keys -> canonical config keys with range clamping. Returns (clean, errors)."""
    import math

    def num(v, lo, hi, is_int=False, allow_null=False):
        if v is None or (isinstance(v, str) and v.strip() in ("", "null", "none", "auto")):
            if allow_null:
                return None, None
            return None, "required"
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None, "not a number"
        if isinstance(f, float) and (math.isnan(f) or math.isinf(f)):
            return None, "not a number"
        f = max(lo, min(hi, f))
        return (int(round(f)) if is_int else f), None

    def choice(v, allowed):
        try:
            i = int(float(v))
        except (TypeError, ValueError):
            return None, "not a number"
        if i not in allowed:
            return None, f"must be one of {sorted(allowed)}"
        return i, None

    aliases = {
        "detection_threshold": "DETECTION_THRESHOLD",
        "DETECTION_THRESHOLD": "DETECTION_THRESHOLD",
        "species_confidence_threshold": "SPECIES_CONFIDENCE_THRESHOLD",
        "species_threshold": "SPECIES_CONFIDENCE_THRESHOLD",
        "SPECIES_CONFIDENCE_THRESHOLD": "SPECIES_CONFIDENCE_THRESHOLD",
        "grace_period": "GRACE_PERIOD_SEC",
        "GRACE_PERIOD_SEC": "GRACE_PERIOD_SEC",
        "motion_threshold": "MOTION_THRESHOLD",
        "MOTION_THRESHOLD": "MOTION_THRESHOLD",
        "camera_exposure_value": "CAMERA_EXPOSURE_VALUE",
        "CAMERA_EXPOSURE_VALUE": "CAMERA_EXPOSURE_VALUE",
        "camera_ae_metering_mode": "CAMERA_AE_METERING_MODE",
        "CAMERA_AE_METERING_MODE": "CAMERA_AE_METERING_MODE",
        "camera_ae_exposure_mode": "CAMERA_AE_EXPOSURE_MODE",
        "CAMERA_AE_EXPOSURE_MODE": "CAMERA_AE_EXPOSURE_MODE",
        "camera_af_mode": "CAMERA_AF_MODE",
        "CAMERA_AF_MODE": "CAMERA_AF_MODE",
        "camera_lens_position": "CAMERA_LENS_POSITION",
        "CAMERA_LENS_POSITION": "CAMERA_LENS_POSITION",
        "sharpness_blur_threshold": "SHARPNESS_BLUR_THRESHOLD",
        "SHARPNESS_BLUR_THRESHOLD": "SHARPNESS_BLUR_THRESHOLD",
        "sharpness_good_ref": "SHARPNESS_GOOD_REF",
        "SHARPNESS_GOOD_REF": "SHARPNESS_GOOD_REF",
        "max_bbox_fraction": "MAX_BBOX_FRACTION",
        "MAX_BBOX_FRACTION": "MAX_BBOX_FRACTION",
    }
    clean, errors = {}, {}
    for raw_k, v in (payload or {}).items():
        canon = aliases.get(raw_k)
        if not canon or canon not in config.WRITABLE_KEYS:
            continue
        if canon in ("DETECTION_THRESHOLD", "SPECIES_CONFIDENCE_THRESHOLD"):
            cv, err = num(v, 0.30, 0.95)
        elif canon == "GRACE_PERIOD_SEC":
            cv, err = num(v, 1.0, 10.0)
        elif canon == "MOTION_THRESHOLD":
            cv, err = num(v, 5, 100, is_int=True)
        elif canon == "CAMERA_EXPOSURE_VALUE":
            cv, err = num(v, -2.0, 3.0)
        elif canon in ("CAMERA_AE_METERING_MODE", "CAMERA_AE_EXPOSURE_MODE", "CAMERA_AF_MODE"):
            cv, err = choice(v, {0, 1, 2})
        elif canon == "CAMERA_LENS_POSITION":
            cv, err = num(v, 0.0, 15.0, allow_null=True)
        elif canon == "SHARPNESS_BLUR_THRESHOLD":
            cv, err = num(v, 20.0, 1000.0)
        elif canon == "SHARPNESS_GOOD_REF":
            cv, err = num(v, 200.0, 3000.0)
        elif canon == "MAX_BBOX_FRACTION":
            cv, err = num(v, 0.25, 0.95)
        else:
            continue
        if err:
            errors[canon] = err
        else:
            clean[canon] = cv
    return clean, errors


@app.route("/api/settings", methods=["POST"])
def api_settings_save():
    import json

    try:
        payload = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"error": "invalid JSON"}), 400
    clean, errors = _coerce_settings(payload)
    if errors and not clean:
        return jsonify({"error": "validation failed", "errors": errors}), 400
    try:
        current = {}
        if config.OVERRIDES_PATH.exists():
            try:
                current = json.loads(config.OVERRIDES_PATH.read_text(encoding="utf-8") or "{}")
            except Exception:
                current = {}
        current.update(clean)
        # Drop keys matching compiled-in defaults? No - keep explicit so UI round-trips.
        tmp = config.OVERRIDES_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(config.OVERRIDES_PATH)
        for k, v in clean.items():
            setattr(config, k, v)
        logger.info(f"Settings saved {clean}")
        return jsonify(
            {
                "saved": clean,
                "errors": errors,
                "note": "Saved. Restart bird-feeder (or reboot Pi) for camera/pipeline to pick up.",
            }
        )
    except Exception as e:
        logger.error(f"Settings save failed: {e}")
        return jsonify({"error": str(e)}), 500


def _run_detached(cmd: str):
    import subprocess

    try:
        subprocess.Popen(
            ["bash", "-lc", f"sleep 2; {cmd}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True
    except Exception as e:
        logger.error(f"Detached run failed {cmd}: {e}")
        return False


@app.route("/api/system/restart-pipeline", methods=["POST"])
def api_restart_pipeline():
    ok = _run_detached("sudo -n systemctl restart bird-feeder")
    if ok:
        return jsonify({"ok": True, "action": "restart-pipeline"})
    return jsonify({"error": "failed to schedule restart (sudo -n systemctl?)"}), 500


@app.route("/api/system/reboot", methods=["POST"])
def api_reboot():
    ok = _run_detached("sudo -n reboot")
    if ok:
        return jsonify({"ok": True, "action": "reboot", "note": "Pi rebooting in ~2s; dashboard will drop."})
    return jsonify({"error": "failed to schedule reboot (sudo -n reboot?)"}), 500


@app.route("/api/health")
def health():
    return jsonify(
        {"status": "ok", "timestamp": datetime.utcnow().isoformat(), "total_visits": db.get_stats()["total_visits"]}
    )


if __name__ == "__main__":
    logger.info(f"Starting dashboard {config.DASHBOARD_HOST}:{config.DASHBOARD_PORT}")
    app.run(host=config.DASHBOARD_HOST, port=config.DASHBOARD_PORT, debug=config.DASHBOARD_DEBUG, threaded=True)
