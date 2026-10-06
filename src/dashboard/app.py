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


@app.template_filter("short_time")
def short_time(value):
    """Full ISO/SQLite timestamp -> '9:29 AM' (hour + minute only)."""
    try:
        if value is None:
            return ""
        s = str(value).strip()
        if not s:
            return ""
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            dt = datetime.strptime(s.split(".")[0], "%Y-%m-%d %H:%M:%S")
        return f"{dt.hour % 12 or 12}:{dt.minute:02d} {'AM' if dt.hour < 12 else 'PM'}"
    except Exception:
        return str(value)


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
    try:
        species_list = db.distinct_species()
    except Exception:
        species_list = [s["species"] for s in db.get_stats()["species_frequency"]]
    return render_template(
        "gallery.html", visits=visits, species_list=species_list, current_species=species, page=page
    )


@app.route("/analytics")
def analytics_page():
    species = request.args.get("species", "all")
    stats = db.get_stats(species_filter=species if species != "all" else None)
    try:
        species_list = db.distinct_species()
    except Exception:
        species_list = [s["species"] for s in db.get_stats()["species_frequency"]]
    return render_template("analytics.html", stats=stats, species_list=species_list, current_species=species)


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


@app.route("/api/visits/<int:visit_id>", methods=["GET"])
def api_visit_detail(visit_id):
    visit = db.get_visit_by_id(visit_id)
    if not visit:
        return jsonify({"error": "not found"}), 404
    return jsonify(visit)


def _safe_asset_path(rel_path):
    """Resolve a DB-stored relative asset path, ensuring it stays inside assets/. Returns Path or None."""
    try:
        if not rel_path:
            return None
        p = (BASE_DIR / rel_path).resolve()
        assets_root = (BASE_DIR / "assets").resolve()
        try:
            p.relative_to(assets_root)
        except ValueError:
            return None
        return p
    except Exception:
        return None


def _refresh_latest_after_delete():
    """Point assets/latest.jpg at the next most-recent best frame, or remove it if none remain."""
    try:
        import shutil

        best_dir = BASE_DIR / "assets" / "best"
        latest = BASE_DIR / "assets" / "latest.jpg"
        newest = None
        if best_dir.exists():
            files = sorted(
                (p for p in best_dir.glob("*.jpg") if p.is_file()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            newest = files[0] if files else None
        if newest:
            shutil.copy(str(newest), str(latest))
            return newest.name
        else:
            try:
                if latest.exists():
                    latest.unlink()
            except Exception:
                pass
            return None
    except Exception as e:
        logger.error(f"latest.jpg refresh failed: {e}")
        return None


@app.route("/api/visits/<int:visit_id>", methods=["DELETE", "POST"])
def api_visit_delete(visit_id):
    # POST fallback for plain forms: require _method=delete or action=delete
    if request.method == "POST":
        marker = (request.args.get("_method") or request.args.get("action") or "").lower()
        body_marker = ""
        try:
            body = request.get_json(force=True, silent=True) or {}
            body_marker = str(body.get("_method") or body.get("action") or "").lower()
        except Exception:
            body = {}
        form_marker = str((request.form.get("_method") or request.form.get("action") or "")).lower()
        if marker not in ("delete", "destroy") and body_marker not in ("delete", "destroy") and form_marker not in ("delete", "destroy"):
            return jsonify({"error": "use DELETE method (or POST with _method=delete)"}), 405
    visit = db.get_visit_by_id(visit_id)
    if not visit:
        return jsonify({"error": "not found"}), 404
    deleted_files = []
    for key in ("best_image_path", "image_path"):
        p = _safe_asset_path(visit.get(key))
        if p and p.exists() and p.is_file():
            try:
                p.unlink()
                deleted_files.append(str(p.relative_to(BASE_DIR)).replace("\\", "/"))
            except Exception as e:
                logger.error(f"Failed to delete image {p}: {e}")
    db.delete_visit(visit_id)
    live_refreshed = _refresh_latest_after_delete()
    logger.info(f"Visit {visit_id} deleted files={deleted_files} live={live_refreshed}")
    return jsonify({"ok": True, "deleted_id": visit_id, "deleted_files": deleted_files, "live_frame": live_refreshed})


@app.route("/api/stats")
def api_stats():
    species = request.args.get("species", "all")
    return jsonify(db.get_stats(species_filter=species if species != "all" else None))


@app.route("/api/gallery")
def api_gallery():
    species = request.args.get("species", "all")
    limit = int(request.args.get("limit", 24))
    offset = int(request.args.get("offset", 0))
    return jsonify(db.get_gallery(species_filter=species, limit=limit, offset=offset))


@app.route("/visit/<int:visit_id>")
def visit_page(visit_id):
    from flask import abort

    visit = db.get_visit_by_id(visit_id)
    if not visit:
        abort(404)
    candidates = []
    try:
        session_id = visit.get("session_id") or ""
        # Filenames are <timestamp>_<session_id>.jpg so substring match is safe
        # (session_ids are generated as visit_<epoch>_<n>, no path separators).
        if session_id and "/" not in session_id and "\\" not in session_id:
            cap_dir = BASE_DIR / "assets" / "captures"
            if cap_dir.exists():
                files = sorted(cap_dir.glob(f"*{session_id}*.jpg"))
                candidates = [f"assets/captures/{p.name}" for p in files if p.is_file()]
    except Exception as e:
        logger.error(f"Candidate listing failed for visit {visit_id}: {e}")
    return render_template("visit.html", visit=visit, candidates=candidates)


@app.route("/assets/<path:filename>")
def serve_assets(filename):    return send_from_directory(str(BASE_DIR / "assets"), filename)


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


_LOG_UNITS = {
    "bird-feeder": "bird-feeder.service",
    "bird-dashboard": "bird-dashboard.service",
}


@app.route("/api/logs")
def api_logs():
    import shutil
    import subprocess

    unit_key = (request.args.get("unit") or "bird-feeder").strip()
    try:
        lines = int(request.args.get("lines", 200))
    except (TypeError, ValueError):
        lines = 200
    lines = max(50, min(1000, lines))
    if unit_key not in _LOG_UNITS:
        return jsonify({"ok": False, "error": f"unknown unit '{unit_key}'", "units": sorted(_LOG_UNITS)}), 400
    service = _LOG_UNITS[unit_key]
    journalctl = shutil.which("journalctl")
    if not journalctl:
        return jsonify(
            {
                "ok": False,
                "unit": unit_key,
                "service": service,
                "error": "journalctl not available on this host",
                "hint": "Run on the Pi, or check `journalctl -u bird-feeder` / `journalctl -u bird-dashboard` via SSH.",
            }
        ), 200
    try:
        proc = subprocess.run(
            [journalctl, "--no-pager", "-n", str(lines), "-u", service],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "unit": unit_key, "service": service, "error": "journalctl timed out after 5s"}), 504
    except Exception as e:
        logger.error(f"journalctl failed: {e}")
        return jsonify({"ok": False, "unit": unit_key, "service": service, "error": "failed to read logs"}), 500
    if proc.returncode != 0:
        return jsonify(
            {
                "ok": False,
                "unit": unit_key,
                "service": service,
                "error": (proc.stderr or "journalctl returned no output").strip()[-500:],
                "hint": f"Try SSH: journalctl -u {service} --no-pager | tail -n {lines}",
            }
        ), 200
    return jsonify({"ok": True, "unit": unit_key, "service": service, "lines": lines, "output": proc.stdout[-20000:]})


@app.route("/api/health")
def health():
    return jsonify(
        {"status": "ok", "timestamp": datetime.utcnow().isoformat(), "total_visits": db.get_stats()["total_visits"]}
    )


if __name__ == "__main__":
    logger.info(f"Starting dashboard {config.DASHBOARD_HOST}:{config.DASHBOARD_PORT}")
    app.run(host=config.DASHBOARD_HOST, port=config.DASHBOARD_PORT, debug=config.DASHBOARD_DEBUG, threaded=True)
