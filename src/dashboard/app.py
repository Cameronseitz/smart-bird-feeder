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
        }
    )


@app.route("/api/health")
def health():
    return jsonify(
        {"status": "ok", "timestamp": datetime.utcnow().isoformat(), "total_visits": db.get_stats()["total_visits"]}
    )


if __name__ == "__main__":
    logger.info(f"Starting dashboard {config.DASHBOARD_HOST}:{config.DASHBOARD_PORT}")
    app.run(host=config.DASHBOARD_HOST, port=config.DASHBOARD_PORT, debug=config.DASHBOARD_DEBUG, threaded=True)
