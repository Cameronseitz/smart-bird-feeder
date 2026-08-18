"""
Local Web Dashboard (Flask)

Per blueprint Section 5:
- Tailwind CSS + Chart.js
- Live feed, gallery view (species/timestamp/confidence), analytics, settings
- Runs on Pi 3 via Flask 0.0.0.0:5000
"""
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, jsonify, request, send_from_directory, Response
from flask_cors import CORS

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

import config
from src.database import BirdDatabase

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder=str(BASE_DIR / "src/dashboard/templates"), static_folder=str(BASE_DIR / "src/dashboard/static"))
CORS(app)

# DB singleton
db = BirdDatabase()

# For live feed MJPEG - optional shared frame holder
# The main pipeline can update this via file or shared memory
# Simplest: if file assets/latest.jpg exists, serve it, else low-power streaming endpoint attempts camera capture

@app.route("/")
def index():
    stats = db.get_stats()
    recent = db.get_recent_visits(limit=10)
    return render_template("index.html", stats=stats, recent=recent)

@app.route("/gallery")
def gallery():
    species = request.args.get("species","all")
    page = int(request.args.get("page", 0))
    limit = 24
    visits = db.get_gallery(species_filter=species, limit=limit, offset=page*limit)
    stats = db.get_stats()
    # unique species for filter
    species_list = [s["species"] for s in stats["species_frequency"]]
    return render_template("gallery.html", visits=visits, species_list=species_list, current_species=species, page=page)

@app.route("/analytics")
def analytics_page():
    stats = db.get_stats()
    return render_template("analytics.html", stats=stats)

@app.route("/settings")
def settings_page():
    return render_template("settings.html", config={
        "detection_threshold": config.DETECTION_THRESHOLD,
        "species_threshold": config.SPECIES_CONFIDENCE_THRESHOLD,
        "grace_period": config.GRACE_PERIOD_SEC,
        "motion_threshold": config.MOTION_THRESHOLD,
    })

@app.route("/api/visits")
def api_visits():
    limit = int(request.args.get("limit", config.MAX_RECENT_VISITS))
    recent = db.get_recent_visits(limit=limit)
    return jsonify(recent)

@app.route("/api/visits/<int:visit_id>")
def api_visit_detail(visit_id):
    visit = db.get_visit_by_id(visit_id)
    if not visit:
        return jsonify({"error":"not found"}), 404
    return jsonify(visit)

@app.route("/api/stats")
def api_stats():
    stats = db.get_stats()
    return jsonify(stats)

@app.route("/api/gallery")
def api_gallery():
    species = request.args.get("species","all")
    limit = int(request.args.get("limit",24))
    offset = int(request.args.get("offset",0))
    data = db.get_gallery(species_filter=species, limit=limit, offset=offset)
    return jsonify(data)

@app.route("/assets/<path:filename>")
def serve_assets(filename):
    assets_root = BASE_DIR / "assets"
    return send_from_directory(str(assets_root), filename)

@app.route("/data/captures/<path:filename>")
def serve_captures(filename):
    # legacy path
    cap_dir = BASE_DIR / "assets" / "captures"
    return send_from_directory(str(cap_dir), filename)

@app.route("/api/live/frame")
def api_live_frame():
    # Serve latest captured best frame if exists
    best_dir = BASE_DIR / "assets" / "best"
    latest = None
    if best_dir.exists():
        files = sorted(best_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            latest = files[0]
    if latest:
        return send_from_directory(str(best_dir), latest.name)
    # try assets/captures/latest.jpg if pipeline writes it
    latest_path = BASE_DIR / "assets" / "latest.jpg"
    if latest_path.exists():
        return send_from_directory(str(BASE_DIR / "assets"), "latest.jpg")
    return jsonify({"error":"no frame yet"}), 404

@app.route("/api/live/mjpeg")
def mjpeg_stream():
    # Optional live MJPEG if camera available and dashboard wants real-time
    # Note: This will attempt to open camera, may conflict with main pipeline if both open.
    # Prefer using saved latest.jpg in production; mjpeg here is for testing.
    def generate():
        try:
            from src.camera import CameraInterface
            import cv2
            cam = CameraInterface(width=640, height=480, fps=5)
            for frame in cam.frames():
                ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ret:
                    continue
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        except Exception as e:
            logger.error(f"MJPEG stream error: {e}")
            # Yield a placeholder
            blank = b''
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + blank + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/api/settings", methods=["GET","POST"])
def api_settings():
    global db
    if request.method == "GET":
        return jsonify({
            "detection_threshold": config.DETECTION_THRESHOLD,
            "species_confidence_threshold": config.SPECIES_CONFIDENCE_THRESHOLD,
            "grace_period": config.GRACE_PERIOD_SEC,
            "motion_threshold": config.MOTION_THRESHOLD
        })
    else:
        data = request.get_json()
        # In real implementation save to DB or config file; here just echo
        # For persistence we can store in settings table
        # Update in memory (won't survive restart unless written to file)
        # TODO: persist to file
        logger.info(f"Settings update requested: {data}")
        return jsonify({"status":"ok", "received": data})

@app.route("/api/health")
def health():
    return jsonify({"status":"ok", "timestamp": datetime.utcnow().isoformat(), "total_visits": db.get_stats()["total_visits"]})

if __name__ == "__main__":
    logger.info(f"Starting dashboard on {config.DASHBOARD_HOST}:{config.DASHBOARD_PORT}")
    app.run(host=config.DASHBOARD_HOST, port=config.DASHBOARD_PORT, debug=config.DASHBOARD_DEBUG, threaded=True)
