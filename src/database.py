"""
SQLite persistence for visits

Tables:
- visits: id, session_id, timestamp, species, raw_species, confidence, is_confident,
         image_path (best frame), bbox_json, sharpness, inference_ms, top_k_json, duration_sec
- settings: key/value for thresholds etc (optional)
"""
import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

try:
    import config
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config

logger = logging.getLogger(__name__)

CREATE_VISITS_TABLE = """
CREATE TABLE IF NOT EXISTS visits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    species TEXT,
    raw_species TEXT,
    confidence REAL,
    is_confident BOOLEAN,
    image_path TEXT,
    best_image_path TEXT,
    bbox_json TEXT,
    sharpness REAL,
    top_k_json TEXT,
    inference_ms REAL,
    duration_sec REAL,
    num_candidates INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_SETTINGS_TABLE = """
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

class BirdDatabase:
    def __init__(self, db_path=config.DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        cur = self.conn.cursor()
        cur.execute(CREATE_VISITS_TABLE)
        cur.execute(CREATE_SETTINGS_TABLE)
        self.conn.commit()
        logger.info(f"DB initialized at {self.db_path}")

    def add_visit(
        self,
        session,
        best_candidate,
        classification_result: Dict,
        best_image_path: str,
        image_path: str = None,
    ) -> int:
        """
        session: VisitSession
        best_candidate: CandidateFrame
        classification_result: from SpeciesClassifier.classify()
        best_image_path: path to saved best frame
        """
        try:
            cur = self.conn.cursor()
            bbox_json = json.dumps(best_candidate.detection) if best_candidate else "{}"
            top_k_json = json.dumps(classification_result.get("top_k", []))
            sharpness = float(best_candidate.sharpness) if best_candidate else 0.0
            inference_ms = float(classification_result.get("inference_ms", 0))
            is_conf = 1 if classification_result.get("is_confident") else 0

            cur.execute("""
                INSERT INTO visits
                (session_id, timestamp, species, raw_species, confidence, is_confident,
                 image_path, best_image_path, bbox_json, sharpness, top_k_json,
                 inference_ms, duration_sec, num_candidates)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.session_id,
                datetime.fromtimestamp(session.start_time).isoformat(),
                classification_result.get("species","unknown"),
                classification_result.get("raw_species","unknown"),
                float(classification_result.get("confidence",0)),
                is_conf,
                image_path or best_image_path,
                best_image_path,
                bbox_json,
                sharpness,
                top_k_json,
                inference_ms,
                float(session.duration()),
                len(session.candidates)
            ))
            self.conn.commit()
            row_id = cur.lastrowid
            logger.info(f"Visit logged id={row_id} species={classification_result.get('species')} conf={classification_result.get('confidence'):.2%} sess={session.session_id}")
            return row_id
        except Exception as e:
            logger.error(f"Failed to add visit: {e}")
            self.conn.rollback()
            return -1

    def get_recent_visits(self, limit=config.MAX_RECENT_VISITS) -> List[Dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM visits ORDER BY created_at DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        return [dict(r) for r in rows]

    def get_visit_by_id(self, visit_id: int) -> Optional[Dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM visits WHERE id=?", (visit_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def get_stats(self) -> Dict:
        cur = self.conn.cursor()
        # total visits
        cur.execute("SELECT COUNT(*) as total FROM visits")
        total = cur.fetchone()["total"]

        # species frequency
        cur.execute("""
            SELECT species, COUNT(*) as count, AVG(confidence) as avg_conf
            FROM visits WHERE is_confident=1
            GROUP BY species ORDER BY count DESC
        """)
        species_freq = [dict(r) for r in cur.fetchall()]

        # hourly distribution
        cur.execute("""
            SELECT CAST(strftime('%H', timestamp) AS INTEGER) as hour, COUNT(*) as count
            FROM visits GROUP BY hour ORDER BY hour
        """)
        hourly = [dict(r) for r in cur.fetchall()]
        # fill missing hours 0-23
        hourly_map = {r["hour"]: r["count"] for r in hourly}
        full_hourly = [{"hour": h, "count": hourly_map.get(h,0)} for h in range(24)]

        # daily last 7 days
        cur.execute("""
            SELECT date(timestamp) as day, COUNT(*) as count
            FROM visits WHERE timestamp >= date('now','-7 days')
            GROUP BY day ORDER BY day
        """)
        daily = [dict(r) for r in cur.fetchall()]

        # unknown rate
        cur.execute("SELECT COUNT(*) as unknown_cnt FROM visits WHERE species='unknown' OR is_confident=0")
        unknown_cnt = cur.fetchone()["unknown_cnt"]
        unknown_rate = unknown_cnt / total if total>0 else 0

        return {
            "total_visits": total,
            "species_frequency": species_freq,
            "hourly_distribution": full_hourly,
            "daily_last_7": daily,
            "unknown_count": unknown_cnt,
            "unknown_rate": unknown_rate
        }

    def get_gallery(self, species_filter=None, limit=100, offset=0) -> List[Dict]:
        cur = self.conn.cursor()
        if species_filter and species_filter != "all":
            cur.execute("SELECT * FROM visits WHERE species=? ORDER BY created_at DESC LIMIT ? OFFSET ?", (species_filter, limit, offset))
        else:
            cur.execute("SELECT * FROM visits ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset))
        rows = cur.fetchall()
        return [dict(r) for r in rows]

    def delete_visit(self, visit_id):
        cur = self.conn.cursor()
        cur.execute("DELETE FROM visits WHERE id=?", (visit_id,))
        self.conn.commit()

    def close(self):
        try:
            self.conn.close()
        except:
            pass
