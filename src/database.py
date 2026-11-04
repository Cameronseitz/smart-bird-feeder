"""
SQLite persistence for visits
"""
import sys
import json
import logging
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
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


def _common(label) -> str:
    try:
        from src.dashboard.names import common_name

        return common_name(label)
    except Exception:
        return str(label or "unknown").strip() or "unknown"


def _normalize_visit_row(row: Dict) -> Dict:
    try:
        if "species" in row:
            row["species"] = _common(row.get("species"))
        # Keep raw_species key for backwards compat but mirror common name
        # so templates/APIs never leak scientific names.
        if "raw_species" in row:
            row["raw_species"] = row.get("species", "unknown")
        tk = row.get("top_k_json")
        if isinstance(tk, str) and tk:
            try:
                data = json.loads(tk)
                if isinstance(data, list):
                    normed = []
                    for item in data:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            normed.append([_common(item[0]), item[1]])
                        else:
                            normed.append(item)
                    row["top_k_json"] = json.dumps(normed)
            except Exception:
                pass
    except Exception:
        pass
    return row


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
        return [_normalize_visit_row(dict(r)) for r in rows]

    def get_visit_by_id(self, visit_id: int) -> Optional[Dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM visits WHERE id=?", (visit_id,))
        row = cur.fetchone()
        return _normalize_visit_row(dict(row)) if row else None

    def _all_visits_legacy_aware(self):
        cur = self.conn.cursor()
        cur.execute("SELECT species, raw_species, confidence, is_confident, timestamp FROM visits")
        return [dict(r) for r in cur.fetchall()]

    def distinct_species(self) -> List[str]:
        cur = self.conn.cursor()
        cur.execute("SELECT species, raw_species FROM visits WHERE is_confident=1")
        seen = {}
        for r in cur.fetchall():
            name = _common(r["species"])
            if name == "unknown":
                name = _common(r["raw_species"])
            if name and name != "unknown" and name not in seen:
                seen[name] = True
        return sorted(seen.keys())

    def get_stats(self, species_filter=None) -> Dict:
        filt = _common(species_filter) if species_filter and species_filter != "all" else None
        cur = self.conn.cursor()
        if not filt:
            # total visits
            cur.execute("SELECT COUNT(*) as total FROM visits")
            total = cur.fetchone()["total"]

            # species frequency (merge legacy scientific labels via common-name)
            cur.execute("""
                SELECT species, raw_species, COUNT(*) as count, AVG(confidence) as avg_conf
                FROM visits WHERE is_confident=1
                GROUP BY species, raw_species ORDER BY count DESC
            """)
            merged: Dict[str, Dict] = {}
            conf_sum: Dict[str, float] = {}
            conf_n: Dict[str, int] = {}
            for r in cur.fetchall():
                name = _common(r["species"])
                if name == "unknown":
                    name = _common(r["raw_species"])
                if name == "unknown":
                    continue
                merged[name] = merged.get(name, 0) + r["count"]
            # avg conf per normalized species
            cur.execute("SELECT species, raw_species, confidence FROM visits WHERE is_confident=1")
            for r in cur.fetchall():
                name = _common(r["species"])
                if name == "unknown":
                    name = _common(r["raw_species"])
                if name == "unknown":
                    continue
                conf_sum[name] = conf_sum.get(name, 0.0) + float(r["confidence"] or 0)
                conf_n[name] = conf_n.get(name, 0) + 1
            species_freq = [
                {"species": k, "count": v, "avg_conf": (conf_sum.get(k, 0) / conf_n.get(k, 1))}
                for k, v in sorted(merged.items(), key=lambda kv: kv[1], reverse=True)
            ]

            # hourly distribution
            cur.execute("""
                SELECT CAST(strftime('%H', timestamp) AS INTEGER) as hour, COUNT(*) as count
                FROM visits GROUP BY hour ORDER BY hour
            """)
            hourly = [dict(r) for r in cur.fetchall()]
            hourly_map = {r["hour"]: r["count"] for r in hourly}
            full_hourly = [{"hour": h, "count": hourly_map.get(h, 0)} for h in range(24)]

            # daily last 7 days
            cur.execute("""
                SELECT date(timestamp) as day, COUNT(*) as count
                FROM visits WHERE timestamp >= date('now','-7 days')
                GROUP BY day ORDER BY day
            """)
            daily = [dict(r) for r in cur.fetchall()]

            # daily last 30 days (parallel to last-7, same sparse shape)
            cur.execute("""
                SELECT date(timestamp) as day, COUNT(*) as count
                FROM visits WHERE timestamp >= date('now','-30 days')
                GROUP BY day ORDER BY day
            """)
            daily_30 = [dict(r) for r in cur.fetchall()]

            # unknown rate
            cur.execute("SELECT COUNT(*) as unknown_cnt FROM visits WHERE species='unknown' OR is_confident=0")
            unknown_cnt = cur.fetchone()["unknown_cnt"]
            unknown_rate = unknown_cnt / total if total > 0 else 0

            return {
                "total_visits": total,
                "species_frequency": species_freq,
                "hourly_distribution": full_hourly,
                "daily_last_7": daily,
                "daily_last_30": daily_30,
                "unknown_count": unknown_cnt,
                "unknown_rate": unknown_rate,
            }

        # Species-filtered path: legacy-aware Python aggregation so old
        # scientific rows still match their common name.
        rows = self._all_visits_legacy_aware()
        if filt == "unknown":
            match = [r for r in rows if _common(r["species"]) == "unknown"]
        else:
            match = [
                r
                for r in rows
                if _common(r["species"]) == filt or _common(r["raw_species"]) == filt
            ]
        total = len(match)
        confident = [r for r in match if r["is_confident"] == 1 and _common(r["species"]) != "unknown"]
        avg_conf = sum(float(r["confidence"] or 0) for r in confident) / len(confident) if confident else 0
        species_freq = [{"species": filt, "count": len(confident), "avg_conf": avg_conf}] if filt != "unknown" else []

        hourly_map = {h: 0 for h in range(24)}
        for r in match:
            try:
                h = int(str(r["timestamp"])[11:13])
                if 0 <= h <= 23:
                    hourly_map[h] += 1
            except Exception:
                pass
        full_hourly = [{"hour": h, "count": hourly_map[h]} for h in range(24)]

        from datetime import date, timedelta

        days = [(date.today() - timedelta(days=i)).isoformat() for i in range(6, -1, -1)]
        day_counts = {d: 0 for d in days}
        for r in match:
            try:
                d = str(r["timestamp"])[:10]
                if d in day_counts:
                    day_counts[d] += 1
            except Exception:
                pass
        daily = [{"day": d, "count": day_counts[d]} for d in days if day_counts[d] > 0]

        days_30 = [(date.today() - timedelta(days=i)).isoformat() for i in range(29, -1, -1)]
        day_counts_30 = {d: 0 for d in days_30}
        for r in match:
            try:
                d = str(r["timestamp"])[:10]
                if d in day_counts_30:
                    day_counts_30[d] += 1
            except Exception:
                pass
        daily_30 = [{"day": d, "count": day_counts_30[d]} for d in days_30 if day_counts_30[d] > 0]

        unknown_cnt = sum(1 for r in match if _common(r["species"]) == "unknown" or r["is_confident"] == 0)
        return {
            "total_visits": total,
            "species_frequency": species_freq,
            "hourly_distribution": full_hourly,
            "daily_last_7": daily,
            "daily_last_30": daily_30,
            "unknown_count": unknown_cnt,
            "unknown_rate": (unknown_cnt / total if total > 0 else 0),
        }

    def get_gallery(self, species_filter=None, limit=100, offset=0) -> List[Dict]:
        cur = self.conn.cursor()
        if not species_filter or species_filter == "all":
            cur.execute("SELECT * FROM visits ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset))
            rows = cur.fetchall()
            return [_normalize_visit_row(dict(r)) for r in rows]
        filt = _common(species_filter)
        if filt == "unknown":
            cur.execute(
                "SELECT * FROM visits WHERE species='unknown' OR is_confident=0 "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
            rows = cur.fetchall()
            return [_normalize_visit_row(dict(r)) for r in rows]
        # Legacy-aware: exact SQL match first, then Python merge with
        # raw_species-normalized rows (covers old scientific labels).
        cur.execute("SELECT * FROM visits ORDER BY created_at DESC")
        out = []
        for r in cur.fetchall():
            d = dict(r)
            if _common(d.get("species")) == filt or _common(d.get("raw_species")) == filt:
                out.append(_normalize_visit_row(d))
        return out[offset : offset + limit]

    def delete_visit(self, visit_id):
        cur = self.conn.cursor()
        cur.execute("DELETE FROM visits WHERE id=?", (visit_id,))
        self.conn.commit()

    def close(self):
        try:
            self.conn.close()
        except:
            pass
