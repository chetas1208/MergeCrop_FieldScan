PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS vision_runs (
  run_id               TEXT PRIMARY KEY,
  created_at           TEXT NOT NULL,
  source_filename      TEXT,
  status               TEXT NOT NULL DEFAULT 'completed',
  segmentation_backend TEXT,
  dino_backend         TEXT,
  used_fallback        INTEGER,
  frames_sampled       INTEGER,
  frames_usable        INTEGER,
  zone_count           INTEGER,
  field_detected       INTEGER,
  total_runtime_sec    REAL,
  stage_latency_json   TEXT,
  results_path         TEXT,
  metrics_json         TEXT,
  report_json          TEXT
);

CREATE INDEX IF NOT EXISTS idx_vision_runs_created ON vision_runs (created_at DESC);

CREATE TABLE IF NOT EXISTS schema_meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('version', '1');

-- Live Drone mode (Docker Compose only). See docs/LIVE_MODE_LIMITATIONS.md.

CREATE TABLE IF NOT EXISTS live_sessions (
  id           TEXT PRIMARY KEY,
  created_at   TEXT NOT NULL,
  updated_at   TEXT NOT NULL,
  ended_at     TEXT,
  status       TEXT NOT NULL,
  stream_path  TEXT NOT NULL,
  sample_fps   REAL NOT NULL,
  simulated    INTEGER NOT NULL DEFAULT 0,
  whep_url     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_live_sessions_created ON live_sessions (created_at DESC);

CREATE TABLE IF NOT EXISTS live_telemetry (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id     TEXT NOT NULL,
  received_at    TEXT NOT NULL,
  timestamp_ms   INTEGER NOT NULL,
  latitude       REAL,
  longitude      REAL,
  altitude_m     REAL,
  heading_deg    REAL,
  gimbal_pitch_deg REAL,
  raw_json       TEXT
);

CREATE INDEX IF NOT EXISTS idx_live_telemetry_session_ts ON live_telemetry (session_id, timestamp_ms);

CREATE TABLE IF NOT EXISTS live_saved_areas (
  id                     TEXT PRIMARY KEY,
  session_id             TEXT NOT NULL,
  zone_json              TEXT NOT NULL,
  telemetry_availability TEXT NOT NULL,
  telemetry_json         TEXT,
  snapshot_path          TEXT,
  saved_at               TEXT NOT NULL,
  note                   TEXT
);

CREATE INDEX IF NOT EXISTS idx_live_saved_areas_session ON live_saved_areas (session_id, saved_at DESC);
