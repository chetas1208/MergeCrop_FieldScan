-- Runs on first Postgres container boot.
CREATE DATABASE cropmerge_vision;
GRANT ALL PRIVILEGES ON DATABASE cropmerge TO cropmerge;
GRANT ALL PRIVILEGES ON DATABASE cropmerge_vision TO cropmerge;

\c cropmerge

CREATE TABLE IF NOT EXISTS analyses (
  id            TEXT PRIMARY KEY,
  status        TEXT NOT NULL,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  filename      TEXT NOT NULL,
  progress      DOUBLE PRECISION NOT NULL DEFAULT 0,
  message       TEXT,
  error         TEXT,
  run_id        TEXT,
  report_json   JSONB,
  artifacts_json JSONB
);

CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analyses_status ON analyses (status);

\c cropmerge_vision

CREATE TABLE IF NOT EXISTS vision_runs (
  run_id              TEXT PRIMARY KEY,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  source_filename     TEXT,
  status              TEXT NOT NULL DEFAULT 'completed',
  segmentation_backend TEXT,
  dino_backend        TEXT,
  used_fallback       BOOLEAN,
  frames_sampled      INTEGER,
  frames_usable       INTEGER,
  zone_count          INTEGER,
  field_detected      BOOLEAN,
  total_runtime_sec   DOUBLE PRECISION,
  stage_latency_json  JSONB,
  results_path        TEXT,
  metrics_json        JSONB,
  report_json         JSONB
);

CREATE INDEX IF NOT EXISTS idx_vision_runs_created ON vision_runs (created_at DESC);
