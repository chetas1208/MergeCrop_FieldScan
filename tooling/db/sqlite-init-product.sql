PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS analyses (
  id             TEXT PRIMARY KEY,
  status         TEXT NOT NULL,
  created_at     TEXT NOT NULL,
  updated_at     TEXT NOT NULL,
  filename       TEXT NOT NULL,
  progress       REAL NOT NULL DEFAULT 0,
  message        TEXT,
  error          TEXT,
  run_id         TEXT,
  report_json    TEXT,
  artifacts_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analyses_status ON analyses (status);

CREATE TABLE IF NOT EXISTS schema_meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('version', '1');
