/**
 * Product database — SQLite by default, Postgres if DATABASE_URL is postgresql://
 */
import { existsSync, mkdirSync, readFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { DatabaseSync } from 'node:sqlite'
import type { AnalysisJob, ArtifactPaths, FieldTriageReport } from '@cropmerge/types'

type SqlDb = DatabaseSync

let _db: SqlDb | null = null
let _pg: any = null
let _mode: 'sqlite' | 'postgres' | null = null

function repoRootFromConfig(): string {
  const config = useRuntimeConfig()
  // uploadsDir = <root>/data/uploads → root is two levels up from uploads parent? 
  // Prefer explicit databaseDir
  const dbDir = (config.databaseDir as string) || ''
  if (dbDir) return resolve(dbDir, '..', '..')
  const uploads = config.uploadsDir as string
  if (uploads) return resolve(uploads, '..', '..')
  return resolve(process.cwd(), '../..')
}

function sqlitePath(): string {
  const config = useRuntimeConfig()
  const url = (config.databaseUrl as string) || process.env.DATABASE_URL || process.env.NUXT_DATABASE_URL || ''
  if (url.startsWith('sqlite:///')) {
    return resolve(url.replace(/^sqlite:\/\//, ''))
  }
  const dir = resolve((config.databaseDir as string) || join(repoRootFromConfig(), 'data', 'db'))
  mkdirSync(dir, { recursive: true })
  return join(dir, 'cropmerge.sqlite')
}

function initSqliteSchema(db: SqlDb) {
  const root = repoRootFromConfig()
  const sqlPath = resolve(root, 'tooling/db/sqlite-init-product.sql')
  if (existsSync(sqlPath)) {
    db.exec(readFileSync(sqlPath, 'utf-8'))
  } else {
    db.exec(`
      CREATE TABLE IF NOT EXISTS analyses (
        id TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        filename TEXT NOT NULL,
        progress REAL NOT NULL DEFAULT 0,
        message TEXT,
        error TEXT,
        run_id TEXT,
        report_json TEXT,
        artifacts_json TEXT
      );
      CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses (created_at DESC);
    `)
  }
}

function ensureDb(): void {
  if (_mode) return
  const url = (useRuntimeConfig().databaseUrl as string) || process.env.DATABASE_URL || ''
  if (url.startsWith('postgresql://') || url.startsWith('postgres://')) {
    // Lazy postgres — optional dependency
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const pg = require('pg')
      _pg = new pg.Pool({ connectionString: url })
      _mode = 'postgres'
      return
    } catch {
      console.warn('[db] pg not installed; falling back to sqlite')
    }
  }
  const path = sqlitePath()
  mkdirSync(dirname(path), { recursive: true })
  _db = new DatabaseSync(path)
  _db.exec('PRAGMA journal_mode = WAL; PRAGMA foreign_keys = ON;')
  initSqliteSchema(_db)
  _mode = 'sqlite'
}

function rowToJob(row: Record<string, unknown>): AnalysisJob {
  return {
    id: String(row.id),
    status: row.status as AnalysisJob['status'],
    createdAt: String(row.created_at),
    updatedAt: String(row.updated_at),
    filename: String(row.filename),
    progress: Number(row.progress ?? 0),
    message: (row.message as string) || undefined,
    error: (row.error as string) || undefined,
    report: row.report_json ? (JSON.parse(String(row.report_json)) as FieldTriageReport) : null,
    artifacts: row.artifacts_json
      ? (JSON.parse(String(row.artifacts_json)) as ArtifactPaths)
      : null,
  }
}

export function saveJob(job: AnalysisJob): void {
  ensureDb()
  const reportJson = job.report ? JSON.stringify(job.report) : null
  const artifactsJson = job.artifacts ? JSON.stringify(job.artifacts) : null
  const runId = job.report?.runId ?? null

  if (_mode === 'postgres' && _pg) {
    // fire-and-forget sync via deasync-less approach: use await in callers later
    // For parity, use nested wait with Atomics — better expose async API
    throw new Error('Use saveJobAsync for postgres')
  }

  _db!.prepare(`
    INSERT INTO analyses (
      id, status, created_at, updated_at, filename, progress, message, error, run_id, report_json, artifacts_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET
      status=excluded.status,
      updated_at=excluded.updated_at,
      progress=excluded.progress,
      message=excluded.message,
      error=excluded.error,
      run_id=excluded.run_id,
      report_json=excluded.report_json,
      artifacts_json=excluded.artifacts_json
  `).run(
    job.id,
    job.status,
    job.createdAt,
    job.updatedAt,
    job.filename,
    job.progress,
    job.message ?? null,
    job.error ?? null,
    runId,
    reportJson,
    artifactsJson,
  )
}

export function loadJob(id: string): AnalysisJob | null {
  ensureDb()
  const row = _db!.prepare('SELECT * FROM analyses WHERE id = ?').get(id) as
    | Record<string, unknown>
    | undefined
  if (!row) return null
  return rowToJob(row)
}

export function listJobs(limit = 50): AnalysisJob[] {
  ensureDb()
  const rows = _db!
    .prepare('SELECT * FROM analyses ORDER BY created_at DESC LIMIT ?')
    .all(limit) as Record<string, unknown>[]
  return rows.map(rowToJob)
}

export function dbHealth(): { ok: boolean; mode: string; path?: string } {
  try {
    ensureDb()
    if (_mode === 'sqlite') {
      const path = sqlitePath()
      _db!.prepare('SELECT 1').get()
      return { ok: true, mode: 'sqlite', path }
    }
    return { ok: true, mode: 'postgres' }
  } catch (e) {
    return { ok: false, mode: _mode || 'unknown' }
  }
}
