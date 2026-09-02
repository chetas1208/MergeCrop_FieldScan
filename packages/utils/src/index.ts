export function formatSec(t: number): string {
  const m = Math.floor(t / 60)
  const s = t - m * 60
  return `${String(m).padStart(2, '0')}:${s.toFixed(1).padStart(4, '0')}`
}

export function clamp01(n: number): number {
  return Math.min(1, Math.max(0, n))
}
