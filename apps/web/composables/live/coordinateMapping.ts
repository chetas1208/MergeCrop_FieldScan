export interface DisplayRect {
  x: number
  y: number
  w: number
  h: number
}

export interface NormRect {
  x: number
  y: number
  w: number
  h: number
}

/**
 * The video element uses object-fit: contain, so its intrinsic frame is
 * letterboxed inside the container. Overlay coordinates must be computed
 * against this letterboxed rect, not the raw container size, or boxes
 * drift out of alignment whenever the aspect ratios differ.
 */
export function computeLetterboxRect(
  videoWidth: number,
  videoHeight: number,
  containerWidth: number,
  containerHeight: number,
): DisplayRect {
  if (!videoWidth || !videoHeight || !containerWidth || !containerHeight) {
    return { x: 0, y: 0, w: containerWidth, h: containerHeight }
  }
  const videoRatio = videoWidth / videoHeight
  const containerRatio = containerWidth / containerHeight
  if (videoRatio > containerRatio) {
    const w = containerWidth
    const h = w / videoRatio
    return { x: 0, y: (containerHeight - h) / 2, w, h }
  }
  const h = containerHeight
  const w = h * videoRatio
  return { x: (containerWidth - w) / 2, y: 0, w, h }
}

/** Maps a 0-1 normalized box (InspectionZone.bboxNorm) into the letterboxed
 * display rect's pixel space. */
export function mapNormBoxToDisplay(box: NormRect, letterbox: DisplayRect): DisplayRect {
  return {
    x: letterbox.x + box.x * letterbox.w,
    y: letterbox.y + box.y * letterbox.h,
    w: box.w * letterbox.w,
    h: box.h * letterbox.h,
  }
}
