import { onBeforeUnmount, onMounted, watch, type Ref } from 'vue'

export type ViewportPlaybackPauseOptions = {
  /** Pause when visible ratio drops below this (0–1). Default 0.3. */
  threshold?: number
  onPause: () => void
  isPlaying: () => boolean
}

/** Pause active replay when the container is mostly off-screen (IntersectionObserver). */
export function useViewportPlaybackPause(
  target: Ref<HTMLElement | null>,
  options: ViewportPlaybackPauseOptions,
) {
  const threshold = options.threshold ?? 0.3
  let observer: IntersectionObserver | null = null

  onMounted(() => {
    if (typeof IntersectionObserver === 'undefined') return

    observer = new IntersectionObserver(
      (entries) => {
        const entry = entries[0]
        if (!entry) return
        if (document.fullscreenElement) return
        if (options.isPlaying() && entry.intersectionRatio < threshold) {
          options.onPause()
        }
      },
      { threshold: [0, threshold, 0.5, 1] },
    )

    watch(
      target,
      (el, _, onCleanup) => {
        observer?.disconnect()
        if (el) observer?.observe(el)
        onCleanup(() => observer?.disconnect())
      },
      { immediate: true },
    )
  })

  onBeforeUnmount(() => {
    observer?.disconnect()
    observer = null
  })
}

/** Scroll the filmstrip horizontally without moving the page. */
export function scrollFilmstripThumb(
  strip: HTMLElement,
  thumb: HTMLElement,
  behavior: ScrollBehavior = 'smooth',
) {
  const targetLeft = thumb.offsetLeft - strip.clientWidth / 2 + thumb.clientWidth / 2
  strip.scrollTo({ left: Math.max(0, targetLeft), behavior })
}

export function isInsideFrameReview(target: EventTarget | null): boolean {
  return Boolean((target as HTMLElement | null)?.closest?.('.frame-review'))
}
