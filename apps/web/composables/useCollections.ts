/**
 * ZIP photo-collection upload/analyze API client. Mirrors the shape of
 * apps/vision/api/main.py's /vision/collections routes exactly (see that
 * file for the authoritative field list) -- these are plain FastAPI dict
 * responses, not yet promoted to packages/types, since the collection
 * feature is still new and evolving; keep this file as the single source
 * of truth for its response shapes on the frontend side.
 */
import { useVisionApi } from '~/composables/useVisionApi'

export type CollectionImageSummary = {
  id: string
  relativeGroup: string | null
  status: 'valid' | 'duplicate' | 'failed'
  duplicateOf?: string | null
  error?: string | null
  sizeBytes?: number
}

export type CollectionUploadResponse = {
  collectionId: string
  createdAt: string
  summary: {
    validImages: number
    duplicateImages: number
    failedImages: number
    rejectedMembers: number
  }
  images: CollectionImageSummary[]
}

export type CollectionImageResult = {
  imageId: string
  relativeGroup: string | null
  status: 'analyzed' | 'failed'
  error: string | null
  fieldDetected: boolean | null
  cropCoverage: number | null
  bareSoilFraction: number | null
}

export type GroupStatistics = {
  group: string
  imageCount: number
  cropCoverageMedian: number | null
  cropCoverageIqr: [number, number] | null
  bareSoilMedian: number | null
  bareSoilIqr: [number, number] | null
}

export type CollectionAnalysisResult = {
  collectionId: string
  analyzedAt: string
  imageResults: CollectionImageResult[]
  groupStatistics: GroupStatistics[]
  summary: {
    totalValidImages: number
    analyzedCount: number
    failedCount: number
  }
}

export type CollectionImageDetail = {
  runId: string
  disclaimer?: string
  field?: { detected?: boolean; meanCropCoverage?: number | null; meanBareSoil?: number | null }
  inspectionZones?: unknown[]
  summary?: { headline?: string; keyFindings?: string[] } | null
  artifactUrls: Record<string, string>
}

export function useCollections() {
  const { request } = useVisionApi()

  async function uploadCollection(file: File, signal?: AbortSignal): Promise<CollectionUploadResponse> {
    const body = new FormData()
    body.append('file', file)
    return request<CollectionUploadResponse>('/vision/collections', { method: 'POST', body, signal })
  }

  async function analyzeCollection(
    collectionId: string,
    options?: { segmentationBackend?: string; dinoBackend?: string },
    signal?: AbortSignal,
  ): Promise<CollectionAnalysisResult> {
    return request<CollectionAnalysisResult>(`/vision/collections/${collectionId}/analyze`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        segmentationBackend: options?.segmentationBackend,
        dinoBackend: options?.dinoBackend,
      }),
      signal,
    })
  }

  async function getCollectionAnalysis(collectionId: string): Promise<CollectionAnalysisResult> {
    return request<CollectionAnalysisResult>(`/vision/collections/${collectionId}/analysis`)
  }

  async function getCollectionImage(collectionId: string, imageId: string): Promise<CollectionImageDetail> {
    return request<CollectionImageDetail>(`/vision/collections/${collectionId}/images/${imageId}`)
  }

  return { uploadCollection, analyzeCollection, getCollectionAnalysis, getCollectionImage }
}
