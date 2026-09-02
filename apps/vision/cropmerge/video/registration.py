from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RegistrationResult:
    success: bool
    transform: np.ndarray | None  # 3x3 homography or None
    n_matches: int
    n_inliers: int
    confidence: float
    method: str = "orb_homography"


def estimate_homography(
    prev_bgr: np.ndarray,
    curr_bgr: np.ndarray,
    cfg: dict | None = None,
) -> RegistrationResult:
    """Sparse ORB + RANSAC homography. Fail closed — never fabricate."""
    rcfg = (cfg or {}).get("registration", {})
    max_corners = int(rcfg.get("max_corners", 400))
    ratio = float(rcfg.get("match_ratio", 0.75))
    min_inliers = int(rcfg.get("min_inliers", 12))
    reproj = float(rcfg.get("ransac_reproj_threshold", 3.0))

    g0 = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=max_corners)
    k0, d0 = orb.detectAndCompute(g0, None)
    k1, d1 = orb.detectAndCompute(g1, None)
    if d0 is None or d1 is None or len(k0) < 8 or len(k1) < 8:
        return RegistrationResult(False, None, 0, 0, 0.0)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(d0, d1, k=2)
    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    n_matches = len(good)
    if n_matches < min_inliers:
        return RegistrationResult(False, None, n_matches, 0, 0.0)

    pts0 = np.float32([k0[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts1 = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, reproj)
    if H is None or mask is None:
        return RegistrationResult(False, None, n_matches, 0, 0.0)
    n_inliers = int(mask.sum())
    if n_inliers < min_inliers:
        return RegistrationResult(False, None, n_matches, n_inliers, 0.0)
    conf = float(np.clip(n_inliers / max(n_matches, 1), 0.0, 1.0))
    # Reject degenerate scales
    det = float(np.linalg.det(H[:2, :2]))
    if det < 0.2 or det > 5.0:
        return RegistrationResult(False, H, n_matches, n_inliers, conf * 0.2)
    return RegistrationResult(True, H, n_matches, n_inliers, conf)


def warp_mask(mask: np.ndarray, H: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    return cv2.warpPerspective(mask.astype(np.uint8), H, (w, h), flags=cv2.INTER_NEAREST)
