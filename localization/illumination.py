"""Bounded, neutral illumination correction; never changes image geometry."""
import cv2
import numpy as np


def compensate_shading(reference, current):
    """Fit a smooth multiplicative light field, preserving ink and piece edges.

    Median support exceeds one piece diameter in the canonical board. Local
    additions/removals cannot be explained as an illumination field. A common
    gain for B/G/R preserves side information; no per-pixel color substitution.
    """
    if reference.shape != current.shape:
        raise ValueError("Illumination images must have identical shapes")
    height, width = reference.shape[:2]
    ref = (cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if reference.ndim == 3 else reference).astype(np.float32)
    cur = (cv2.cvtColor(current, cv2.COLOR_BGR2GRAY) if current.ndim == 3 else current).astype(np.float32)
    ratio = np.log((cur+8)/(ref+8))
    # Resolution-independent support: ~80 canonical pixels, larger than a
    # physical piece. Smooth away compression/ink differences before fitting.
    small = cv2.resize(ratio, (max(8,width//4),max(8,height//4)), interpolation=cv2.INTER_AREA)
    # OpenCV's wide median is defined for uint8; quantization is <2% in gain.
    encoded = np.clip((small+2)*60,0,255).astype(np.uint8)
    field = cv2.medianBlur(encoded, 21).astype(np.float32)/60-2
    field = cv2.GaussianBlur(field, (0,0), 3)
    gain = np.exp(cv2.resize(field, (width,height), interpolation=cv2.INTER_LINEAR))
    bounded = np.clip(gain, .45, 1.8)
    divisor = bounded[...,None] if current.ndim == 3 else bounded
    corrected = np.clip((current.astype(np.float32)+8)/divisor-8,0,255).astype(np.uint8)
    return corrected, dict(illumination_gain_min=float(gain.min()),
                           illumination_gain_max=float(gain.max()))
