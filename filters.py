"""
filters.py — OpenCV image filter functions (Advanced Edition)
Based on: ml_image_preprocessing_class1.ipynb + new features

All functions:
  Input  : NumPy BGR array
  Output : NumPy BGR array
"""

import cv2
import numpy as np


# ── Original 6 filters ────────────────────────────────────────────────────────

def apply_blur(img: np.ndarray, ksize: int = 1) -> np.ndarray:
    if ksize <= 1:
        return img.copy()
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def apply_sharpness(img: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    if abs(alpha - 1.0) < 0.01:
        return img.copy()
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(img, alpha, blurred, 1.0 - alpha, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_brightness(img: np.ndarray, beta: int = 0) -> np.ndarray:
    if beta == 0:
        return img.copy()
    return cv2.convertScaleAbs(img, alpha=1.0, beta=beta)


def apply_contrast(img: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    if abs(alpha - 1.0) < 0.01:
        return img.copy()
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)


def apply_edge_detection(img: np.ndarray, enabled: bool = False,
                          thresh1: int = 100, thresh2: int = 200) -> np.ndarray:
    if not enabled:
        return img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, thresh1, thresh2)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def apply_grayscale(img: np.ndarray, enabled: bool = False) -> np.ndarray:
    # Notebook formula: L = R*299/1000 + G*587/1000 + B*114/1000
    if not enabled:
        return img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ── NEW: Noise Reduction ──────────────────────────────────────────────────────

def apply_denoise(img: np.ndarray, strength: int = 0) -> np.ndarray:
    """Non-local means denoising. strength=0 → off."""
    if strength == 0:
        return img.copy()
    return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)


# ── NEW: Color Filters ────────────────────────────────────────────────────────

def apply_color_filter(img: np.ndarray, style: str = "none") -> np.ndarray:
    if style == "none":
        return img.copy()

    if style == "invert":
        return cv2.bitwise_not(img)

    if style == "sepia":
        kernel = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189],
        ])
        return np.clip(cv2.transform(img, kernel), 0, 255).astype(np.uint8)

    if style == "warm":
        lut = np.arange(256, dtype=np.uint8)
        b, g, r = cv2.split(img)
        return cv2.merge([
            cv2.LUT(b, np.clip(lut * 0.85, 0, 255).astype(np.uint8)),
            cv2.LUT(g, np.clip(lut * 1.05, 0, 255).astype(np.uint8)),
            cv2.LUT(r, np.clip(lut * 1.15, 0, 255).astype(np.uint8)),
        ])

    if style == "cool":
        lut = np.arange(256, dtype=np.uint8)
        b, g, r = cv2.split(img)
        return cv2.merge([
            cv2.LUT(b, np.clip(lut * 1.15, 0, 255).astype(np.uint8)),
            cv2.LUT(g, np.clip(lut * 1.00, 0, 255).astype(np.uint8)),
            cv2.LUT(r, np.clip(lut * 0.85, 0, 255).astype(np.uint8)),
        ])

    return img.copy()


# ── NEW: Rotate ───────────────────────────────────────────────────────────────

def apply_rotate(img: np.ndarray, angle: int = 0) -> np.ndarray:
    if angle == 0:
        return img.copy()
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)


# ── NEW: Crop ─────────────────────────────────────────────────────────────────

def apply_crop(img: np.ndarray, x1_pct=0.0, y1_pct=0.0,
               x2_pct=100.0, y2_pct=100.0) -> np.ndarray:
    """Crop using percentage-based coordinates (0–100)."""
    h, w = img.shape[:2]
    x1, y1 = int(w * x1_pct / 100), int(h * y1_pct / 100)
    x2, y2 = int(w * x2_pct / 100), int(h * y2_pct / 100)
    if x2 <= x1 or y2 <= y1:
        return img.copy()
    return img[y1:y2, x1:x2].copy()


# ── NEW: Face Detection ───────────────────────────────────────────────────────

def apply_face_detection(img: np.ndarray, enabled: bool = False) -> np.ndarray:
    """Haar cascade face detection — draws bounding boxes."""
    if not enabled:
        return img.copy()
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    result = img.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 100), 2)
        cv2.putText(result, f"Face", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
    return result


# ── NEW: Super-Resolution Quality Enhance ─────────────────────────────────────

def enhance_quality(img: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    One-click quality enhancement:
      1. Bicubic upscale (INTER_CUBIC)
      2. Unsharp mask sharpening
      3. CLAHE on LAB L-channel for natural contrast
      4. Light denoising to clean interpolation artifacts
    """
    h, w = img.shape[:2]
    # Step 1: bicubic upscale
    up = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    # Step 2: unsharp mask
    blur = cv2.GaussianBlur(up, (0, 0), sigmaX=2)
    sharp = np.clip(cv2.addWeighted(up, 1.5, blur, -0.5, 0), 0, 255).astype(np.uint8)
    # Step 3: CLAHE on LAB
    lab = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
    # Step 4: light denoise
    return cv2.fastNlMeansDenoisingColored(enhanced, None, 3, 3, 7, 21)


# ── Full pipeline ─────────────────────────────────────────────────────────────

def apply_all_filters(
    original: np.ndarray,
    blur_ksize: int = 1,
    sharpness_alpha: float = 1.0,
    brightness_beta: int = 0,
    contrast_alpha: float = 1.0,
    edge_enabled: bool = False,
    edge_thresh1: int = 100,
    edge_thresh2: int = 200,
    grayscale_enabled: bool = False,
    denoise_strength: int = 0,
    color_filter: str = "none",
    rotate_angle: int = 0,
    crop_x1: float = 0.0, crop_y1: float = 0.0,
    crop_x2: float = 100.0, crop_y2: float = 100.0,
    face_detect: bool = False,
) -> np.ndarray:
    img = original.copy()
    img = apply_crop(img, crop_x1, crop_y1, crop_x2, crop_y2)
    img = apply_rotate(img, rotate_angle)
    img = apply_denoise(img, denoise_strength)
    img = apply_blur(img, blur_ksize)
    img = apply_sharpness(img, sharpness_alpha)
    img = apply_brightness(img, brightness_beta)
    img = apply_contrast(img, contrast_alpha)
    img = apply_color_filter(img, color_filter)
    img = apply_grayscale(img, grayscale_enabled)
    img = apply_edge_detection(img, edge_enabled, edge_thresh1, edge_thresh2)
    img = apply_face_detection(img, face_detect)
    return img