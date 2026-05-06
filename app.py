"""
Egg Analyzer API — Two-Layer Cascade Classifier
Layer 1: Candling Viability  →  infertile | dead | fertile
Layer 2: Stage Detection     →  initial_stage | middle_stage | late_stage

Annotation format on image:
  Infertile/Dead  →  "INFERTILE  92.3%"  (red box)
  Fertile         →  "FERTILE 88.1% / INITIAL STAGE 76.4%"  (green box)
"""

from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
LOGS_DIR   = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE   = LOGS_DIR / "analysis_results.txt"

MODEL_L1_PATH = Path(
    r"C:\Users\Administrator\Desktop\egg-cubator"
    r"\egg-analyzer-model-training\egg_analyzer_layer_one_model"
    r"\egg_condition_model_finetuned.keras"
)
MODEL_L2_PATH = Path(
    r"C:\Users\Administrator\Desktop\egg-cubator"
    r"\egg-analyzer-model-training\egg_analyzer_layer_two_model"
    r"\egg_stage_model_finetuned.keras"
)

# ─────────────────────────────────────────────
# Class labels (must match training order)
# ─────────────────────────────────────────────
L1_CLASSES = ["infertile", "dead", "fertile"]   # alphabetical = Keras default
L2_CLASSES = ["initial_stage", "late_stage", "middle_stage"]  # alphabetical

IMG_SIZE = 224

# ─────────────────────────────────────────────
# Load models once at startup
# ─────────────────────────────────────────────
print("Loading Layer 1 model …")
model_l1: tf.keras.Model = tf.keras.models.load_model(MODEL_L1_PATH)
print("  ✓ Layer 1 loaded")

print("Loading Layer 2 model …")
model_l2: tf.keras.Model = tf.keras.models.load_model(MODEL_L2_PATH)
print("  ✓ Layer 2 loaded")

# Warm-up inference (avoids first-request latency)
_dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
model_l1.predict(_dummy, verbose=0)
model_l2.predict(_dummy, verbose=0)
print("Models warm-up complete.\n")

# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────
app = FastAPI(
    title="Egg Analyzer API",
    description="Two-layer cascade classifier for egg candling analysis.",
    version="1.0.0",
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """Resize + MobileNetV2 preprocess → (1, 224, 224, 3) float32."""
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img / 127.5) - 1.0          # MobileNetV2 [-1, 1]
    return np.expand_dims(img, 0)


def _locate_egg(img_bgr: np.ndarray) -> tuple[int, int, int, int]:
    """
    Detect the egg's bounding box using brightness-based contour detection.
    Candling images: egg appears as the brightest elliptical region.
    Returns (x, y, w, h) of the tightest bounding rect around the egg.
    Falls back to full-image bounds if detection fails.
    """
    h, w = img_bgr.shape[:2]

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Blur to suppress noise, then threshold the bright egg area
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological closing to fill holes in the egg mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        pad = max(4, min(h, w) // 40)
        return pad, pad, w - 2 * pad, h - 2 * pad

    # Pick the largest contour (the egg body)
    largest = max(contours, key=cv2.contourArea)

    # Reject if it's too small (< 5 % of image area) — fallback to full image
    if cv2.contourArea(largest) < 0.05 * h * w:
        pad = max(4, min(h, w) // 40)
        return pad, pad, w - 2 * pad, h - 2 * pad

    bx, by, bw, bh = cv2.boundingRect(largest)

    # Small padding so the box doesn't clip the edge of the egg
    pad = max(4, min(h, w) // 50)
    bx = max(0, bx - pad)
    by = max(0, by - pad)
    bw = min(w - bx, bw + 2 * pad)
    bh = min(h - by, bh + 2 * pad)

    return bx, by, bw, bh


def _draw_annotation(
    img_bgr: np.ndarray,
    label: str,
    color_bgr: tuple[int, int, int],
) -> np.ndarray:
    """Detect egg, draw tight bounding box + label onto image. Returns annotated copy."""
    out = img_bgr.copy()
    h, w = out.shape[:2]

    bx, by, bw, bh = _locate_egg(img_bgr)
    pt1 = (bx, by)
    pt2 = (bx + bw, by + bh)
    box_thick = max(2, min(h, w) // 150)
    cv2.rectangle(out, pt1, pt2, color_bgr, thickness=box_thick)

    # Smaller font — label lives outside the box
    font_scale = max(0.3, min(w / 700, 0.7))
    thickness  = max(1, int(font_scale * 2))
    font       = cv2.FONT_HERSHEY_DUPLEX

    # Split label onto two lines if it contains " / "
    lines = label.split(" / ")
    pad   = 5

    # Measure each line
    line_sizes = [cv2.getTextSize(l, font, font_scale, thickness) for l in lines]
    line_heights = [s[0][1] for s in line_sizes]
    line_widths  = [s[0][0] for s in line_sizes]
    baselines    = [s[1] for s in line_sizes]
    max_lw       = max(line_widths)
    total_text_h = sum(line_heights) + sum(baselines) + pad * (len(lines) + 1)

    bg_x1 = bx
    bg_x2 = min(w, bx + max_lw + pad * 2)

    # Prefer above the box; fall back to below if not enough vertical room
    if by >= total_text_h:
        # Room above — label floats above the top edge
        bg_y1 = by - total_text_h
        bg_y2 = by
    elif (h - (by + bh)) >= total_text_h:
        # Room below — label sits below the bottom edge
        bg_y1 = by + bh
        bg_y2 = by + bh + total_text_h
    else:
        # No clean room either way — anchor to the top of the image and
        # draw the label above the box (overlapping image edge is acceptable)
        bg_y1 = 0
        bg_y2 = total_text_h

    cv2.rectangle(out, (bg_x1, bg_y1), (bg_x2, bg_y2), color_bgr, -1)

    # Draw each line of text inside the background rect
    cursor_y = bg_y1 + pad
    for i, line in enumerate(lines):
        lh = line_heights[i]
        cursor_y += lh
        cv2.putText(
            out, line,
            (bg_x1 + pad, cursor_y),
            font, font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        cursor_y += baselines[i] + pad

    return out


def _run_inference(img_bgr: np.ndarray) -> dict:
    """
    Full two-layer inference.
    Returns result dict with label, annotation color, and per-layer probabilities.
    """
    tensor = _preprocess(img_bgr)

    # ── Layer 1 ──────────────────────────────
    l1_probs = model_l1.predict(tensor, verbose=0)[0]
    l1_idx   = int(np.argmax(l1_probs))
    l1_class = L1_CLASSES[l1_idx]
    l1_conf  = float(l1_probs[l1_idx]) * 100

    if l1_class in ("infertile", "dead"):
        label = f"{l1_class.upper()}  {l1_conf:.1f}%"
        color = (0, 0, 220)   # red (BGR)
        return {
            "layer1_class": l1_class,
            "layer1_confidence": round(l1_conf, 2),
            "layer1_probs": {c: round(float(p) * 100, 2) for c, p in zip(L1_CLASSES, l1_probs)},
            "layer2_class": None,
            "layer2_confidence": None,
            "layer2_probs": None,
            "annotation_label": label,
            "annotation_color_bgr": color,
        }

    # ── Layer 2 (fertile only) ───────────────
    l2_probs = model_l2.predict(tensor, verbose=0)[0]
    l2_idx   = int(np.argmax(l2_probs))
    l2_class = L2_CLASSES[l2_idx]
    l2_conf  = float(l2_probs[l2_idx]) * 100

    stage_display = l2_class.replace("_", " ").title()
    label = f"FERTILE {l1_conf:.1f}% / {stage_display.upper()} {l2_conf:.1f}%"
    color = (34, 139, 34)   # green (BGR)

    return {
        "layer1_class": l1_class,
        "layer1_confidence": round(l1_conf, 2),
        "layer1_probs": {c: round(float(p) * 100, 2) for c, p in zip(L1_CLASSES, l1_probs)},
        "layer2_class": l2_class,
        "layer2_confidence": round(l2_conf, 2),
        "layer2_probs": {c: round(float(p) * 100, 2) for c, p in zip(L2_CLASSES, l2_probs)},
        "annotation_label": label,
        "annotation_color_bgr": color,
    }


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Egg Analyzer API is running."}


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "layer1_model": MODEL_L1_PATH.name,
        "layer2_model": MODEL_L2_PATH.name,
        "layer1_classes": L1_CLASSES,
        "layer2_classes": L2_CLASSES,
    }


class AnalyzeResponse(BaseModel):
    layer1_class: str
    layer1_confidence: float
    layer1_probs: dict
    layer2_class: str | None
    layer2_confidence: float | None
    layer2_probs: dict | None
    annotation_label: str
    annotated_image: str
    inference_ms: float


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Inference"])
async def analyze(file: UploadFile = File(...)):
    """
    Analyze a candled egg image.

    Returns classification result and logs to text file.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    raw = await file.read()
    np_arr = np.frombuffer(raw, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=422, detail="Could not decode image.")

    t0 = time.perf_counter()
    result = _run_inference(img_bgr)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    annotated = _draw_annotation(img_bgr, result["annotation_label"], result["annotation_color_bgr"])

    # Encode annotated image as base64 JPEG for frontend display
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    annotated_b64 = base64.b64encode(buf).decode("utf-8")

    # Log analysis result to text file
    log_entry = (
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
        f"File: {file.filename} | "
        f"Layer1: {result['layer1_class']} ({result['layer1_confidence']}%) | "
        f"Layer2: {result['layer2_class'] or 'N/A'} ({result['layer2_confidence'] or 'N/A'}%) | "
        f"Inference: {round(elapsed_ms, 2)}ms\n"
    )
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry)

    return AnalyzeResponse(
        layer1_class       = result["layer1_class"],
        layer1_confidence  = result["layer1_confidence"],
        layer1_probs       = result["layer1_probs"],
        layer2_class       = result["layer2_class"],
        layer2_confidence  = result["layer2_confidence"],
        layer2_probs       = result["layer2_probs"],
        annotation_label   = result["annotation_label"],
        annotated_image    = annotated_b64,
        inference_ms       = round(elapsed_ms, 2),
    )


@app.get("/logs", tags=["Logs"])
def get_logs():
    """Retrieve the analysis results log file."""
    if not LOG_FILE.exists():
        return {"message": "No logs yet.", "content": ""}
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    return {"file": str(LOG_FILE), "content": content}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
