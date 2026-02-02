import time
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2
import torch
import joblib
import clip
from PIL import Image
import pyrealsense2 as rs


CLASS_NAMES: List[str] = [
    "nothing",      # 0
    "drink",        # 1
    "eat",          # 2
    "pick_floor",   # 3
    "reach_back",   # 4
    "sing",         # 5
    "phone_call",   # 6
    "yawn",         # 7 (will be mapped to label 0 / nothing)
]

# Heuristic risk weights (tweak later)
DISTRACTION_WEIGHTS: Dict[str, float] = {
    "nothing": 0.0,
    "drink": 0.5,
    "eat": 0.5,
    "pick_floor": 1.0,
    "reach_back": 1.0,
    "sing": 0.2,
    "phone_call": 1.0,
    "yawn": 0.0,  # treat as nothing
}

# ---- Behavior you requested ----
RAW_THRESHOLD = 0.80       # show label only if raw confidence >= this (try 0.70 if too strict)
SHOW_COOLDOWN_SEC = 1.5    # keep the last confident label visible for this many seconds
PRINT_PROBS = True
USE_ROI = False
ROI_REL = (0.0, 0.0, 1.0, 1.0)  # (x0, y0, x1, y1) in relative coords if USE_ROI is True


def build_model(device: str = "cuda"):
    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()
    return model, preprocess


def maybe_crop_roi(frame: np.ndarray) -> np.ndarray:
    if not USE_ROI:
        return frame

    h, w = frame.shape[:2]
    x0r, y0r, x1r, y1r = ROI_REL
    x0 = int(max(0, min(w - 1, x0r * w)))
    y0 = int(max(0, min(h - 1, y0r * h)))
    x1 = int(max(1, min(w, x1r * w)))
    y1 = int(max(1, min(h, y1r * h)))

    if x1 <= x0 or y1 <= y0:
        return frame

    return frame[y0:y1, x0:x1]


def bgr_to_clip_tensor(frame_bgr: np.ndarray, preprocess, device: str) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(frame_rgb)
    x = preprocess(img).unsqueeze(0).to(device)
    return x


@torch.no_grad()
def infer_one(model, classifier, frame_rgb: np.ndarray, preprocess, device: str):
    img = Image.fromarray(frame_rgb).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)

    img_feat = model.encode_image(x)          # NO normalization (match repo)
    feat_np = img_feat.detach().cpu().numpy()

    probs = classifier.predict_proba(feat_np)[0].astype(float)
    pred_id = int(np.argmax(probs))
    pred_score = float(probs[pred_id])
    return pred_id, pred_score, probs


def weighted_distraction_score(probs: np.ndarray) -> float:
    s = 0.0
    for i, p in enumerate(probs):
        cname = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}"
        w = DISTRACTION_WEIGHTS.get(cname, 0.0)
        s += w * float(p)
    return float(max(0.0, min(1.0, s)))


def format_probs(probs: np.ndarray) -> List[float]:
    return [float(p) for p in probs]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device = {device}")

    classifier_path = "model_ckpt/frontcam_vitl14-per1_hypc_429_1000_ft.pkl"
    classifier = joblib.load(classifier_path)
    print(f"[INFO] loaded classifier: {classifier_path}")

    model, preprocess = build_model(device=device)

    # Warmup
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    _ = infer_one(model, classifier, dummy, preprocess, device)
    print("[INFO] warmup done")

    # ---------------- RealSense config ----------------
    width, height, cam_fps = 640, 480, 15
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, cam_fps)

    pipeline.start(config)
    print(f"[INFO] RealSense started: {width}x{height} @ {cam_fps} FPS (RGB8)")
    time.sleep(0.5)

    # We display every frame, but infer at target_infer_fps
    target_infer_fps = 1.0
    infer_period = 1.0 / target_infer_fps
    last_infer_t = 0.0

    ema: Optional[float] = None
    alpha = 0.7

    # Overlay behavior: show None until a confident prediction appears
    last_label_text = "None"
    last_show_until = 0.0

    try:
        print("[INFO] Press Q to quit.")
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame_rgb = np.asanyarray(color_frame.get_data())
            frame_rgb_roi = maybe_crop_roi(frame_rgb)

            frame_bgr_full = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)  # solo display

            now = time.time()
            label_text = "None" if now > last_show_until else last_label_text

            if (now - last_infer_t) >= infer_period:
                last_infer_t = now

                pred_id, pred_score, probs = infer_one(
                    model, classifier, frame_rgb_roi, preprocess, device
                )

                # Map yawn (7) -> nothing (0)
                if len(probs) > 7:
                    probs[0] = float(probs[0]) + float(probs[7])
                    probs[7] = 0.0
                if pred_id == 7:
                    pred_id = 0

                probs_list = format_probs(probs)

                pairs = list(zip(CLASS_NAMES, probs_list))
                pairs_str = " | ".join([f"{name}:{p:.3f}" for name, p in pairs])
                print(f"[INFER] {pairs_str}")

                distr = weighted_distraction_score(probs)
                ema = distr if ema is None else (alpha * ema + (1.0 - alpha) * distr)

                if pred_score >= RAW_THRESHOLD:
                    name = CLASS_NAMES[pred_id]
                    last_label_text = f"{name}"
                    last_show_until = now + SHOW_COOLDOWN_SEC

                label_text = "None" if time.time() > last_show_until else last_label_text

            # Overlay on full frame (even if ROI is used for inference)
            cv2.putText(
                frame_bgr_full,
                label_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("DriveCLIP RealSense D435", frame_bgr_full)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), ord("Q")):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
