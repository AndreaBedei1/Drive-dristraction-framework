import time
from typing import List, Dict, Optional, Tuple

import numpy as np
import cv2
import torch
import joblib
import clip
from PIL import Image
import pyrealsense2 as rs


CLASS_NAMES: List[str] = [
    "hair_adjust",  # 0
    "drink",        # 1
    "eat",          # 2
    "pick_floor",   # 3
    "reach_back",   # 4
    "sing",         # 5
    "phone_call",   # 6
    "yawn",         # 7
]

# Severity (0..1) assigned to the action when confident
ACTION_SEVERITY: Dict[str, float] = {
    "hair_adjust": 0.20,
    "drink": 0.35,
    "eat": 0.45,
    "pick_floor": 1.00,
    "reach_back": 1.00,
    "sing": 0.10,
    "phone_call": 1.00,
    "yawn": 0.25,
}

RAW_THRESHOLD = 0.80
USE_ROI = False
ROI_REL = (0.0, 0.0, 1.0, 1.0)
SHOW_COOLDOWN_SEC = 1.5


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
def infer_one(
    model,
    classifier,
    frame_bgr: np.ndarray,
    preprocess,
    device: str
) -> Tuple[int, float, np.ndarray]:
    # Keep your conversion path exactly (BGR -> RGB -> PIL -> preprocess)
    x = bgr_to_clip_tensor(frame_bgr, preprocess, device)

    img_feat = model.encode_image(x)  # NO normalization (match repo)
    feat_np = img_feat.detach().cpu().numpy()

    probs = classifier.predict_proba(feat_np)[0].astype(float)
    pred_id = int(np.argmax(probs))
    pred_score = float(probs[pred_id])
    return pred_id, pred_score, probs


def format_probs(probs: np.ndarray) -> List[float]:
    return [float(p) for p in probs]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device = {device}")

    classifier_path = "model_ckpt/syndd1_vitl14-per1_hypc_429_1000_ft.pkl"
    classifier = joblib.load(classifier_path)
    print(f"[INFO] loaded classifier: {classifier_path}")

    model, preprocess = build_model(device=device)

    # Warmup (dummy BGR, since infer_one expects BGR now)
    dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    _ = infer_one(model, classifier, dummy_bgr, preprocess, device)
    print("[INFO] warmup done")

    # RealSense config
    width, height, cam_fps = 640, 480, 15
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, cam_fps)

    pipeline.start(config)
    print(f"[INFO] RealSense started: {width}x{height} @ {cam_fps} FPS (RGB8)")
    time.sleep(0.5)

    target_infer_fps = 1.0
    infer_period = 1.0 / target_infer_fps
    last_infer_t = 0.0

    last_overlay_text = "None"
    last_show_until = 0.0

    try:
        print("[INFO] Press Q to quit.")
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame_rgb = np.asanyarray(color_frame.get_data())
            frame_bgr_full = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)  # keep your display conversion

            # ROI (optional) applied on BGR (keeps your current path consistent)
            frame_bgr_roi = maybe_crop_roi(frame_bgr_full)

            now = time.time()
            label_text = "None" if now > last_show_until else last_overlay_text

            if (now - last_infer_t) >= infer_period:
                last_infer_t = now

                pred_id, pred_score, probs = infer_one(
                    model, classifier, frame_bgr_roi, preprocess, device
                )

                # Print ONLY if distracted with confidence >= threshold
                if pred_score >= RAW_THRESHOLD:
                    action = CLASS_NAMES[pred_id]
                    severity = float(ACTION_SEVERITY.get(action, 0.50))

                    print(
                        f"[DISTRACTED] conf={pred_score:.3f} action={action} severity={severity:.2f}"
                    )

                    last_overlay_text = f"{action} sev={severity:.2f} conf={pred_score:.2f}"
                    last_show_until = now + SHOW_COOLDOWN_SEC

                # Update overlay after potential change
                label_text = "None" if time.time() > last_show_until else last_overlay_text

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
