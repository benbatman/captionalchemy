import json
import logging
from typing import Dict, Any, List

import cv2
import numpy as np
import torch
from scipy.spatial.distance import cdist
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)


def recognize_faces(
    video_path: str,
    start: float,
    end: float,
    embed_faces_json: str,
    threshold: float = 0.35,
    frame_interval: int = 60,
) -> List[Dict[str, Any]]:
    """
    Recognizes faces in a video by comparing to labeled embeddings.

    Args:
        video_path (str): Path to input video.
        known_faces_json (str): JSON file with list of {"name": str, "embedding": List[float]}.
        threshold (float): Cosine‐distance threshold for a valid match.

    Returns:
        List[dict]: Each with:
            - "timestamp": float, seconds into video.
            - "bbox": [x1, y1, x2, y2].
            - "face_id": int, sequential ID per frame.
            - "name": str, matched label or "unknown".
    """
    logger.info(f"Recognizing faces in video: {video_path}")

    # --- load & normalize known embeddings ---
    with open(embed_faces_json, "r") as f:
        known_list = json.load(f)
    known_names = [e["name"] for e in known_list]
    known_embs = np.vstack([e["embedding"] for e in known_list]).astype(np.float32)
    # ensure L2‐unit length
    norms = np.linalg.norm(known_embs, axis=1, keepdims=True)
    known_embs = known_embs / np.clip(norms, 1e-6, None)

    provider = (
        "CUDAExecutionProvider"
        if torch.cuda.is_available()
        else (
            "CoreMLExecutionProvider"
            if torch.backends.mps.is_available()
            else "CPUExecutionProvider"
        )
    )
    ctx_id = 0 if provider == "CUDAExecutionProvider" else -1

    app = FaceAnalysis(
        providers=[provider],
        allowed_modules=["detection", "recognition"],
    )
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    results: List[Dict[str, Any]] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = frame_idx / fps

        # Skip frames before segment start
        if timestamp < start:
            frame_idx += 1
            continue
        # Stop processing frames after segment end
        if timestamp > end:
            break

        # Only run the pipeline every `frame_interval` frames
        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / fps
            faces = app.get(frame)  # detect→align→recognize in one step

            for fid, face in enumerate(faces):
                x1, y1, x2, y2 = face.bbox.astype(int).tolist()
                emb = face.embedding  # already L2-normalized

                # cosine distance (smaller is more similar)
                dists = cdist([emb], known_embs, metric="cosine")[0]
                best_idx = int(np.argmin(dists))
                best_dist = float(dists[best_idx])
                name = known_names[best_idx] if best_dist < threshold else "unknown"

                results.append(
                    {
                        "timestamp": timestamp,
                        "bbox": [x1, y1, x2, y2],
                        "face_id": fid,
                        "name": name,
                    }
                )

        frame_idx += 1

    cap.release()
    return results
