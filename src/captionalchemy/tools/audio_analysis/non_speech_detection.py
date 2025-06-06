from typing import List, Dict
import logging

from panns_inference import labels, SoundEventDetection
import librosa
import numpy as np


def detect_non_speech_segments(
    audio_path: str,
    primary_threshold: float = 0.5,
    early_threshold: float = 0.3,  # Lower threshold for first few seconds
    early_duration: float = 5.0,  # Apply lower threshold for first 10 seconds
    sample_rate: int = 32000,
    device: str = "cpu",
) -> List[Dict]:
    """
    Enhanced detection that uses a lower threshold for early audio segments.
    This helps catch music that the model is less confident about initially.
    """
    logger = logging.getLogger(__name__)

    audio, _ = librosa.core.load(audio_path, sr=sample_rate, mono=True)
    audio = audio[None, :]

    sed = SoundEventDetection(checkpoint_path=None, device=device)
    framewise_output = sed.inference(audio)
    scores = framewise_output[0]
    logger.debug(
        f"Scores shape: {scores.shape}, Labels: {labels}, Sample rate: {sample_rate}"
    )

    speech_index = labels.index("Speech")
    hop_sec = 320 / sample_rate
    early_frames = int(early_duration / hop_sec)

    events = []

    for class_idx in range(scores.shape[1]):
        if class_idx == speech_index:
            continue

        # Create adaptive threshold mask
        mask = np.zeros(len(scores), dtype=bool)

        # Use lower threshold for early frames
        if early_frames > 0:
            early_mask = scores[:early_frames, class_idx] >= early_threshold
            mask[:early_frames] = early_mask

        # Use primary threshold for remaining frames
        if len(scores) > early_frames:
            later_mask = scores[early_frames:, class_idx] >= primary_threshold
            mask[early_frames:] = later_mask

        if not mask.any():
            continue

        diff_mask = np.diff(mask.astype(int))
        start_frames = np.where(diff_mask == 1)[0] + 1
        end_frames = np.where(diff_mask == -1)[0] + 1

        if mask[0]:
            start_frames = np.concatenate([[0], start_frames])
        if mask[-1]:
            end_frames = np.concatenate([end_frames, [len(mask)]])

        for start_frame, end_frame in zip(start_frames, end_frames):
            start_time = start_frame * hop_sec
            end_time = end_frame * hop_sec
            avg_confidence = np.mean(scores[start_frame:end_frame, class_idx])

            event = {
                "start": start_time,
                "end": end_time,
                "label": labels[class_idx],
                "confidence": avg_confidence,
                "duration": end_time - start_time,
                "threshold_used": (
                    early_threshold if start_frame < early_frames else primary_threshold
                ),
            }
            events.append(event)

    events.sort(key=lambda x: x["start"])
    return events
