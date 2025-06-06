from typing import Dict, List
import logging

from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Model
import torch

logger = logging.getLogger(__name__)


def get_speech_segments(
    audio_path: str, hf_token: str, device: str
) -> List[Dict[str, float]]:
    """
    Use pyannote.audio VAD to extract speech-only segments.
    """
    model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=hf_token)
    pipeline = VoiceActivityDetection(segmentation=model)
    HYPERPARAMETERS = {
        # remove speech regions shorter than that many seconds.
        "min_duration_on": 0.0,
        # fill non-speech regions shorter than that many seconds.
        "min_duration_off": 0.0,
    }
    device = torch.device(device)
    pipeline = pipeline.to(device)  # only accepts torch device, not str
    pipeline.instantiate(HYPERPARAMETERS)
    vad_output = pipeline(audio_path)
    if not vad_output:
        return []

    # Extract contiguous speech regions
    return [
        {"start": segment.start, "end": segment.end}
        for segment in vad_output.get_timeline().support()
    ]
