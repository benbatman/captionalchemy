import logging
import os
from typing import Dict

from dotenv import load_dotenv, find_dotenv
import torch
from pyannote.audio import Pipeline

logger = logging.getLogger(__name__)
load_dotenv(find_dotenv(), override=True)


def diarize(audio_path: str) -> Dict[str, Dict[str, float]]:
    """
    Diarizes the audio file to identify speakers and their timestamps.
    """
    if not os.environ.get("HF_AUTH_TOKEN"):
        raise ValueError(
            "Hugging Face authentication token not found. "
            "Set the HF_AUTH_TOKEN environment variable."
        )
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.environ.get("HF_AUTH_TOKEN", None),
    )

    # Send to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    pipeline.to(device)
    logger.info(f"Diarizing audio file: {audio_path} on {device}...")

    diarization = pipeline(audio_path)
    logger.info("Diarization completed.")
    diarization_dict: Dict[str, Dict[str, float]] = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = turn.start
        end = turn.end
        if speaker not in diarization_dict:
            diarization_dict[speaker] = {"start": start, "end": end}
        else:
            diarization_dict[speaker]["end"] = end

    return diarization_dict
