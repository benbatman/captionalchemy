import subprocess
import tempfile
import os
import logging
from sys import platform

from whisper import Whisper
import torch

logger = logging.getLogger(__name__)


def transcribe_audio(
    audio_file: str,
    start: float,
    end: float,
    model: Whisper,
    whisper_build_path: str,
    whisper_model_path: str,
) -> str:
    """
    Transcribe audio to text using OpenAI's Whisper model.

    Args:
        audio_file (str): Path to the audio file.
        model (str): The Whisper model to use. Default is "whisper-1".

    Returns:
        str: The transcribed text.
    """
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_path = tmp_wav.name
    tmp_wav.close()

    # Trim the audio using ffmpeg
    subprocess.run(
        [
            "ffmpeg",
            "-y",  # Overwrite output files without asking
            "-i",
            audio_file,
            "-ss",
            str(start),
            "-to",
            str(end),
            "-ar",
            "16000",  # Set sample rate to 16 kHz
            "-ac",
            "1",  # Set number of channels to 1 (mono)
            "-f",
            "wav",
            tmp_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    logger.debug(f"Trimmed audio saved to {tmp_path}")

    if platform == "darwin" and os.path.exists("whisper.cpp"):
        logger.info("Using whisper.cpp for transcription on macOS")
        cmd = [
            whisper_build_path,
            "-m",
            whisper_model_path,
            "-f",
            tmp_path,
            "--no-timestamps",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Error in transcription: {result.stderr}")

        transcription = result.stdout
        return transcription
    else:
        logger.info("Using Whisper Python API for transcription")
        result = model.transcribe(
            tmp_path, language="en", fp16=torch.cuda.is_available()
        )
        transcription = result["text"]
        os.remove(tmp_path)  # Clean up the temporary file
        logger.debug(f"Transcription result: {transcription}")

        return transcription
