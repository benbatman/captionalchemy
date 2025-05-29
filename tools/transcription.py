import subprocess
import tempfile
import os

import whisper
import torch


def transcribe_audio(
    audio_file: str, start: float, end: float, model: str = "base"
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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Trim the audio ussing ffmpeg
    subprocess.run(
        [
            "ffmpeg",
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

    # Load the Whisper model
    model = whisper.load_model(model, device=device)
    result = model.transcribe(tmp_path, language="en", fp16=torch.cuda.is_available())
    transcript = result["text"]
    os.remove(tmp_path)  # Clean up the temporary file

    return transcript
