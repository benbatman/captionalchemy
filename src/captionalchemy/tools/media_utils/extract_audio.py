import logging

from pydub import AudioSegment

logger = logging.getLogger(__name__)


def extract_audio(video_path: str, audio_output_path: str, format: str = "wav") -> None:
    try:
        audio = AudioSegment.from_file(video_path, format="mp4")

        audio = audio.set_channels(1)  # Convert to mono
        # Set sampling rate to 44.1 kHz
        audio = audio.set_frame_rate(44100)
        audio.export(audio_output_path, format="wav", bitrate="160k")
        logger.info(f"Audio extracted as {audio_output_path}.")

    except Exception as e:
        logger.error(f"Error extracting audio: {e}")
        raise
