import logging
import tempfile
import os
import uuid

from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
import whisper
import torch

from tools.audio_analysis.diarization import diarize
from tools.download_video import VideoManager
from tools.recognize_faces import recognize_faces
from tools.extract_audio import extract_audio
from tools.transcription import transcribe_audio
from tools.embed_known_faces import embed_faces
from tools.caption_formatter import SRTCaptionWriter
from tools.audio_analysis.vad import get_speech_segments
from tools.audio_analysis.non_speech_detection import detect_non_speech_segments
from tools.audio_analysis.audio_segment_integration import integrate_audio_segments


def main(
    video_url: str,
    character_identification: bool = True,
    known_faces_json: str = "known_faces.json",
    embed_faces_json: str = "embed_faces.json",
):
    """Main function to run the inference pipeline."""
    logger.info("Embedding known faces...")
    embed_faces(known_faces_json, embed_faces_json)
    video_manager = VideoManager(use_file_buffer=False)
    writer = SRTCaptionWriter()
    speaker_id_to_name = {}
    # Load Whisper model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base", device=device)

    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = os.path.join(temp_dir, f"video_{uuid.uuid4()}.mp4")
        audio_path = os.path.join(temp_dir, f"audio_{uuid.uuid4()}.wav")
        # Get the video
        video_manager.get_video_data(video_url, video_path)
        logger.info(f"Video downloaded to {video_path}")

        # Extract the audio
        logger.info(f"Extracting audio from {video_path} to {audio_path}")
        extract_audio(video_path, audio_path)

        # Speech Activity Detection (VAD)
        logger.info("Running Voice Activity Detection (VAD)...")
        speech_segments = get_speech_segments(
            audio_path, os.getenv("HF_AUTH_TOKEN", ""), device
        )
        logger.info(f"Speech segments detected: {speech_segments}")

        non_speech_events = detect_non_speech_segments(audio_path, device=device)
        print(non_speech_events)

        if not speech_segments:
            logger.warning("No speech segments detected. Exiting.")
            return

        exit(1)

        # Diarize
        # diarization_result = diarize(audio_path)  looks like this: { "SPEAKER_00": {"start": 3.25409375, "end": 606.2990937500001}, ..., SPEAKER_XX: {} }
        diarization_result = {
            "SPEAKER_00": {"start": 3.25409375, "end": 606.2990937500001}
        }
        logger.info("Completed diarization.")
        logger.debug(f"Diarization result: {diarization_result}")

        # Integrate audio segments
        logger.info("Integrating audio segments...")
        integrated_audio_events = integrate_audio_segments(
            speech_segments,
            non_speech_events,
            diarization_result,
            total_audio_duration=None,
        )

        # Run whisper on each individual speaker segment
        for speaker_id, segment in diarization_result.items():
            start = segment["start"]
            end = segment["end"]
            logger.debug(
                f"Processing speaker {speaker_id} from {start} to {end} seconds..."
            )

            # Does speaker already have a name?
            speaker_identified = speaker_id in speaker_id_to_name
            if not speaker_identified and character_identification:
                logger.debug(
                    f"Character identification enabled for speaker {speaker_id}."
                )

                # Recognize faces in this video segment
                recognized_faces = recognize_faces(
                    video_path,
                    start,
                    end,
                    embed_faces_json,
                )
                logger.debug(
                    f"Recognized faces for speaker {speaker_id}: {recognized_faces}"
                )

                speaker_name = recognized_faces[0]["name"]
                # Map speaker ID to name
                speaker_id_to_name[speaker_id] = speaker_name
            else:
                speaker_name = speaker_id_to_name[speaker_id]

            # Transcribe each audio segment
            logger.info("Beginning transcription...")
            transcription = transcribe_audio(
                audio_path,
                start,
                end,
                model,
                "whisper.cpp/build/bin/whisper-cli",
                "whisper.cpp/models/ggml-base.en.bin",
            )
            logger.info(f"Speaker name: {speaker_name}, Transcription: {transcription}")
            # Add the caption to the writer
            writer.add_caption(
                start=start,
                end=end,
                speaker=speaker_name,
                text=transcription,
            )

        # Write the captions to an SRT file
        writer.write("output_captions.srt")
        logger.info("Captions written to output_captions.srt")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    load_dotenv(find_dotenv(), override=True)

    video_url = "https://ga.video.cdn.pbs.org/videos/pbs-space-time/f6d229fe-d0dd-4207-8032-691aeab2a467/2000057365/hd-16x9-mezzanine-1080p/mfmt0hf5_spac_426_fordotorg-mp4-720p-3000k.mp4"
    logger.info("Starting inference pipeline...")
    main(
        video_url,
    )
