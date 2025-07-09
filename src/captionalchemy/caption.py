import logging
import tempfile
import os
import uuid
from typing import Literal
import argparse

from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
import whisper
import torch

from captionalchemy.tools.audio_analysis.diarization import diarize
from captionalchemy.tools.media_utils.download_video import VideoManager
from captionalchemy.tools.cv.recognize_faces import recognize_faces
from captionalchemy.tools.media_utils.extract_audio import extract_audio
from captionalchemy.tools.captioning.transcriber import Transcriber
from captionalchemy.tools.captioning.timing_analyzer import TimingAnalyzer
from captionalchemy.tools.cv.embed_known_faces import embed_faces
from captionalchemy.tools.captioning.writers.srt_writer import SRTCaptionWriter
from captionalchemy.tools.captioning.writers.vtt_writer import VTTCaptionWriter
from captionalchemy.tools.captioning.writers.sami_writer import SAMICaptionWriter
from captionalchemy.tools.audio_analysis.vad import get_speech_segments
from captionalchemy.tools.audio_analysis.non_speech_detection import (
    detect_non_speech_segments,
)
from captionalchemy.tools.audio_analysis.audio_segment_integration import (
    integrate_audio_segments,
)

logger = logging.getLogger(__name__)


def run_pipeline(
    video_url_or_path: str,
    character_identification: bool = True,
    known_faces_json: str = "known_faces.json",
    embed_faces_json: str = "embed_faces.json",
    caption_output_path: str = "output_captions",
    caption_format: Literal["vtt", "srt", "smi"] = "srt",
):
    """
    Core pipeline that:
    1. Embeds known faces (if enabled),
    2. Downloads/extracts audio,
    3. Runs VAD + diarization,
    4. Runs Whisper transcription + face-based speaker ID,
    5. Writes captions in the chosen format (SRT, VTT, SMI).

    Args:
        video_url_or_path: URL or local path of the video to caption.
        character_identification: Enable/disable face-based speaker identification.
        known_faces_json: Path to JSON file listing known faces to embed.
        embed_faces_json: JSON path to store face embeddings.
        caption_output_path: Base path (without extension) for output captions.
        caption_format: Format for output captions (srt, vtt, smi).
    """
    logger.info("Embedding known faces...")
    if character_identification:
        embed_faces(known_faces_json, embed_faces_json)
    video_manager = VideoManager(use_file_buffer=False)
    if caption_format == "srt":
        writer = SRTCaptionWriter()
    elif caption_format == "vtt":
        writer = VTTCaptionWriter()
    elif caption_format == "smi":
        writer = SAMICaptionWriter()

    transcriber = Transcriber()
    timing_analyzer = TimingAnalyzer()

    speaker_id_to_name = {}
    # Load Whisper model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base", device=device)

    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = os.path.join(temp_dir, f"video_{uuid.uuid4()}.mp4")
        audio_path = os.path.join(temp_dir, f"audio_{uuid.uuid4()}.wav")
        if os.path.exists(video_url_or_path):
            video_path = video_url_or_path
        else:
            # Get the video
            video_manager.get_video_data(video_url_or_path, video_path)
            logger.info(f"Video downloaded to {video_path}")

        # Extract the audio
        logger.info(f"Extracting audio from {video_path} to {audio_path}")
        extract_audio(video_path, audio_path)

        # Speech Activity Detection (VAD)
        logger.info("Running Voice Activity Detection (VAD)...")
        speech_segments = get_speech_segments(
            audio_path, os.getenv("HF_AUTH_TOKEN", ""), device
        )

        logger.info("Detecting non-speech segments...")
        non_speech_events = detect_non_speech_segments(audio_path, device=device)

        if not speech_segments:
            logger.warning("No speech segments detected. Exiting.")
            return

        # Diarize
        logger.info("Running diarization...")
        diarization_result = diarize(
            audio_path
        )  # { "SPEAKER_00": {"start": 3.25409375, "end": 606.2990937500001}, ..., }
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
        logger.debug(f"Integrated audio events: {integrated_audio_events}")

        # Run whisper and character identification on each individual speaker segment
        for audio_event in tqdm(
            integrated_audio_events, desc="Processing audio events"
        ):
            event_type = audio_event.event_type.value
            start = audio_event.start
            end = audio_event.end
            if event_type != "speech":
                writer.add_caption(start, end, event_type=event_type)
                continue
            speaker_id = audio_event.speaker_id
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
                speaker_name = speaker_id_to_name.get(speaker_id, speaker_id)

            # Add speaker name to the audio event
            audio_event.speaker_name = speaker_name

            word_timings = transcriber.transcribe_audio(
                audio_file=audio_path,
                start=start,
                end=end,
                model=model,
                whisper_build_path="whisper.cpp/build/bin/whisper-cli",
                whisper_model_path="whisper.cpp/models/ggml-base.en.bin",
                device=device,
            )

            subtitle_segments = timing_analyzer.suggest_subtitle_segments(word_timings)

            # Loop through each subtitle segment and add it to the writer
            for segment in subtitle_segments:
                transcription = segment.text
                start = segment.start
                end = segment.end

                # Add the caption to the writer
                writer.add_caption(
                    start=start,
                    end=end,
                    speaker=speaker_name,
                    text=transcription,
                    event_type="speech",
                )

        # Write the captions to an SRT file
        writer.write(f"{caption_output_path}.{caption_format}")
        logger.info("Captions written to output_captions.srt")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Constructs the CLI argument parser for captionalchemy."""
    parser = argparse.ArgumentParser(
        prog="captionalchemy",
        description="Download/extract audio from a video,"
        "run diarization + transcription, and output captions.",
    )

    parser.add_argument(
        "video",
        help="URL or local path of the video to caption (e.g., 'myvideo.mp4' or 'https://...').",
    )

    parser.add_argument(
        "-f",
        "--format",
        choices=["srt", "vtt", "smi"],
        default="srt",
        help="Caption output format (default: 'srt').",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="output_captions",
        help="Base path (without extension) for output captions. "
        "Actual file will be written as `<output>.<format>` (default: 'output_captions').",
    )

    parser.add_argument(
        "--no-face-id",
        action="store_false",
        dest="character_identification",
        help="Disable character identification via face recognition.",
    )

    parser.add_argument(
        "--known-faces-json",
        default="known_faces.json",
        help="Path to JSON file listing known faces to embed (default: 'known_faces.json').",
    )

    parser.add_argument(
        "--embed-faces-json",
        default="embed_faces.json",
        help="JSON path to store face embeddings (default: 'embed_faces.json').",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging output."
    )

    return parser


def main():
    load_dotenv(find_dotenv(), override=True)
    parser = _build_arg_parser()
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    logger = logging.getLogger("captionalchemy")
    logger.info("Starting captionalchemy pipeline...")
    run_pipeline(
        video_url_or_path=args.video,
        character_identification=args.character_identification,
        known_faces_json=args.known_faces_json,
        embed_faces_json=args.embed_faces_json,
        caption_output_path=args.output,
        caption_format=args.format,
    )
    logger.info("Pipeline completed successfully.")
    logger.info("Output captions can be found at: %s.%s", args.output, args.format)


if __name__ == "__main__":
    main()
