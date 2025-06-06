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
from tools.captioning.transcriber import Transcriber
from tools.captioning.timing_analyzer import TimingAnalyzer
from tools.embed_known_faces import embed_faces
from tools.captioning.writers.srt_caption_writer import SRTCaptionWriter
from tools.audio_analysis.vad import get_speech_segments
from tools.audio_analysis.non_speech_detection import detect_non_speech_segments
from tools.audio_analysis.audio_segment_integration import (
    integrate_audio_segments,
)


def main(
    video_url_or_path: str,
    character_identification: bool = True,
    known_faces_json: str = "example/known_faces.json",
    embed_faces_json: str = "example/embed_faces.json",
):
    """Main function to run the inference pipeline."""
    logger.info("Embedding known faces...")
    embed_faces(known_faces_json, embed_faces_json)
    video_manager = VideoManager(use_file_buffer=False)
    writer = SRTCaptionWriter()
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
        logger.info(f"Speech segments detected: {speech_segments}")

        logger.info("Detecting non-speech segments...")
        non_speech_events = detect_non_speech_segments(audio_path, device=device)
        print(non_speech_events)

        if not speech_segments:
            logger.warning("No speech segments detected. Exiting.")
            return

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

        print("Integrated audio events:\n")
        print(integrated_audio_events)

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
                speaker_name = speaker_id_to_name[speaker_id]

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
        writer.write("output_captions.srt")
        logger.info("Captions written to output_captions.srt")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    load_dotenv(find_dotenv(), override=True)

    video_url = "mfmt0hf5_spac_426_fordotorg-mp4-720p-3000k.mp4"
    logger.info("Starting inference pipeline...")
    main(
        video_url,
    )
