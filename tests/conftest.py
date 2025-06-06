import tempfile
import os
import json
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass
import wave
import math
import logging

import cv2
import pytest
import numpy as np


from src.captionalchemy.tools.captioning.timing_analyzer import (
    WordTiming,
    SubtitleSegment,
    TimingAnalyzer,
)
from src.captionalchemy.tools.audio_analysis.audio_segment_integration import (
    AudioEvent,
    EventType,
)


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_word_timings():
    """
    Sample word timings for testing.
    """
    return [
        WordTiming(word="Hello", start=0.0, end=0.5),
        WordTiming(word="world", start=0.6, end=1.0),
        WordTiming(word="This", start=1.1, end=1.5),
        WordTiming(word="is", start=1.6, end=2.0),
        WordTiming(word="a", start=2.1, end=2.3),
        WordTiming(word="test", start=2.4, end=3.0),
        WordTiming(
            word=".", start=3.1, end=3.5, is_punctuation=True, is_sentence_ending=True
        ),
    ]


@pytest.fixture
def sample_audio_events():
    """
    Creates sample AudioEvent objects for testing integration.

    This simulates the output from audio analysis components - speech segments,
    music detection, and silence periods. This mix helps us test how the
    integration logic handles different event types.
    """
    return [
        AudioEvent(
            start=0.0, end=5.0, event_type=EventType.SPEECH, speaker_id="SPEAKER_00"
        ),
        AudioEvent(start=5.5, end=8.0, event_type=EventType.MUSIC, label="music"),
        AudioEvent(
            start=8.5, end=12.0, event_type=EventType.SPEECH, speaker_id="SPEAKER_01"
        ),
        AudioEvent(start=12.0, end=13.0, event_type=EventType.SILENCE),
        AudioEvent(
            start=13.5, end=18.0, event_type=EventType.SPEECH, speaker_id="SPEAKER_00"
        ),
    ]


@pytest.fixture
def sample_speech_segments():
    """
    Creates sample speech segments as they would come from VAD.

    Voice Activity Detection (VAD) outputs segments where speech is detected.
    This fixture provides realistic timing data for testing.
    """
    return [
        {"start": 0.0, "end": 5.0, "duration": 5.0},
        {"start": 8.5, "end": 12.0, "duration": 3.5},
        {"start": 13.5, "end": 18.0, "duration": 4.5},
    ]


@pytest.fixture
def sample_diarization_result():
    """
    Creates a sample diarization result.

    This simulates the output from a diarization model, mapping speakers to their
    speaking segments.
    """
    return {
        "SPEAKER_00": {"start": 0.0, "end": 5.0},
        "SPEAKER_01": {"start": 8.5, "end": 12.0},
        "SPEAKER_00": {"start": 13.5, "end": 18.0},
    }


@pytest.fixture
def sample_known_faces_json(temp_directory):
    """
    Creates a sample known_faces.json file for face recognition testing.

    This simulates the input format for face embedding - a JSON file with
    person names and their corresponding image paths.
    """
    known_faces_data = [
        {"name": "John Doe", "image_path": os.path.join(temp_directory, "john.jpg")},
        {"name": "Jane Smith", "image_path": os.path.join(temp_directory, "jane.jpg")},
    ]

    json_path = os.path.join(temp_directory, "known_faces.json")
    with open(json_path, "w") as f:
        json.dump(known_faces_data, f)

    return json_path


@pytest.fixture
def mock_whisper_model():
    """
    Mocks the Whisper model for testing transcription.

    This simulates the behavior of the Whisper model without requiring the actual
    model files, allowing us to test transcription logic in isolation.
    """
    mock_model = MagicMock()
    # Mock the transcribe method to return realistic results
    mock_result = {
        "text": "Hello world. This is a test.",
        "segments": [
            {
                "words": [
                    {"word": " Hello", "start": 0.0, "end": 0.5},
                    {"word": " world", "start": 0.6, "end": 1.1},
                    {"word": ".", "start": 1.1, "end": 1.2},
                ]
            }
        ],
    }
    mock_model.transcribe.return_value = mock_result
    return mock_model


@pytest.fixture
def mock_face_analysis():
    """
    Mocks the face analysis component for testing speaker identification.
    """
    mock_app = MagicMock()
    mock_face = MagicMock()
    mock_face.embedding = np.random.rand(512)
    mock_app.get.return_value = [mock_face]
    return mock_app


@pytest.fixture
def sample_subtitle_segments():
    """
    Creates sample SubtitleSegment objects for testing.

    These represent the final output of the timing analysis - properly
    segmented subtitles with optimal reading speeds and break points.
    """
    return [
        SubtitleSegment(
            start=0.0,
            end=2.5,
            text="Hello world.",
            word_count=2,
            char_count=12,
            reading_speed_cps=4.8,
            break_reason="sentence_end",
        ),
        SubtitleSegment(
            start=2.5,
            end=6.0,
            text="This is a test sentence.",
            word_count=5,
            char_count=24,
            reading_speed_cps=6.9,
            break_reason="duration_limit",
        ),
    ]


@pytest.fixture
def mock_timing_analyzer():
    """
    Creates a TimingAnalyzer instance with test-friendly settings.
    """

    return TimingAnalyzer(
        max_segment_duration=6.0,
        min_segment_duration=1.0,
        max_reading_speed_cps=6.0,
        min_reading_speed_cps=2.0,
    )


# Helper functions to create test data


def create_test_audio_file(filepath: str, duration_seconds: float = 5.0) -> None:
    """
    Creats a minimal .wav audio file for testing.
    """

    sample_rate = 16000  # Standard sample rate
    samples = int(sample_rate * duration_seconds)

    with wave.open(filepath, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        # Generate a simple sine wave for the audio content
        for i in range(samples):
            value = int(32767.0 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate))
            wav_file.writeframesraw(value.to_bytes(2, byteorder="little", signed=True))


def create_test_video_file(filepath: str) -> None:
    """
    Creates a minimal MP4 video file for testing.

    This function generates a simple video file with a single frame.
    """

    # Create a black image
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Write the video file
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(filepath, fourcc, 30.0, (640, 480))

    out.write(frame)
    out.release()


@pytest.fixture(autouse=True)
def setup_logging():
    """
    Automatically sets up logging for all tests.
    """
    logging.basicConfig(level=logging.WARNING)
