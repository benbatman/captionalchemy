import os
import json
import wave
import math
import tempfile

import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock

from captionalchemy.tools.audio_analysis.audio_segment_integration import (
    AudioEvent,
    EventType,
)
from captionalchemy.tools.captioning.transcriber import WordTiming
from captionalchemy.tools.captioning.timing_analyzer import (
    TimingAnalyzer,
    SubtitleSegment,
)


@pytest.fixture
def temp_dir():
    """A temporary directory for files."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_word_timings():
    """A short list of WordTiming objects for subtitle-segmentation tests."""
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
def sample_subtitle_segments():
    """Example SubtitleSegment list from a TimingAnalyzer."""
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
    """A TimingAnalyzer configured for testing (no extreme durations)."""
    return TimingAnalyzer(
        max_segment_duration=6.0,
        min_segment_duration=1.0,
        max_reading_speed_cps=6.0,
        min_reading_speed_cps=2.0,
    )


@pytest.fixture
def mock_whisper_model():
    """A fake Whisper model with a stubbed .transcribe()."""
    model = MagicMock()
    model.transcribe.return_value = {
        "text": "Hello world.",
        "segments": [
            {
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5},
                    {"word": "world", "start": 0.6, "end": 1.1},
                ]
            }
        ],
    }
    return model


@pytest.fixture
def mock_face_analysis():
    """A fake InsightFace FaceAnalysis instance."""
    app = MagicMock()
    face = MagicMock()
    face.embedding = np.random.rand(512)
    app.get.return_value = [face]
    return app


@pytest.fixture
def sample_known_faces_json(temp_dir):
    """Creates a small known_faces.json and empty image files."""
    data = [
        {"name": "Alice", "image_path": os.path.join(temp_dir, "alice.jpg")},
        {"name": "Bob", "image_path": os.path.join(temp_dir, "bob.jpg")},
    ]
    # touch the files
    for e in data:
        open(e["image_path"], "wb").close()
    path = os.path.join(temp_dir, "known_faces.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return path


@pytest.fixture
def sample_audio_events():
    """A mixed list of AudioEvent for integration logic tests."""
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
def sample_test_audio_file(temp_dir):
    """Creates a test WAV file and returns the path."""
    path = os.path.join(temp_dir, "test_audio.wav")
    create_test_audio_file(path)
    return path


@pytest.fixture
def sample_test_video_file(temp_dir):
    """Creates a test MP4 file and returns the path."""
    path = os.path.join(temp_dir, "test_video.mp4")
    create_test_video_file(path)
    return path


def create_test_audio_file(path: str, duration: float = 2.0):
    """Helper to write a tiny WAV sine wave file."""
    sr = 16000
    samples = int(sr * duration)
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        for i in range(samples):
            val = int(32767 * math.sin(2 * math.pi * 440 * i / sr))
            w.writeframesraw(val.to_bytes(2, "little", signed=True))


def create_test_video_file(path: str):
    """Helper to write a one-frame MP4 via OpenCV."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, 24.0, (640, 480))
    out.write(frame)
    out.release()


# Additional fixtures for pytest conversion


@pytest.fixture
def mock_speech_segments():
    """Mock speech segments for testing."""
    return [
        {"start": 0.0, "end": 5.0, "duration": 5.0},
        {"start": 10.0, "end": 15.0, "duration": 5.0},
        {"start": 20.0, "end": 25.0, "duration": 5.0},
    ]


@pytest.fixture
def mock_diarization_result():
    """Mock diarization result."""
    return {
        "SPEAKER_00": {"start": 0.0, "end": 15.0},
        "SPEAKER_01": {"start": 10.0, "end": 25.0},
    }


@pytest.fixture
def mock_face_recognition_results():
    """Mock face recognition results."""
    return [
        {
            "timestamp": 0.0,
            "bbox": [100, 100, 200, 200],
            "face_id": "face_1",
            "name": "John Doe",
        },
        {
            "timestamp": 10.0,
            "bbox": [150, 150, 250, 250],
            "face_id": "face_2",
            "name": "Jane Smith",
        },
    ]


@pytest.fixture
def mock_non_speech_events():
    """Mock non-speech audio events."""
    return [
        {
            "start": 5.5,
            "end": 8.0,
            "label": "music",
            "confidence": 0.9,
            "duration": 2.5,
        }
    ]


# Pytest configuration for better test output
def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add custom markers if needed
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
