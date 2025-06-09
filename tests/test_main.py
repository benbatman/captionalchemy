import unittest
import tempfile
import os
import json
import uuid
from unittest.mock import Mock, patch, MagicMock, call
from typing import List, Dict, Any
import numpy as np
import shutil


class TestMainPipelineIntegration(unittest.TestCase):
    def setUp(self):
        self.test_video_url = "test_video.mp4"
        self.temp_dir = tempfile.mkdtemp()
        self.setup_mocks()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def setup_mocks(self):
        """Set up mocks for pipeline testing."""
        self.mock_video_manager = Mock()
        self.mock_video_manager.get_video_data.return_value = (
            "/path/to/downloaded/video.mp4"
        )

        self.mock_extract_audio = Mock()

        # Mock speech segments (VAD)
        self.mock_speech_segments = [
            {"start": 0.0, "end": 5.0, "duration": 5.0},
            {"start": 10.0, "end": 15.0, "duration": 5.0},
            {"start": 20.0, "end": 25.0, "duration": 5.0},
        ]

        # Mock diarization result
        self.mock_diarization_result = {
            "SPEAKER_00": {"start": 0.0, "end": 15.0},
            "SPEAKER_01": {"start": 10.0, "end": 25.0},
        }

        # Mock non-speech events
        self.mock_non_speech_events = [
            {
                "start": 5.5,
                "end": 8.0,
                "label": "music",
                "confidence": 0.9,
                "duration": 2.5,
            }
        ]

        # Mock face recognition results
        self.mock_recognized_faces = [
            {
                "timesatmp": 0.0,
                "bbox": [100, 100, 200, 200],
                "face_id": "face_1",
                "name": "John Doe",
            },
            {
                "timesatmp": 10.0,
                "bbox": [150, 150, 250, 250],
                "face_id": "face_2",
                "name": "Jane Smith",
            },
        ]

        # Mock word timings from transcriber
        self.mock_word_timings = [
            {"word": "Hello", "start": 0.0, "end": 0.5, "duration": 0.5},
            {"word": "world", "start": 0.6, "end": 1.1, "duration": 0.5},
            {"word": ".", "start": 1.1, "end": 1.2, "duration": 0.1},
            {"word": "This", "start": 1.3, "end": 1.8, "duration": 0.5},
            {"word": "is", "start": 1.9, "end": 2.0, "duration": 0.1},
            {"word": "a", "start": 2.1, "end": 2.2, "duration": 0.1},
            {"word": "test", "start": 2.3, "end": 2.8, "duration": 0.5},
        ]

        # Mock subtitle segments from timing analyzer
        self.mock_subtitle_segments = [
            {
                "start": 0.0,
                "end": 1.2,
                "text": "Hello world.",
                "word_count": 2,
                "char_count": 12,
                "break_reason": "sentence_ending",
            }
        ]


@patch("tools.captioning.timing_analyzer.TimingAnalyzer")
@patch("tools.captioning.transcriber.Transcriber")
@patch("src.captionalchemy.tools.cv.recognize_faces.recognize_faces")
@patch("tools.audio_analysis.audio_segment_integration.integrate_audio_segments")
@patch("tools.audio_analysis.non_speech_detection.detect_non_speech_segments")
@patch("tools.audio_analysis.diarization.diarize")
@patch("tools.audio_analysis.vad.get_speech_segments")
@patch("src.captionalchemy.tools.media_utils.extract_audio.extract_audio")
@patch("src.captionalchemy.tools.media_utils.download_video.VideoManager")
@patch("src.captionalchemy.tools.cv.embed_known_faces.embed_faces")
@patch("whisper.load_model")
@patch("torch.cuda.is_available")
def test_complete_pipeline_success(
    self,
    mock_cuda,
    mock_whisper_load,
    mock_embed_faces,
    mock_video_manager_class,
    mock_extract_audio,
    mock_vad,
    mock_diarize,
    mock_non_speech,
    mock_integrate_audio,
    mock_recognize_faces,
    mock_transcriber_class,
    mock_timing_analyzer_class,
    mock_srt_writer_class,
):
    """Test the complete pipeline with successful execution."""
    mock_cuda.return_value = True

    mock_whisper_model = Mock()
    mock_whisper_load.return_value = mock_whisper_model

    mock_embed_faces.return_value = None

    mock_video_manager = Mock()
    mock_video_manager.get_video_data.return_value = "/tmp/video.mp4"
    mock_video_manager_class.return_value = mock_video_manager

    mock_extract_audio.return_value = None

    mock_vad.return_value = self.mock_speech_segments
    mock_diarize.return_value = self.mock_diarization_result
    mock_non_speech.return_value = self.mock_non_speech_events

    mock_integrated_events = []

    for segment in self.mock_speech_segments:
        mock_event = Mock()
        mock_event.event_type.value = "speech"
        mock_event.start = segment["start"]
        mock_event.end = segment["end"]
        mock_event.speaker_id = "SPEAKER_00" if segment["start"] < 15 else "SPEAKER_01"
        mock_integrated_events.append(mock_event)

    mock_integrate_audio.return_value = mock_integrated_events

    # Set up mock face recognition
    mock_recognize_faces.return_value = self.mock_recognized

    mock_transcriber = Mock()
    mock_transcriber.transcribe_audio.return_value = [
        Mock(**wt) for wt in self.mock_word_timings
    ]
    mock_transcriber_class.return_value = mock_transcriber

    mock_timing_analyzer = Mock()
    mock_timing_analyzer.suggest_subtitle_segments.return_value = [
        Mock(**seg) for seg in self.mock_subtitle_segments
    ]
    mock_timing_analyzer_class.return_value = mock_timing_analyzer

    # Mock SRT writer
    mock_srt_writer = Mock()
    mock_srt_writer_class.return_value = mock_srt_writer

    # Create simplified mock of main function
