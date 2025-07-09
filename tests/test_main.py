import tempfile
import os
import pytest
from unittest.mock import Mock, patch
import shutil
import logging

from captionalchemy.caption import run_pipeline, _build_arg_parser, main


@pytest.fixture
def temp_dir():
    """Create and cleanup temporary directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_video_data():
    """Mock video-related test data."""
    return {"url": "test_video.mp4", "path": "/path/to/downloaded/video.mp4"}


@pytest.fixture
def mock_speech_segments():
    """Mock speech segments (VAD)."""
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
def mock_non_speech_events():
    """Mock non-speech events."""
    return [
        {
            "start": 5.5,
            "end": 8.0,
            "label": "music",
            "confidence": 0.9,
            "duration": 2.5,
        }
    ]


@pytest.fixture
def mock_recognized_faces():
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
def mock_word_timings():
    """Mock word timings from transcriber."""
    return [
        Mock(word="Hello", start=0.0, end=0.5, duration=0.5),
        Mock(word="world", start=0.6, end=1.1, duration=0.5),
        Mock(word=".", start=1.1, end=1.2, duration=0.1),
        Mock(word="This", start=1.3, end=1.8, duration=0.5),
        Mock(word="is", start=1.9, end=2.0, duration=0.1),
        Mock(word="a", start=2.1, end=2.2, duration=0.1),
        Mock(word="test", start=2.3, end=2.8, duration=0.5),
    ]


@pytest.fixture
def mock_subtitle_segments():
    """Mock subtitle segments from timing analyzer."""
    return [
        Mock(
            start=0.0,
            end=1.2,
            text="Hello world.",
            word_count=2,
            char_count=12,
            break_reason="sentence_ending",
        ),
        Mock(
            start=1.3,
            end=2.8,
            text="This is a test.",
            word_count=4,
            char_count=15,
            break_reason="duration_limit",
        ),
    ]


@pytest.fixture
def mock_integrated_events(mock_speech_segments):
    """Mock integrated audio events."""
    events = []
    for i, segment in enumerate(mock_speech_segments):
        event = Mock()
        event.event_type = Mock()
        event.event_type.value = "speech"
        event.start = segment["start"]
        event.end = segment["end"]
        event.speaker_id = "SPEAKER_00" if segment["start"] < 15 else "SPEAKER_01"
        events.append(event)

    # Add a non-speech event
    music_event = Mock()
    music_event.event_type = Mock()
    music_event.event_type.value = "music"
    music_event.start = 5.5
    music_event.end = 8.0
    events.append(music_event)

    return events


class TestRunPipeline:
    """Test cases for the main run_pipeline function."""

    @patch("captionalchemy.caption.SRTCaptionWriter")
    @patch("captionalchemy.caption.TimingAnalyzer")
    @patch("captionalchemy.caption.Transcriber")
    @patch("captionalchemy.caption.recognize_faces")
    @patch("captionalchemy.caption.integrate_audio_segments")
    @patch("captionalchemy.caption.detect_non_speech_segments")
    @patch("captionalchemy.caption.diarize")
    @patch("captionalchemy.caption.get_speech_segments")
    @patch("captionalchemy.caption.extract_audio")
    @patch("captionalchemy.caption.VideoManager")
    @patch("captionalchemy.caption.embed_faces")
    @patch("captionalchemy.caption.whisper.load_model")
    @patch("captionalchemy.caption.torch.cuda.is_available")
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
        mock_video_data,
        mock_speech_segments,
        mock_diarization_result,
        mock_non_speech_events,
        mock_recognized_faces,
        mock_word_timings,
        mock_subtitle_segments,
        mock_integrated_events,
    ):
        """Test the complete pipeline with successful execution."""

        # Setup mocks
        mock_cuda.return_value = True

        mock_whisper_model = Mock()
        mock_whisper_load.return_value = mock_whisper_model

        mock_embed_faces.return_value = None

        mock_video_manager = Mock()
        mock_video_manager.get_video_data.return_value = mock_video_data["path"]
        mock_video_manager_class.return_value = mock_video_manager

        mock_extract_audio.return_value = None

        mock_vad.return_value = mock_speech_segments
        mock_diarize.return_value = mock_diarization_result
        mock_non_speech.return_value = mock_non_speech_events

        mock_integrate_audio.return_value = mock_integrated_events

        mock_recognize_faces.return_value = mock_recognized_faces

        mock_transcriber = Mock()
        mock_transcriber.transcribe_audio.return_value = mock_word_timings
        mock_transcriber_class.return_value = mock_transcriber

        mock_timing_analyzer = Mock()
        mock_timing_analyzer.suggest_subtitle_segments.return_value = (
            mock_subtitle_segments
        )
        mock_timing_analyzer_class.return_value = mock_timing_analyzer

        # Mock SRT writer
        mock_srt_writer = Mock()
        mock_srt_writer_class.return_value = mock_srt_writer

        # Run the pipeline
        run_pipeline(
            video_url_or_path=mock_video_data["url"],
            character_identification=True,
            known_faces_json="test_known_faces.json",
            embed_faces_json="test_embed_faces.json",
            caption_output_path="test_output",
            caption_format="srt",
        )

        # Verify key function calls
        mock_embed_faces.assert_called_once_with(
            "test_known_faces.json", "test_embed_faces.json"
        )
        mock_video_manager_class.assert_called_once_with(use_file_buffer=False)
        mock_whisper_load.assert_called_once()
        mock_vad.assert_called_once()
        mock_diarize.assert_called_once()
        mock_non_speech.assert_called_once()
        mock_integrate_audio.assert_called_once()

        # Verify transcriber was called for speech events
        speech_events = [
            e for e in mock_integrated_events if e.event_type.value == "speech"
        ]
        assert mock_transcriber.transcribe_audio.call_count == len(speech_events)

        # Verify timing analyzer was called for speech events
        assert mock_timing_analyzer.suggest_subtitle_segments.call_count == len(
            speech_events
        )

        # Verify writer methods were called
        mock_srt_writer.write.assert_called_once_with("test_output.srt")

    @patch("captionalchemy.caption.detect_non_speech_segments")
    @patch("captionalchemy.caption.embed_faces")
    @patch("captionalchemy.caption.VideoManager")
    @patch("captionalchemy.caption.extract_audio")
    @patch("captionalchemy.caption.get_speech_segments")
    @patch("captionalchemy.caption.torch.cuda.is_available")
    @patch("captionalchemy.caption.whisper.load_model")
    def test_pipeline_no_speech_segments(
        self,
        mock_whisper_load,
        mock_cuda,
        mock_vad,
        mock_extract_audio,
        mock_video_manager_class,
        mock_embed_faces,
        mock_detect_non_speech,
        mock_video_data,
    ):
        """Test pipeline behavior when no speech segments are detected."""

        mock_cuda.return_value = False
        mock_embed_faces.return_value = None
        mock_whisper_load.return_value = Mock()

        mock_video_manager = Mock()
        mock_video_manager.get_video_data.return_value = mock_video_data["path"]
        mock_video_manager_class.return_value = mock_video_manager

        mock_extract_audio.return_value = None

        # Return empty speech segments
        mock_vad.return_value = []
        mock_detect_non_speech.return_value = []

        # This should complete without error, just log a warning
        run_pipeline(
            video_url_or_path=mock_video_data["url"],
            character_identification=False,
        )

        # Verify mock_embed_faces was not called
        mock_embed_faces.assert_not_called()
        # Verify early functions were called
        mock_vad.assert_called_once()

    @patch("captionalchemy.caption.detect_non_speech_segments")
    @patch("captionalchemy.caption.VTTCaptionWriter")
    @patch("captionalchemy.caption.embed_faces")
    @patch("captionalchemy.caption.VideoManager")
    @patch("captionalchemy.caption.extract_audio")
    @patch("captionalchemy.caption.get_speech_segments")
    @patch("captionalchemy.caption.torch.cuda.is_available")
    @patch("captionalchemy.caption.whisper.load_model")
    def test_pipeline_vtt_format(
        self,
        mock_whisper_load,
        mock_cuda,
        mock_vad,
        mock_extract_audio,
        mock_video_manager_class,
        mock_embed_faces,
        mock_vtt_writer_class,
        mock_detect_non_speech,
        mock_video_data,
    ):
        """Test pipeline with VTT caption format."""

        mock_cuda.return_value = False
        mock_embed_faces.return_value = None
        mock_whisper_load.return_value = Mock()

        mock_video_manager = Mock()
        mock_video_manager.get_video_data.return_value = mock_video_data["path"]
        mock_video_manager_class.return_value = mock_video_manager

        mock_extract_audio.return_value = None
        mock_vad.return_value = []  # No speech to simplify test
        mock_detect_non_speech.return_value = []

        mock_vtt_writer = Mock()
        mock_vtt_writer_class.return_value = mock_vtt_writer

        run_pipeline(
            video_url_or_path=mock_video_data["url"],
            character_identification=False,
            caption_format="vtt",
        )

        # Verify VTT writer was instantiated
        mock_vtt_writer_class.assert_called_once()

    @patch("captionalchemy.caption.detect_non_speech_segments")
    @patch("captionalchemy.caption.SAMICaptionWriter")
    @patch("captionalchemy.caption.embed_faces")
    @patch("captionalchemy.caption.VideoManager")
    @patch("captionalchemy.caption.extract_audio")
    @patch("captionalchemy.caption.get_speech_segments")
    @patch("captionalchemy.caption.torch.cuda.is_available")
    @patch("captionalchemy.caption.whisper.load_model")
    def test_pipeline_sami_format(
        self,
        mock_whisper_load,
        mock_cuda,
        mock_vad,
        mock_extract_audio,
        mock_video_manager_class,
        mock_embed_faces,
        mock_sami_writer_class,
        mock_detect_non_speech,
        mock_video_data,
    ):
        """Test pipeline with SAMI caption format."""

        mock_cuda.return_value = False
        mock_embed_faces.return_value = None
        mock_whisper_load.return_value = Mock()

        mock_video_manager = Mock()
        mock_video_manager.get_video_data.return_value = mock_video_data["path"]
        mock_video_manager_class.return_value = mock_video_manager

        mock_extract_audio.return_value = None
        mock_vad.return_value = []
        mock_detect_non_speech.return_value = []

        mock_sami_writer = Mock()
        mock_sami_writer_class.return_value = mock_sami_writer

        run_pipeline(
            video_url_or_path=mock_video_data["url"],
            character_identification=False,
            caption_format="smi",
        )

        mock_sami_writer_class.assert_called_once()

    def test_pipeline_with_existing_video_file(self, tmp_path):
        """Test pipeline when video_url_or_path is an existing file."""

        # Create a dummy video file
        video_file = tmp_path / "test_video.mp4"
        video_file.write_text("dummy video content")

        with (
            patch("captionalchemy.caption.embed_faces"),
            patch("captionalchemy.caption.VideoManager") as mock_vm_class,
            patch("captionalchemy.caption.extract_audio"),
            patch("captionalchemy.caption.get_speech_segments", return_value=[]),
            patch("captionalchemy.caption.detect_non_speech_segments", return_value=[]),
            patch("captionalchemy.caption.torch.cuda.is_available", return_value=False),
            patch("captionalchemy.caption.whisper.load_model"),
        ):

            mock_vm = Mock()
            mock_vm_class.return_value = mock_vm

            run_pipeline(
                video_url_or_path=str(video_file),
                character_identification=False,
            )

            # Should not call get_video_data since file exists
            mock_vm.get_video_data.assert_not_called()

    @patch("captionalchemy.caption.SRTCaptionWriter")
    @patch("captionalchemy.caption.TimingAnalyzer")
    @patch("captionalchemy.caption.Transcriber")
    @patch("captionalchemy.caption.integrate_audio_segments")
    @patch("captionalchemy.caption.detect_non_speech_segments")
    @patch("captionalchemy.caption.diarize")
    @patch("captionalchemy.caption.get_speech_segments")
    @patch("captionalchemy.caption.extract_audio")
    @patch("captionalchemy.caption.VideoManager")
    @patch("captionalchemy.caption.embed_faces")
    @patch("captionalchemy.caption.whisper.load_model")
    @patch("captionalchemy.caption.torch.cuda.is_available")
    def test_pipeline_character_identification_disabled(
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
        mock_transcriber_class,
        mock_timing_analyzer_class,
        mock_srt_writer_class,
        mock_speech_segments,
        mock_integrated_events,
        mock_word_timings,
        mock_subtitle_segments,
    ):
        """Test pipeline with character identification disabled."""

        # Setup basic mocks
        mock_cuda.return_value = False
        mock_whisper_load.return_value = Mock()
        mock_embed_faces.return_value = None

        mock_vm = Mock()
        mock_vm.get_video_data.return_value = "/tmp/video.mp4"
        mock_video_manager_class.return_value = mock_vm

        mock_extract_audio.return_value = None
        mock_vad.return_value = mock_speech_segments
        mock_diarize.return_value = {"SPEAKER_00": {"start": 0.0, "end": 30.0}}
        mock_non_speech.return_value = []

        # Only speech events for this test
        speech_events = [
            e for e in mock_integrated_events if e.event_type.value == "speech"
        ]
        mock_integrate_audio.return_value = speech_events

        mock_transcriber = Mock()
        mock_transcriber.transcribe_audio.return_value = mock_word_timings
        mock_transcriber_class.return_value = mock_transcriber

        mock_timing_analyzer = Mock()
        mock_timing_analyzer.suggest_subtitle_segments.return_value = (
            mock_subtitle_segments
        )
        mock_timing_analyzer_class.return_value = mock_timing_analyzer

        mock_writer = Mock()
        mock_srt_writer_class.return_value = mock_writer

        run_pipeline(
            video_url_or_path="test_video.mp4",
            character_identification=False,
        )

        # embed_faces should not be called
        mock_embed_faces.assert_not_called()

        # Should process speech events even without face recognition
        assert mock_transcriber.transcribe_audio.call_count == len(speech_events)

    @patch("captionalchemy.caption.SRTCaptionWriter")
    @patch("captionalchemy.caption.TimingAnalyzer")
    @patch("captionalchemy.caption.Transcriber")
    @patch("captionalchemy.caption.recognize_faces")
    @patch("captionalchemy.caption.integrate_audio_segments")
    @patch("captionalchemy.caption.detect_non_speech_segments")
    @patch("captionalchemy.caption.diarize")
    @patch("captionalchemy.caption.get_speech_segments")
    @patch("captionalchemy.caption.extract_audio")
    @patch("captionalchemy.caption.VideoManager")
    @patch("captionalchemy.caption.embed_faces")
    @patch("captionalchemy.caption.whisper.load_model")
    @patch("captionalchemy.caption.torch.cuda.is_available")
    def test_pipeline_speaker_name_mapping(
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
        mock_speech_segments,
        mock_word_timings,
        mock_subtitle_segments,
    ):
        """Test that speaker names are properly mapped and reused."""

        # Setup mocks
        mock_cuda.return_value = True
        mock_whisper_load.return_value = Mock()
        mock_embed_faces.return_value = None

        mock_vm = Mock()
        mock_vm.get_video_data.return_value = "/tmp/video.mp4"
        mock_video_manager_class.return_value = mock_vm

        mock_extract_audio.return_value = None
        mock_vad.return_value = mock_speech_segments
        mock_diarize.return_value = {
            "SPEAKER_00": {"start": 0.0, "end": 15.0},
            "SPEAKER_01": {"start": 15.0, "end": 30.0},
        }
        mock_non_speech.return_value = []

        # Create events with same speaker appearing multiple times
        events = []
        for i, segment in enumerate(mock_speech_segments):
            event = Mock()
            event.event_type = Mock()
            event.event_type.value = "speech"
            event.start = segment["start"]
            event.end = segment["end"]
            event.speaker_id = "SPEAKER_00"  # Same speaker for all
            events.append(event)

        mock_integrate_audio.return_value = events

        # Mock face recognition to return same name
        mock_recognize_faces.return_value = [{"name": "John Doe"}]

        mock_transcriber = Mock()
        mock_transcriber.transcribe_audio.return_value = mock_word_timings
        mock_transcriber_class.return_value = mock_transcriber

        mock_timing_analyzer = Mock()
        mock_timing_analyzer.suggest_subtitle_segments.return_value = (
            mock_subtitle_segments
        )
        mock_timing_analyzer_class.return_value = mock_timing_analyzer

        mock_writer = Mock()
        mock_srt_writer_class.return_value = mock_writer

        run_pipeline(
            video_url_or_path="test_video.mp4",
            character_identification=True,
        )

        # Face recognition should only be called once per unique speaker
        assert mock_recognize_faces.call_count == 1


class TestArgumentParser:
    """Test cases for the command line argument parser."""

    def test_build_arg_parser_default_values(self):
        """Test that argument parser has correct default values."""

        parser = _build_arg_parser()
        args = parser.parse_args(["test_video.mp4"])

        assert args.video == "test_video.mp4"
        assert args.format == "srt"
        assert args.output == "output_captions"
        assert args.character_identification is True
        assert args.known_faces_json == "known_faces.json"
        assert args.embed_faces_json == "embed_faces.json"
        assert args.verbose is False

    def test_build_arg_parser_custom_values(self):
        """Test argument parser with custom values."""

        parser = _build_arg_parser()
        args = parser.parse_args(
            [
                "https://example.com/video.mp4",
                "--format",
                "vtt",
                "--output",
                "my_captions",
                "--no-face-id",
                "--known-faces-json",
                "my_faces.json",
                "--embed-faces-json",
                "my_embeddings.json",
                "--verbose",
            ]
        )

        assert args.video == "https://example.com/video.mp4"
        assert args.format == "vtt"
        assert args.output == "my_captions"
        assert args.character_identification is False
        assert args.known_faces_json == "my_faces.json"
        assert args.embed_faces_json == "my_embeddings.json"
        assert args.verbose is True

    def test_build_arg_parser_format_choices(self):
        """Test that format argument only accepts valid choices."""

        parser = _build_arg_parser()

        # Valid formats should work
        for fmt in ["srt", "vtt", "smi"]:
            args = parser.parse_args(["video.mp4", "--format", fmt])
            assert args.format == fmt

        # Invalid format should raise SystemExit
        with pytest.raises(SystemExit):
            parser.parse_args(["video.mp4", "--format", "invalid"])

    def test_build_arg_parser_help_text(self):
        """Test that help text is properly configured."""

        parser = _build_arg_parser()

        assert parser.prog == "captionalchemy"
        assert "Download/extract audio from a video" in parser.description


class TestMainFunction:
    """Test cases for the main() function."""

    @patch("captionalchemy.caption.run_pipeline")
    @patch("captionalchemy.caption.logging.getLogger")
    @patch("captionalchemy.caption.load_dotenv")
    def test_main_with_default_args(
        self, mock_load_dotenv, mock_get_logger, mock_run_pipeline
    ):
        """Test main function with default arguments."""

        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        with patch("sys.argv", ["captionalchemy", "test_video.mp4"]):
            main()

        mock_load_dotenv.assert_called_once()
        mock_run_pipeline.assert_called_once_with(
            video_url_or_path="test_video.mp4",
            character_identification=True,
            known_faces_json="known_faces.json",
            embed_faces_json="embed_faces.json",
            caption_output_path="output_captions",
            caption_format="srt",
        )

    @patch("captionalchemy.caption.run_pipeline")
    @patch("captionalchemy.caption.logging.getLogger")
    @patch("captionalchemy.caption.load_dotenv")
    def test_main_with_verbose_logging(
        self, mock_load_dotenv, mock_get_logger, mock_run_pipeline
    ):
        """Test main function with verbose logging enabled."""

        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        with (
            patch("sys.argv", ["captionalchemy", "video.mp4", "--verbose"]),
            patch("captionalchemy.caption.logging.getLogger") as mock_root_logger,
        ):

            main()

            # Should set logging level to DEBUG
            mock_root_logger().setLevel.assert_called_with(logging.DEBUG)

    @patch("captionalchemy.caption.run_pipeline")
    @patch("captionalchemy.caption.logging.getLogger")
    @patch("captionalchemy.caption.load_dotenv")
    def test_main_with_custom_args(
        self, mock_load_dotenv, mock_get_logger, mock_run_pipeline
    ):
        """Test main function with custom arguments."""

        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        test_args = [
            "captionalchemy",
            "https://example.com/video.mp4",
            "--format",
            "vtt",
            "--output",
            "custom_output",
            "--no-face-id",
            "--known-faces-json",
            "custom_faces.json",
            "--embed-faces-json",
            "custom_embeddings.json",
        ]

        with patch("sys.argv", test_args):
            main()

        mock_run_pipeline.assert_called_once_with(
            video_url_or_path="https://example.com/video.mp4",
            character_identification=False,
            known_faces_json="custom_faces.json",
            embed_faces_json="custom_embeddings.json",
            caption_output_path="custom_output",
            caption_format="vtt",
        )

    @patch("captionalchemy.caption.run_pipeline")
    @patch("captionalchemy.caption.logging.getLogger")
    @patch("captionalchemy.caption.load_dotenv")
    def test_main_info_logging_default(
        self, mock_load_dotenv, mock_get_logger, mock_run_pipeline
    ):
        """Test that main sets INFO logging by default."""

        with (
            patch("sys.argv", ["captionalchemy", "video.mp4"]),
            patch("captionalchemy.caption.logging.getLogger") as mock_root_logger,
        ):

            main()

            # Should set logging level to INFO by default
            mock_root_logger().setLevel.assert_called_with(logging.INFO)


class TestIntegration:
    """Integration tests for the complete pipeline."""

    @patch.dict(os.environ, {"HF_AUTH_TOKEN": "test_token"})
    def test_pipeline_environment_variables(self):
        """Test that environment variables are properly used."""

        with (
            patch("captionalchemy.caption.embed_faces"),
            patch("captionalchemy.caption.VideoManager"),
            patch("captionalchemy.caption.extract_audio"),
            patch("captionalchemy.caption.get_speech_segments") as mock_vad,
            patch("captionalchemy.caption.detect_non_speech_segments", return_value=[]),
            patch("captionalchemy.caption.torch.cuda.is_available", return_value=False),
            patch("captionalchemy.caption.whisper.load_model"),
        ):

            mock_vad.return_value = []

            run_pipeline("test_video.mp4")

            # Should pass environment token to VAD
            mock_vad.assert_called_once()
            call_args = mock_vad.call_args
            assert call_args[0][1] == "test_token"  # HF_AUTH_TOKEN

    def test_pipeline_temp_directory_cleanup(self, tmp_path):
        """Test that temporary directories are properly cleaned up."""

        # Track created temp directories
        temp_dirs_created = []

        class MockTempDir:
            def __init__(self):
                self.name = str(tmp_path / f"temp_{len(temp_dirs_created)}")
                os.makedirs(self.name, exist_ok=True)
                temp_dirs_created.append(self.name)

            def __enter__(self):
                return self.name

            def __exit__(self, *args):
                if os.path.exists(self.name):
                    shutil.rmtree(self.name)

        with (
            patch("tempfile.TemporaryDirectory", MockTempDir),
            patch("captionalchemy.caption.embed_faces"),
            patch("captionalchemy.caption.VideoManager"),
            patch("captionalchemy.caption.extract_audio"),
            patch("captionalchemy.caption.get_speech_segments", return_value=[]),
            patch("captionalchemy.caption.detect_non_speech_segments", return_value=[]),
            patch("captionalchemy.caption.torch.cuda.is_available", return_value=False),
            patch("captionalchemy.caption.whisper.load_model"),
        ):

            run_pipeline("test_video.mp4")

            # All temp directories should be cleaned up
            for temp_dir in temp_dirs_created:
                assert not os.path.exists(temp_dir)
