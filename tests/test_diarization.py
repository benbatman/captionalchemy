import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from captionalchemy.tools.audio_analysis.diarization import diarize


class TestDiarization:
    """Test suite for diarization functionality."""

    @patch.dict(os.environ, {}, clear=True)
    def test_diarize_missing_hf_token(self):
        """Test that diarize raises ValueError when HF_AUTH_TOKEN is missing."""
        with pytest.raises(
            ValueError, match="Hugging Face authentication token not found"
        ):
            diarize("dummy_audio.wav")

    @patch.dict(os.environ, {"HF_AUTH_TOKEN": ""}, clear=True)
    def test_diarize_empty_hf_token(self):
        """Test that diarize raises ValueError when HF_AUTH_TOKEN is empty."""
        with pytest.raises(
            ValueError, match="Hugging Face authentication token not found"
        ):
            diarize("dummy_audio.wav")

    @patch("captionalchemy.tools.audio_analysis.diarization.Pipeline")
    @patch("captionalchemy.tools.audio_analysis.diarization.torch")
    @patch.dict(os.environ, {"HF_AUTH_TOKEN": "test_token"}, clear=True)
    def test_diarize_success_cpu(self, mock_torch, mock_pipeline_class):
        """Test successful diarization on CPU."""
        # Mock torch.cuda.is_available() to return False (CPU)
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = "cpu"

        # Mock pipeline instance and its methods
        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        # Mock diarization result
        mock_turn1 = SimpleNamespace(start=1.0, end=5.0)
        mock_turn2 = SimpleNamespace(start=6.0, end=10.0)
        mock_turn3 = SimpleNamespace(start=11.0, end=15.0)

        mock_diarization = MagicMock()
        mock_diarization.itertracks.return_value = [
            (mock_turn1, None, "SPEAKER_00"),
            (mock_turn2, None, "SPEAKER_01"),
            (mock_turn3, None, "SPEAKER_00"),  # Same speaker continues
        ]
        mock_pipeline.return_value = mock_diarization

        result = diarize("test_audio.wav")

        # Verify pipeline was created correctly
        mock_pipeline_class.from_pretrained.assert_called_once_with(
            "pyannote/speaker-diarization-3.1", use_auth_token="test_token"
        )

        # Verify pipeline was moved to device
        mock_pipeline.to.assert_called_once_with("cpu")

        # Verify pipeline was called with audio path
        mock_pipeline.assert_called_once_with("test_audio.wav")

        # Verify result structure
        expected_result = {
            "SPEAKER_00": {
                "start": 1.0,
                "end": 15.0,
            },  # First speaker spans 1-5 and 11-15
            "SPEAKER_01": {"start": 6.0, "end": 10.0},  # Second speaker spans 6-10
        }
        assert result == expected_result

    @patch("captionalchemy.tools.audio_analysis.diarization.Pipeline")
    @patch("captionalchemy.tools.audio_analysis.diarization.torch")
    @patch.dict(os.environ, {"HF_AUTH_TOKEN": "test_token"}, clear=True)
    def test_diarize_success_gpu(self, mock_torch, mock_pipeline_class):
        """Test successful diarization on GPU."""
        # Mock torch.cuda.is_available() to return True (GPU)
        mock_torch.cuda.is_available.return_value = True
        mock_device = MagicMock()
        mock_torch.device.return_value = mock_device

        # Mock pipeline instance
        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        # Mock diarization result
        mock_turn = SimpleNamespace(start=2.5, end=8.7)
        mock_diarization = MagicMock()
        mock_diarization.itertracks.return_value = [
            (mock_turn, None, "SPEAKER_00"),
        ]
        mock_pipeline.return_value = mock_diarization

        result = diarize("test_audio.wav")

        # Verify GPU device was used
        mock_torch.device.assert_called_once_with("cuda")
        mock_pipeline.to.assert_called_once_with(mock_device)

        # Verify result
        expected_result = {"SPEAKER_00": {"start": 2.5, "end": 8.7}}
        assert result == expected_result

    @patch("captionalchemy.tools.audio_analysis.diarization.Pipeline")
    @patch("captionalchemy.tools.audio_analysis.diarization.torch")
    @patch.dict(os.environ, {"HF_AUTH_TOKEN": "test_token"}, clear=True)
    def test_diarize_multiple_speakers_complex(self, mock_torch, mock_pipeline_class):
        """Test diarization with multiple speakers and complex timing."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = "cpu"

        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        # Create complex scenario with overlapping and multiple segments
        mock_turns = [
            SimpleNamespace(start=0.0, end=3.0),  # SPEAKER_00
            SimpleNamespace(start=2.5, end=5.0),  # SPEAKER_01 (overlaps)
            SimpleNamespace(start=5.5, end=8.0),  # SPEAKER_00 (continues)
            SimpleNamespace(start=8.5, end=12.0),  # SPEAKER_02 (new speaker)
            SimpleNamespace(start=12.5, end=15.0),  # SPEAKER_01 (returns)
            SimpleNamespace(start=15.5, end=18.0),  # SPEAKER_00 (returns)
        ]

        mock_diarization = MagicMock()
        mock_diarization.itertracks.return_value = [
            (mock_turns[0], None, "SPEAKER_00"),
            (mock_turns[1], None, "SPEAKER_01"),
            (mock_turns[2], None, "SPEAKER_00"),
            (mock_turns[3], None, "SPEAKER_02"),
            (mock_turns[4], None, "SPEAKER_01"),
            (mock_turns[5], None, "SPEAKER_00"),
        ]
        mock_pipeline.return_value = mock_diarization

        result = diarize("complex_audio.wav")

        # The function extends end times for repeated speakers
        expected_result = {
            "SPEAKER_00": {
                "start": 0.0,
                "end": 18.0,
            },  # Start at 0.0, extended to final segment
            "SPEAKER_01": {
                "start": 2.5,
                "end": 15.0,
            },  # Start at 2.5, extended to final segment
            "SPEAKER_02": {"start": 8.5, "end": 12.0},  # Only one segment
        }
        assert result == expected_result

    @patch("captionalchemy.tools.audio_analysis.diarization.Pipeline")
    @patch("captionalchemy.tools.audio_analysis.diarization.torch")
    @patch.dict(os.environ, {"HF_AUTH_TOKEN": "test_token"}, clear=True)
    def test_diarize_single_speaker(self, mock_torch, mock_pipeline_class):
        """Test diarization with a single speaker."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = "cpu"

        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        # Single speaker with multiple segments
        mock_turns = [
            SimpleNamespace(start=1.0, end=5.0),
            SimpleNamespace(start=7.0, end=12.0),
            SimpleNamespace(start=15.0, end=20.0),
        ]

        mock_diarization = MagicMock()
        mock_diarization.itertracks.return_value = [
            (mock_turns[0], None, "SPEAKER_00"),
            (mock_turns[1], None, "SPEAKER_00"),
            (mock_turns[2], None, "SPEAKER_00"),
        ]
        mock_pipeline.return_value = mock_diarization

        result = diarize("single_speaker_audio.wav")

        expected_result = {"SPEAKER_00": {"start": 1.0, "end": 20.0}}
        assert result == expected_result

    @patch("captionalchemy.tools.audio_analysis.diarization.Pipeline")
    @patch("captionalchemy.tools.audio_analysis.diarization.torch")
    @patch.dict(os.environ, {"HF_AUTH_TOKEN": "test_token"}, clear=True)
    def test_diarize_no_speech(self, mock_torch, mock_pipeline_class):
        """Test diarization with no speech detected."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = "cpu"

        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        # No speech segments
        mock_diarization = MagicMock()
        mock_diarization.itertracks.return_value = []
        mock_pipeline.return_value = mock_diarization

        result = diarize("silent_audio.wav")

        assert result == {}

    @patch("captionalchemy.tools.audio_analysis.diarization.Pipeline")
    @patch("captionalchemy.tools.audio_analysis.diarization.torch")
    @patch.dict(os.environ, {"HF_AUTH_TOKEN": "test_token"}, clear=True)
    def test_diarize_pipeline_exception(self, mock_torch, mock_pipeline_class):
        """Test handling of pipeline exceptions."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = "cpu"

        # Mock pipeline to raise an exception
        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        mock_pipeline.side_effect = RuntimeError("Pipeline processing failed")

        with pytest.raises(RuntimeError, match="Pipeline processing failed"):
            diarize("problematic_audio.wav")

    @patch("captionalchemy.tools.audio_analysis.diarization.Pipeline")
    @patch("captionalchemy.tools.audio_analysis.diarization.torch")
    @patch.dict(os.environ, {"HF_AUTH_TOKEN": "test_token"}, clear=True)
    def test_diarize_pipeline_creation_exception(self, mock_torch, mock_pipeline_class):
        """Test handling of pipeline creation exceptions."""
        mock_torch.cuda.is_available.return_value = False

        # Mock pipeline creation to fail
        mock_pipeline_class.from_pretrained.side_effect = Exception(
            "Failed to load model"
        )

        with pytest.raises(Exception, match="Failed to load model"):
            diarize("audio.wav")

    @patch("captionalchemy.tools.audio_analysis.diarization.Pipeline")
    @patch("captionalchemy.tools.audio_analysis.diarization.torch")
    @patch.dict(os.environ, {"HF_AUTH_TOKEN": "test_token"}, clear=True)
    def test_diarize_with_real_file_path(self, mock_torch, mock_pipeline_class):
        """Test diarization with a real file path."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = "cpu"

        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        mock_turn = SimpleNamespace(start=0.0, end=10.0)
        mock_diarization = MagicMock()
        mock_diarization.itertracks.return_value = [
            (mock_turn, None, "SPEAKER_00"),
        ]
        mock_pipeline.return_value = mock_diarization

        # Use a real temporary file path
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            result = diarize(temp_path)

            # Verify the actual file path was passed to the pipeline
            mock_pipeline.assert_called_once_with(temp_path)

            expected_result = {"SPEAKER_00": {"start": 0.0, "end": 10.0}}
            assert result == expected_result

        finally:
            os.unlink(temp_path)

    @patch("captionalchemy.tools.audio_analysis.diarization.Pipeline")
    @patch("captionalchemy.tools.audio_analysis.diarization.torch")
    @patch.dict(os.environ, {"HF_AUTH_TOKEN": "hf_test123"}, clear=True)
    def test_diarize_auth_token_passed_correctly(self, mock_torch, mock_pipeline_class):
        """Test that the auth token is passed correctly to the pipeline."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = "cpu"

        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        mock_diarization = MagicMock()
        mock_diarization.itertracks.return_value = []
        mock_pipeline.return_value = mock_diarization

        diarize("test.wav")

        # Verify the correct auth token was used
        mock_pipeline_class.from_pretrained.assert_called_once_with(
            "pyannote/speaker-diarization-3.1", use_auth_token="hf_test123"
        )

    @patch("captionalchemy.tools.audio_analysis.diarization.Pipeline")
    @patch("captionalchemy.tools.audio_analysis.diarization.torch")
    @patch.dict(os.environ, {"HF_AUTH_TOKEN": "test_token"}, clear=True)
    def test_diarize_fractional_timestamps(self, mock_torch, mock_pipeline_class):
        """Test diarization with fractional timestamps."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = "cpu"

        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        # Use fractional timestamps
        mock_turns = [
            SimpleNamespace(start=1.234, end=5.678),
            SimpleNamespace(start=6.789, end=10.123),
        ]

        mock_diarization = MagicMock()
        mock_diarization.itertracks.return_value = [
            (mock_turns[0], None, "SPEAKER_00"),
            (mock_turns[1], None, "SPEAKER_01"),
        ]
        mock_pipeline.return_value = mock_diarization

        result = diarize("fractional_audio.wav")

        expected_result = {
            "SPEAKER_00": {"start": 1.234, "end": 5.678},
            "SPEAKER_01": {"start": 6.789, "end": 10.123},
        }
        assert result == expected_result

    @patch("captionalchemy.tools.audio_analysis.diarization.logger")
    @patch("captionalchemy.tools.audio_analysis.diarization.Pipeline")
    @patch("captionalchemy.tools.audio_analysis.diarization.torch")
    @patch.dict(os.environ, {"HF_AUTH_TOKEN": "test_token"}, clear=True)
    def test_diarize_logging(self, mock_torch, mock_pipeline_class, mock_logger):
        """Test that appropriate logging messages are generated."""
        mock_torch.cuda.is_available.return_value = True
        mock_device = MagicMock()
        mock_device.__str__ = MagicMock(return_value="cuda:0")
        mock_torch.device.return_value = mock_device

        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        mock_diarization = MagicMock()
        mock_diarization.itertracks.return_value = []
        mock_pipeline.return_value = mock_diarization

        diarize("test_audio.wav")

        # Check that logging messages were called
        mock_logger.info.assert_any_call(
            "Diarizing audio file: test_audio.wav on cuda:0..."
        )
        mock_logger.info.assert_any_call("Diarization completed.")
