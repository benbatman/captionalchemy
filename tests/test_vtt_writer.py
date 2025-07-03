import tempfile
import os
from unittest.mock import patch, mock_open

from captionalchemy.tools.captioning.writers.vtt_writer import VTTCaptionWriter


class TestVTTCaptionWriter:
    """Test suite for VTTCaptionWriter class."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        writer = VTTCaptionWriter()

        assert writer.min_silence_duration == 1.0
        assert writer.min_music_duration == 1.5
        assert writer.last_speaker is None
        assert writer._captions == []

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        writer = VTTCaptionWriter(min_silence_duration=2.0, min_music_duration=3.0)

        assert writer.min_silence_duration == 2.0
        assert writer.min_music_duration == 3.0

    def test_add_caption_with_text(self):
        """Test adding a caption with text."""
        writer = VTTCaptionWriter()
        writer.add_caption(
            start=1.0,
            end=3.0,
            speaker="John Doe",
            text="Hello world",
            event_type="speech",
        )

        assert len(writer._captions) == 1
        caption = writer._captions[0]
        assert caption["start"] == 1.0
        assert caption["end"] == 3.0
        assert caption["speaker"] == "John Doe"
        assert caption["text"] == "Hello world"
        assert caption["event_type"] == "speech"

    def test_add_caption_with_newlines(self):
        """Test adding a caption with newlines that should be replaced."""
        writer = VTTCaptionWriter()
        writer.add_caption(
            start=1.0, end=3.0, text="Hello\nworld\ntest", event_type="speech"
        )

        caption = writer._captions[0]
        assert caption["text"] == "Hello world test"

    def test_add_caption_music_long_duration(self):
        """Test adding a music caption with long duration."""
        writer = VTTCaptionWriter()
        writer.add_caption(
            start=1.0,
            end=3.0,  # 2 seconds, > min_music_duration (1.5)
            event_type="music",
        )

        assert len(writer._captions) == 1
        caption = writer._captions[0]
        assert caption["text"] == "[MUSIC PLAYING]"

    def test_add_caption_music_short_duration(self):
        """Test adding a music caption with short duration (should be skipped)."""
        writer = VTTCaptionWriter()
        writer.add_caption(
            start=1.0,
            end=2.0,  # 1 second, < min_music_duration (1.5)
            event_type="music",
        )

        assert len(writer._captions) == 0

    def test_add_caption_silence_long_duration(self):
        """Test adding a silence caption with long duration."""
        writer = VTTCaptionWriter()
        writer.add_caption(
            start=1.0,
            end=3.0,  # 2 seconds, > min_silence_duration (1.0)
            event_type="silence",
        )

        assert len(writer._captions) == 1
        caption = writer._captions[0]
        assert caption["text"] == "[SILENCE]"

    def test_add_caption_silence_short_duration(self):
        """Test adding a silence caption with short duration (should be skipped)."""
        writer = VTTCaptionWriter()
        writer.add_caption(
            start=1.0,
            end=1.5,  # 0.5 seconds, < min_silence_duration (1.0)
            event_type="silence",
        )

        assert len(writer._captions) == 0

    def test_add_caption_with_label(self):
        """Test adding a caption with a label."""
        writer = VTTCaptionWriter()
        writer.add_caption(start=1.0, end=3.0, event_type="other", label="applause")

        assert len(writer._captions) == 1
        caption = writer._captions[0]
        assert caption["text"] == "[APPLAUSE]"

    def test_add_caption_fallback(self):
        """Test adding a caption with fallback text."""
        writer = VTTCaptionWriter()
        writer.add_caption(start=1.0, end=3.0, event_type="unknown")

        assert len(writer._captions) == 1
        caption = writer._captions[0]
        assert caption["text"] == "[AUDIO]"

    def test_format_timestamp_zero(self):
        """Test timestamp formatting for zero seconds."""
        result = VTTCaptionWriter._format_timestamp(0.0)
        assert result == "00:00:00.000"

    def test_format_timestamp_basic(self):
        """Test basic timestamp formatting."""
        result = VTTCaptionWriter._format_timestamp(65.5)
        assert result == "00:01:05.500"

    def test_format_timestamp_with_hours(self):
        """Test timestamp formatting with hours."""
        result = VTTCaptionWriter._format_timestamp(3665.123)
        assert result == "01:01:05.123"

    def test_format_timestamp_edge_cases(self):
        """Test timestamp formatting edge cases."""
        # Test milliseconds rounding
        result = VTTCaptionWriter._format_timestamp(1.9999)
        assert result == "00:00:01.999"

        # Test very small time
        result = VTTCaptionWriter._format_timestamp(0.001)
        assert result == "00:00:00.001"

        # Test large time
        result = VTTCaptionWriter._format_timestamp(7323.456)
        assert result == "02:02:03.456"

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    def test_write_empty_captions(self, mock_makedirs, mock_file):
        """Test writing with no captions."""
        writer = VTTCaptionWriter()
        writer.write("test.vtt")

        mock_makedirs.assert_called_once_with(".", exist_ok=True)
        mock_file.assert_called_once_with("test.vtt", "w", encoding="utf-8")

        # Check that the WebVTT header is written
        handle = mock_file.return_value
        written_content = "".join(call[0][0] for call in handle.write.call_args_list)

        assert written_content == "WEBVTT\n\n"

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    def test_write_with_speech_captions(self, mock_makedirs, mock_file):
        """Test writing with speech captions."""
        writer = VTTCaptionWriter()
        writer.add_caption(
            start=1.0,
            end=3.0,
            speaker="John Doe",
            text="Hello world",
            event_type="speech",
        )
        writer.add_caption(
            start=4.0,
            end=6.0,
            speaker="Jane Smith",
            text="Hi there",
            event_type="speech",
        )

        writer.write("test.vtt")

        handle = mock_file.return_value
        written_content = "".join(call[0][0] for call in handle.write.call_args_list)

        # Check for WebVTT header
        assert "WEBVTT" in written_content

        # Check for timestamps
        assert "00:00:01.000 --> 00:00:03.000" in written_content
        assert "00:00:04.000 --> 00:00:06.000" in written_content

        # Check for voice tags
        assert "<v John Doe>Hello world</v>" in written_content
        assert "<v Jane Smith>Hi there</v>" in written_content

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    def test_write_with_same_speaker_consecutive(self, mock_makedirs, mock_file):
        """Test writing with consecutive captions from the same speaker."""
        writer = VTTCaptionWriter()
        writer.add_caption(
            start=1.0,
            end=3.0,
            speaker="John Doe",
            text="Hello world",
            event_type="speech",
        )
        writer.add_caption(
            start=4.0,
            end=6.0,
            speaker="John Doe",
            text="How are you?",
            event_type="speech",
        )

        writer.write("test.vtt")

        handle = mock_file.return_value
        written_content = "".join(call[0][0] for call in handle.write.call_args_list)

        # Check that voice tags are still used for consistency
        assert "<v John Doe>Hello world</v>" in written_content
        assert "<v John Doe>How are you?</v>" in written_content

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    def test_write_with_non_speech_captions(self, mock_makedirs, mock_file):
        """Test writing with non-speech captions."""
        writer = VTTCaptionWriter()
        writer.add_caption(start=1.0, end=3.0, event_type="music")
        writer.add_caption(start=4.0, end=6.0, event_type="silence")

        writer.write("test.vtt")

        handle = mock_file.return_value
        written_content = "".join(call[0][0] for call in handle.write.call_args_list)

        # Check for timestamps
        assert "00:00:01.000 --> 00:00:03.000" in written_content
        assert "00:00:04.000 --> 00:00:06.000" in written_content

        # Check for event descriptions (no voice tags)
        assert "[MUSIC PLAYING]" in written_content
        assert "[SILENCE]" in written_content

        # Ensure no voice tags are used for non-speech
        assert "<v" not in written_content

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    def test_write_sorting(self, mock_makedirs, mock_file):
        """Test that captions are sorted by start time."""
        writer = VTTCaptionWriter()
        writer.add_caption(start=3.0, end=4.0, text="Third", event_type="speech")
        writer.add_caption(start=1.0, end=2.0, text="First", event_type="speech")
        writer.add_caption(start=2.0, end=3.0, text="Second", event_type="speech")

        writer.write("test.vtt")

        handle = mock_file.return_value
        written_content = "".join(call[0][0] for call in handle.write.call_args_list)

        # Find positions of the timestamps
        first_pos = written_content.find("00:00:01.000 --> 00:00:02.000")
        second_pos = written_content.find("00:00:02.000 --> 00:00:03.000")
        third_pos = written_content.find("00:00:03.000 --> 00:00:04.000")

        assert first_pos < second_pos < third_pos

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    def test_write_mixed_content(self, mock_makedirs, mock_file):
        """Test writing with mixed speech and non-speech content."""
        writer = VTTCaptionWriter()
        writer.add_caption(
            start=1.0, end=3.0, speaker="John Doe", text="Hello", event_type="speech"
        )
        writer.add_caption(start=4.0, end=6.0, event_type="music")
        writer.add_caption(
            start=7.0,
            end=9.0,
            speaker="Jane Smith",
            text="Goodbye",
            event_type="speech",
        )

        writer.write("test.vtt")

        handle = mock_file.return_value
        written_content = "".join(call[0][0] for call in handle.write.call_args_list)

        # Check for mixed content
        assert "<v John Doe>Hello</v>" in written_content
        assert "[MUSIC PLAYING]" in written_content
        assert "<v Jane Smith>Goodbye</v>" in written_content

    def test_write_creates_directory(self):
        """Test that write creates the output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "subdir", "test.vtt")
            writer = VTTCaptionWriter()
            writer.add_caption(start=1.0, end=2.0, text="Test", event_type="speech")

            writer.write(output_path)

            # Check that the directory was created
            assert os.path.exists(os.path.dirname(output_path))
            assert os.path.exists(output_path)

    def test_write_real_file(self):
        """Test writing to a real file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".vtt", delete=False
        ) as temp_file:
            temp_path = temp_file.name

        try:
            writer = VTTCaptionWriter()
            writer.add_caption(
                start=1.0,
                end=3.0,
                speaker="John Doe",
                text="Hello world",
                event_type="speech",
            )
            writer.add_caption(start=4.0, end=6.0, event_type="music")

            writer.write(temp_path)

            # Read the file and check content
            with open(temp_path, "r", encoding="utf-8") as f:
                content = f.read()

            assert "WEBVTT" in content
            assert "00:00:01.000 --> 00:00:03.000" in content
            assert "00:00:04.000 --> 00:00:06.000" in content
            assert "<v John Doe>Hello world</v>" in content
            assert "[MUSIC PLAYING]" in content

        finally:
            os.unlink(temp_path)

    def test_write_with_special_characters(self):
        """Test writing with special characters in text."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".vtt", delete=False
        ) as temp_file:
            temp_path = temp_file.name

        try:
            writer = VTTCaptionWriter()
            writer.add_caption(
                start=1.0,
                end=3.0,
                speaker="José María",
                text="Hola, ¿cómo estás?",
                event_type="speech",
            )

            writer.write(temp_path)

            # Read the file and check content
            with open(temp_path, "r", encoding="utf-8") as f:
                content = f.read()

            assert "José María" in content
            assert "Hola, ¿cómo estás?" in content

        finally:
            os.unlink(temp_path)

    def test_write_with_long_timestamps(self):
        """Test writing with long timestamps (hours)."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".vtt", delete=False
        ) as temp_file:
            temp_path = temp_file.name

        try:
            writer = VTTCaptionWriter()
            writer.add_caption(
                start=3661.5,  # 1 hour, 1 minute, 1.5 seconds
                end=3665.0,  # 1 hour, 1 minute, 5 seconds
                text="Long time",
                event_type="speech",
            )

            writer.write(temp_path)

            # Read the file and check content
            with open(temp_path, "r", encoding="utf-8") as f:
                content = f.read()

            assert "01:01:01.500 --> 01:01:05.000" in content

        finally:
            os.unlink(temp_path)

    def test_multiple_captions_same_time(self):
        """Test handling multiple captions with the same start time."""
        writer = VTTCaptionWriter()
        writer.add_caption(start=1.0, end=2.0, text="First", event_type="speech")
        writer.add_caption(start=1.0, end=2.0, text="Second", event_type="speech")

        assert len(writer._captions) == 2

        # Both should be preserved in the output
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".vtt", delete=False
        ) as temp_file:
            temp_path = temp_file.name

        try:
            writer.write(temp_path)

            with open(temp_path, "r", encoding="utf-8") as f:
                content = f.read()

            assert "First" in content
            assert "Second" in content

        finally:
            os.unlink(temp_path)

    def test_custom_duration_thresholds(self):
        """Test custom duration thresholds for silence and music."""
        writer = VTTCaptionWriter(min_silence_duration=0.5, min_music_duration=1.0)

        # These should now be included with the lower thresholds
        writer.add_caption(start=1.0, end=1.6, event_type="silence")  # 0.6 seconds
        writer.add_caption(start=2.0, end=3.1, event_type="music")  # 1.1 seconds

        assert len(writer._captions) == 2
        assert writer._captions[0]["text"] == "[SILENCE]"
        assert writer._captions[1]["text"] == "[MUSIC PLAYING]"
