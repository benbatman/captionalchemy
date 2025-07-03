import tempfile
import os
from unittest.mock import patch, mock_open

from captionalchemy.tools.captioning.writers.sami_writer import SAMICaptionWriter


class TestSAMICaptionWriter:
    """Test suite for SAMICaptionWriter class."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        writer = SAMICaptionWriter()

        assert writer.min_silence_duration == 1.0
        assert writer.min_music_duration == 1.5
        assert writer.language == "en-US"
        assert writer.title == "Generated Captions"
        assert writer.last_speaker is None
        assert writer._captions == []

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        writer = SAMICaptionWriter(
            min_silence_duration=2.0,
            min_music_duration=3.0,
            language="fr-FR",
            title="Custom Title",
        )

        assert writer.min_silence_duration == 2.0
        assert writer.min_music_duration == 3.0
        assert writer.language == "fr-FR"
        assert writer.title == "Custom Title"

    def test_add_caption_with_text(self):
        """Test adding a caption with text."""
        writer = SAMICaptionWriter()
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
        writer = SAMICaptionWriter()
        writer.add_caption(
            start=1.0, end=3.0, text="Hello\nworld\ntest", event_type="speech"
        )

        caption = writer._captions[0]
        assert caption["text"] == "Hello world test"

    def test_add_caption_music_long_duration(self):
        """Test adding a music caption with long duration."""
        writer = SAMICaptionWriter()
        writer.add_caption(
            start=1.0,
            end=3.0,  # 2 seconds, > min_music_duration (1.5)
            event_type="music",
        )

        assert len(writer._captions) == 1
        caption = writer._captions[0]
        assert caption["text"] == "[music playing]"

    def test_add_caption_music_short_duration(self):
        """Test adding a music caption with short duration (should be skipped)."""
        writer = SAMICaptionWriter()
        writer.add_caption(
            start=1.0,
            end=2.0,  # 1 second, < min_music_duration (1.5)
            event_type="music",
        )

        assert len(writer._captions) == 0

    def test_add_caption_silence_long_duration(self):
        """Test adding a silence caption with long duration."""
        writer = SAMICaptionWriter()
        writer.add_caption(
            start=1.0,
            end=3.0,  # 2 seconds, > min_silence_duration (1.0)
            event_type="silence",
        )

        assert len(writer._captions) == 1
        caption = writer._captions[0]
        assert caption["text"] == "[silence]"

    def test_add_caption_silence_short_duration(self):
        """Test adding a silence caption with short duration (should be skipped)."""
        writer = SAMICaptionWriter()
        writer.add_caption(
            start=1.0,
            end=1.5,  # 0.5 seconds, < min_silence_duration (1.0)
            event_type="silence",
        )

        assert len(writer._captions) == 0

    def test_add_caption_with_label(self):
        """Test adding a caption with a label."""
        writer = SAMICaptionWriter()
        writer.add_caption(start=1.0, end=3.0, event_type="other", label="APPLAUSE")

        assert len(writer._captions) == 1
        caption = writer._captions[0]
        assert caption["text"] == "[applause]"

    def test_add_caption_fallback(self):
        """Test adding a caption with fallback text."""
        writer = SAMICaptionWriter()
        writer.add_caption(start=1.0, end=3.0, event_type="unknown")

        assert len(writer._captions) == 1
        caption = writer._captions[0]
        assert caption["text"] == "[audio]"

    def test_escape_text(self):
        """Test the _escape_text method."""
        writer = SAMICaptionWriter()

        # Test HTML entities
        assert writer._escape_text("Hello & world") == "Hello &amp; world"
        assert writer._escape_text("Hello < world") == "Hello &lt; world"
        assert writer._escape_text("Hello > world") == "Hello &gt; world"

        # Test quotes
        assert writer._escape_text('Hello "world"') == "Hello &quot;world&quot;"
        assert writer._escape_text("Hello 'world'") == "Hello &#39;world&#39;"

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    def test_write_empty_captions(self, mock_makedirs, mock_file):
        """Test writing with no captions."""
        writer = SAMICaptionWriter()
        writer.write("test.smi")

        mock_makedirs.assert_called_once_with(".", exist_ok=True)
        mock_file.assert_called_once_with("test.smi", "w", encoding="utf-8")

        # Check that the basic SAMI structure is written
        handle = mock_file.return_value
        written_content = "".join(call[0][0] for call in handle.write.call_args_list)

        assert "<sami>" in written_content
        assert "</sami>" in written_content
        assert "<head>" in written_content
        assert "</head>" in written_content
        assert "<body>" in written_content
        assert "</body>" in written_content
        assert "Generated Captions" in written_content

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    def test_write_with_captions(self, mock_makedirs, mock_file):
        """Test writing with captions."""
        writer = SAMICaptionWriter()
        writer.add_caption(
            start=1.0,
            end=3.0,
            speaker="John Doe",
            text="Hello world",
            event_type="speech",
        )
        writer.add_caption(start=4.0, end=6.0, event_type="music")

        writer.write("test.smi")

        handle = mock_file.return_value
        written_content = "".join(call[0][0] for call in handle.write.call_args_list)

        # Check for sync blocks
        assert 'sync start="1000"' in written_content
        assert 'sync start="4000"' in written_content
        assert "Hello world" in written_content
        assert "[music playing]" in written_content
        assert "John Doe" in written_content

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    def test_write_with_long_text(self, mock_makedirs, mock_file):
        """Test writing with long text that should be split into lines."""
        writer = SAMICaptionWriter()
        long_text = """This is a very long text that should be split into
                    multiple lines because it exceeds the character limit"""
        writer.add_caption(start=1.0, end=3.0, text=long_text, event_type="speech")

        writer.write("test.smi")

        handle = mock_file.return_value
        written_content = "".join(call[0][0] for call in handle.write.call_args_list)

        # Check that line breaks are added
        assert "<br/>" in written_content

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    def test_write_with_html_entities(self, mock_makedirs, mock_file):
        """Test writing with text containing HTML entities."""
        writer = SAMICaptionWriter()
        writer.add_caption(
            start=1.0, end=3.0, text="Hello & <world>", event_type="speech"
        )

        writer.write("test.smi")

        handle = mock_file.return_value
        written_content = "".join(call[0][0] for call in handle.write.call_args_list)

        # Check that HTML entities are escaped
        assert "&amp;" in written_content
        assert "&lt;" in written_content
        assert "&gt;" in written_content

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    def test_write_sorting(self, mock_makedirs, mock_file):
        """Test that captions are sorted by start time."""
        writer = SAMICaptionWriter()
        writer.add_caption(start=3.0, end=4.0, text="Third", event_type="speech")
        writer.add_caption(start=1.0, end=2.0, text="First", event_type="speech")
        writer.add_caption(start=2.0, end=3.0, text="Second", event_type="speech")

        writer.write("test.smi")

        handle = mock_file.return_value
        written_content = "".join(call[0][0] for call in handle.write.call_args_list)

        # Find positions of the sync blocks
        first_pos = written_content.find('sync start="1000"')
        second_pos = written_content.find('sync start="2000"')
        third_pos = written_content.find('sync start="3000"')

        assert first_pos < second_pos < third_pos

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    def test_write_custom_language_and_title(self, mock_makedirs, mock_file):
        """Test writing with custom language and title."""
        writer = SAMICaptionWriter(language="fr-FR", title="Test Title")
        writer.add_caption(start=1.0, end=3.0, text="Bonjour", event_type="speech")

        writer.write("test.smi")

        handle = mock_file.return_value
        written_content = "".join(call[0][0] for call in handle.write.call_args_list)

        # Check for custom title
        assert "Test Title" in written_content

        # Check for custom language class
        assert "FRFR" in written_content  # fr-FR becomes FRFR
        assert "lang: fr-FR" in written_content

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    def test_write_clearing_syncs(self, mock_makedirs, mock_file):
        """Test that clearing syncs are added appropriately."""
        writer = SAMICaptionWriter()
        writer.add_caption(start=1.0, end=2.0, text="First", event_type="speech")
        writer.add_caption(
            start=4.0, end=5.0, text="Second", event_type="speech"
        )  # Gap > 500ms

        writer.write("test.smi")

        handle = mock_file.return_value
        written_content = "".join(call[0][0] for call in handle.write.call_args_list)

        # Check for clearing syncs
        assert 'sync start="2000"' in written_content  # End of first caption
        assert 'sync start="5000"' in written_content  # End of second caption
        assert "&nbsp;" in written_content  # Non-breaking space for clearing

    def test_write_creates_directory(self):
        """Test that write creates the output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "subdir", "test.smi")
            writer = SAMICaptionWriter()
            writer.add_caption(start=1.0, end=2.0, text="Test", event_type="speech")

            writer.write(output_path)

            # Check that the directory was created
            assert os.path.exists(os.path.dirname(output_path))
            assert os.path.exists(output_path)

    def test_write_real_file(self):
        """Test writing to a real file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".smi", delete=False
        ) as temp_file:
            temp_path = temp_file.name

        try:
            writer = SAMICaptionWriter()
            writer.add_caption(
                start=1.0,
                end=3.0,
                speaker="John Doe",
                text="Hello world",
                event_type="speech",
            )

            writer.write(temp_path)

            # Read the file and check content
            with open(temp_path, "r", encoding="utf-8") as f:
                content = f.read()

            assert "<sami>" in content
            assert "</sami>" in content
            assert "Hello world" in content
            assert "John Doe" in content
            assert 'sync start="1000"' in content

        finally:
            os.unlink(temp_path)
