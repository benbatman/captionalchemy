import pytest
from unittest.mock import patch, Mock

from captionalchemy.tools.captioning.transcriber import Transcriber, WordTiming


def test_word_timing_creation():
    """Test creating a WordTiming instance."""
    word_timing = WordTiming(word="test", start=0.0, end=1.0, is_subword=True)
    assert word_timing.word == "test"
    assert word_timing.start == 0.0
    assert word_timing.end == 1.0
    assert word_timing.is_punctuation is False
    assert word_timing.is_sentence_ending is False
    assert word_timing.is_subword is True


def test_word_timing_invalid_empty_word():
    with pytest.raises(ValueError):
        WordTiming(word="", start=0.0, end=1.0)  # empty word


def test_word_timing_invalid_negative_start():
    with pytest.raises(ValueError):
        WordTiming(word="hi", start=-1.0, end=0.5)  # negative start


def test_word_timing_invalid_end_before_start():
    with pytest.raises(ValueError):
        WordTiming(word="hi", start=1.0, end=0.5)  # end < start


@pytest.fixture
def transcriber():
    return Transcriber()


def test_parse_timestamps(transcriber):
    """Test parsing timestamps from a Whisper cpp output."""
    result = transcriber._parse_timestamps("00:01:23.456")
    assert abs(result - 83.456) < 1e-3

    result = transcriber._parse_timestamps("2:03:04.007")
    expected = 2 * 3600 + 3 * 60 + 4.007
    assert abs(result - expected) < 1e-3

    with pytest.raises(ValueError):
        transcriber._parse_timestamps("99:99:99")


def test_parse_line(transcriber):
    """Test parsing line from Whisper cpp output."""
    line = "[00:00:00.000 --> 00:00:00.500]   Hello"
    wt = transcriber._parse_line(line)
    assert wt.word == "Hello"
    assert abs(wt.start - 0.0) < 1e-3
    assert abs(wt.end - 0.5) < 1e-3

    # punctuation
    line2 = "[00:00:01.000 --> 00:00:01.200]   ."
    wt2 = transcriber._parse_line(line2)
    assert wt2.word == "."
    assert wt2.is_punctuation is True


@patch("captionalchemy.tools.captioning.transcriber.tempfile.NamedTemporaryFile")
@patch("captionalchemy.tools.captioning.transcriber.subprocess.run")
@patch("captionalchemy.tools.captioning.transcriber.whisper.load_model")
@patch("captionalchemy.tools.captioning.transcriber.os.remove")
def test_transcribe_audio_python_api(mock_remove, mock_load, mock_run, mock_tempfile):
    """Test transcribing audio using Whisper Python API."""
    transcriber = Transcriber()
    # Mock temp file
    fake = Mock()
    fake.name = "/tmp/fake.wav"
    mock_tempfile.return_value = fake

    # Mock ffmpeg trim
    mock_run.return_value = Mock(returncode=0)

    # Mock whisper model load & transcribe
    mock_model = Mock()
    mock_load.return_value = mock_model
    mock_model.transcribe.return_value = {
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

    words = transcriber.transcribe_audio(
        audio_file="in.wav",
        start=0.0,
        end=1.1,
        model="base",
        whisper_build_path=None,
        whisper_model_path=None,
        platform="linux",
    )

    assert len(words) == 2
    assert words[0].word == "Hello"
    assert abs(words[0].start - 0.0) < 1e-3
    assert abs(words[0].end - 0.5) < 1e-3

    assert words[1].word == "world"
    assert abs(words[1].start - 0.6) < 1e-3
    assert abs(words[1].end - 1.1) < 1e-3

    mock_remove.assert_called_once_with(fake.name)
