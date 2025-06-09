import unittest
import tempfile
import os
import subprocess
from unittest.mock import patch, Mock
import numpy as np
import torch

from captionalchemy.tools.captioning.transcriber import Transcriber, WordTiming


class TestWordTimingDataclass(unittest.TestCase):
    def test_word_timing_creation(self):
        """Test creating a WordTiming instance."""
        word_timing = WordTiming(word="test", start=0.0, end=1.0, is_subword=True)
        self.assertEqual(word_timing.word, "test")
        self.assertEqual(word_timing.start, 0.0)
        self.assertEqual(word_timing.end, 1.0)
        self.assertFalse(word_timing.is_punctuation)
        self.assertFalse(word_timing.is_sentence_ending)
        self.assertTrue(word_timing.is_subword)

    def test_word_timing_invalid(self):
        with self.assertRaises(ValueError):
            WordTiming(word="", start=0.0, end=1.0)  # empty word
        with self.assertRaises(ValueError):
            WordTiming(word="hi", start=-1.0, end=0.5)  # negative start
        with self.assertRaises(ValueError):
            WordTiming(word="hi", start=1.0, end=0.5)  # end < start


class TestTranscriberParse(unittest.TestCase):
    def setUp(self):
        self.transcriber = Transcriber()

    def test_parse_timestamps(self):
        """Test parsing timestamps from a Whisper cpp output."""
        self.assertAlmostEqual(
            self.transcriber._parse_timestamps("00:01:23.456"), 83.456, places=3
        )
        self.assertAlmostEqual(
            self.transcriber._parse_timestamps("2:03:04.007"),
            2 * 3600 + 3 * 60 + 4.007,
            places=3,
        )
        with self.assertRaises(ValueError):
            self.transcriber._parse_timestamps("99:99:99")

    def test_parse_line(self):
        """Test parsing line from Whisper cpp output."""
        line = "[00:00:00.000 --> 00:00:00.500]   Hello"
        wt = self.transcriber._parse_line(line)
        self.assertEqual(wt.word, "Hello")
        self.assertAlmostEqual(wt.start, 0.0, places=3)
        self.assertAlmostEqual(wt.end, 0.5, places=3)
        # punctuation
        line2 = "[00:00:01.000 --> 00:00:01.200]   ."
        wt2 = self.transcriber._parse_line(line2)
        self.assertEqual(wt2.word, ".")
        self.assertTrue(wt2.is_punctuation)

    @patch("tempfile.NamedTemporaryFile")
    @patch("subprocess.run")
    @patch("whisper.load_model")
    @patch("os.remove")
    def test_transcribe_audio_python_api(
        self, mock_remove, mock_load, mock_run, mock_tempfile
    ):
        """Test transcribing audio using Whisper Python API."""
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

        transcriber = Transcriber()
        words = transcriber.transcribe_audio(
            audio_file="in.wav",
            start=1.0,
            end=2.0,
            model="base",
            whisper_build_path=None,
            whisper_model_path=None,
            platform="linux",
        )

        self.assertEqual(len(words), 2)
        self.assertEqual(words[0].word, "Hello")
        self.assertAlmostEqual(words[0].start, 0.0, places=3)
        self.assertAlmostEqual(words[0].end, 0.5, places=3)
        self.assertEqual(words[1].word, "world")
        self.assertAlmostEqual(words[1].start, 0.6, places=3)
        self.assertAlmostEqual(words[1].end, 1.1, places=3)


if __name__ == "__main__":
    unittest.main()
