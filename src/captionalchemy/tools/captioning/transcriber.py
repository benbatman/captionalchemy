import subprocess
import tempfile
import os
import logging
from sys import platform
import re
from typing import List, Optional
from dataclasses import dataclass

from whisper import Whisper
import whisper
import torch

logger = logging.getLogger(__name__)


@dataclass
class WordTiming:
    word: str
    start: float
    end: float
    duration: Optional[float] = None
    is_punctuation: bool = False
    is_sentence_ending: bool = False
    is_clause_ending: bool = False
    is_subword: bool = False

    def __post_init__(self):
        if self.duration is None:
            self.duration = self.end - self.start
        if self.start < 0 or self.end < 0:
            raise ValueError("Start and end times must be non-negative.")
        if self.end < self.start:
            raise ValueError("End time must be greater than start time.")
        if not self.word:
            raise ValueError("Word cannot be empty.")


class Transcriber:
    """
    Transcriber class for converting audio to text using OpenAI's Whisper model.
    It supports both Whisper Python API and Whisper.cpp for macOS.
    It can handle word-level timing and punctuation detection.
    It parses lines in the format:
    [HH:MM:SS.mmm --> HH:MM:SS.mmm]   word
    where HH:MM:SS.mmm is the timestamp format.
    It returns a list of WordTiming objects containing the word, start time, end time,
    duration, and flags for punctuation and subwords.
    It also handles subwords by checking the number of leading spaces in the raw token.
    It uses ffmpeg to trim audio files to the specified start and end times.
    It can run on different platforms (macOS, Linux, Windows) and uses the appropriate
    method for transcription based on the platform and availability of Whisper.cpp.
    """

    def __init__(self):

        self.timestamp_pattern = re.compile(
            r"""
        ^\[
          (?P<start>\d{2}:\d{2}:\d{2}\.\d{3})
          \s*-->\s*
          (?P<end>\d{2}:\d{2}:\d{2}\.\d{3})
        \]
        (?P<raw_token>.*)      # capture the rest of the line, including any leading spaces
        $""",
            re.VERBOSE,
        )
        self.punctuation_set = {
            ".",
            ",",
            "!",
            "?",
            ";",
            ":",
            "-",
            "—",
            "...",
            "'",
            '"',
            "(",
            ")",
            "[",
            "]",
        }

    def _parse_timestamps(self, timestamp_str: str) -> float:
        """
        Convert HH:MM:SS.mmm format to total seconds as float.

        Args:
            timestamp_str (str): Timestamp string in HH:MM:SS.mmm format.
        Returns:
            float: Total seconds represented by the timestamp.
        """
        stripped_str = timestamp_str.strip()
        if not stripped_str:
            raise ValueError("Timestamp string cannot be empty.")
        try:
            time_parts = timestamp_str.split(":")
            hours = int(time_parts[0])
            minutes = int(time_parts[1])

            # Handle seconds and milliseconds
            seconds_parts = time_parts[2].split(".")
            seconds = int(seconds_parts[0])
            milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0

            # Run sanity checks
            if hours < 0 or minutes < 0 or seconds < 0 or milliseconds < 0:
                raise ValueError("Negative values are not allowed in timestamps.")
            if minutes >= 60 or seconds >= 60 or milliseconds >= 1000:
                raise ValueError(
                    "Invalid timestamp values: minutes must be < 60, "
                    "seconds must be < 60, milliseconds must be < 1000."
                )
            if hours > 24:
                raise ValueError("Hours must be less than or equal to 24.")

            # Calculate total seconds
            total_seconds = (
                hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
            )
            return total_seconds
        except (ValueError, IndexError) as e:
            logger.error(f"Invalid timestamp format: {timestamp_str}. Error: {e}")
            raise ValueError(f"Invalid timestamp format: {timestamp_str}") from e

    def _parse_line(self, line: str) -> WordTiming:
        """
        Parse a single line containing word timing data.

        Example input: "[00:00:00.040 --> 00:00:00.110]   If"
        Returns: WordTiming object with start=0.040, end=0.110, word="If"
        """
        line = line.rstrip("\n")

        match = self.timestamp_pattern.match(line)

        if not match:
            raise ValueError(f"Invalid line format: {line}")

        start_time_str = match.group("start")
        end_time_str = match.group("end")
        raw_token = match.group("raw_token")

        # Convert timestamp strings to seconds
        start_seconds = self._parse_timestamps(start_time_str)
        end_seconds = self._parse_timestamps(end_time_str)
        if start_seconds < 0 or end_seconds < 0:
            raise ValueError("Start and end times must be non-negative.")

        if end_seconds < start_seconds:
            raise ValueError("End time must be greater than start time.")
        if not raw_token:
            raise ValueError("Raw token cannot be empty.")

        n_leading_spaces = len(raw_token) - len(raw_token.lstrip(" "))
        # Whisper outputs 3 space if whole word, 2 spaces if subword
        # Punctuations are taken care of separately
        is_subword = raw_token not in self.punctuation_set and n_leading_spaces == 2
        token = raw_token.strip()

        return WordTiming(
            word=token,
            start=start_seconds,
            end=end_seconds,
            duration=end_seconds - start_seconds,
            is_subword=is_subword,
            is_punctuation=token in self.punctuation_set,
            is_sentence_ending=token in {".", "!", "?"},
            is_clause_ending=token in {",", ";", ":", "—"},
        )

    def transcribe_audio(
        self,
        audio_file: str,
        start: float,
        end: float,
        model: Whisper | str,
        whisper_build_path: Optional[str] = None,
        whisper_model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        platform: str = platform.lower(),  # 'darwin', 'linux', 'windows'
    ) -> List[WordTiming]:
        """
        Transcribe audio to text using OpenAI's Whisper model.

        Args:
            audio_file (str): Path to the audio file.
            start (float): Start time in seconds for the audio segment.
            end (float): End time in seconds for the audio segment.
            model (Whisper | str): Whisper model instance or path to the model.
            whisper_build_path (str): Path to the Whisper build directory.
            whisper_model_path (str): Path to the Whisper model file.
            device (str): Device to run the model on ('cuda' or 'cpu').
            platform (str): Platform type ('darwin', 'linux', 'windows').

        Returns:
            str: The transcribed text.
        """
        tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_path = tmp_wav.name
        tmp_wav.close()

        # Trim the audio using ffmpeg
        subprocess.run(
            [
                "ffmpeg",
                "-y",  # Overwrite output files without asking
                "-i",
                audio_file,
                "-ss",
                str(start),
                "-to",
                str(end),
                "-ar",
                "16000",  # Set sample rate to 16 kHz
                "-ac",
                "1",  # Set number of channels to 1 (mono)
                "-f",
                "wav",
                tmp_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.debug(f"Trimmed audio saved to {tmp_path}")

        # Optional use of Whisper.cpp for macOS
        if (
            platform == "darwin"
            and os.path.exists("whisper.cpp")
            and whisper_build_path
            and whisper_model_path
        ):
            cmd = [
                whisper_build_path,
                "-m",
                whisper_model_path,
                "-f",
                tmp_path,
                "-ml",
                "1",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Error in transcription: {result.stderr}")

            os.unlink(tmp_path)

            transcription = result.stdout
            logger.debug(f"Whisper.cpp output: {transcription}")

            word_timings = []
            for line in transcription.splitlines():
                stripped_line = line.strip()
                if not stripped_line or not stripped_line.startswith("["):
                    continue
                try:
                    word_timing = self._parse_line(line)
                    # Adjust timestamp to original audio timeline
                    word_timing.start += start
                    word_timing.end += start
                    word_timings.append(word_timing)
                except ValueError as e:
                    logger.warning(f"Skipping invalid line: {line}. Error: {e}")

            logger.debug(f"Transcription result: {transcription}")
            return word_timings

        # Use Whisper Python API for Linux and Windows, ideally have cuda
        else:
            # Whisper python API outputs punctuation as part of the word
            # Nor does it output word fragments
            word_timings = []
            logger.info("Using Whisper Python API for transcription")
            if isinstance(model, str):
                model = whisper.load_model(model, device=device)
            elif not isinstance(model, Whisper):
                raise ValueError(
                    "Model must be a Whisper instance or a model name string"
                )
            result = model.transcribe(
                tmp_path,
                language="en",
                word_timestamps=True,
                fp16=torch.cuda.is_available(),
            )
            transcription = result["text"]
            os.remove(tmp_path)  # Clean up the temporary file
            segments = result.get("segments", [])
            for seg in segments:
                for w in seg.get("words", []):
                    text = w.get("word", "").strip()
                    # Skip if empty (just in case)
                    if not text:
                        continue
                    start_w = w.get("start", 0.0)
                    end_w = w.get("end", 0.0)
                    # sanity check
                    if end_w < start_w:
                        continue

                    # Adjust timestamps to original audio timeline
                    adjusted_start = start + start_w
                    adjusted_end = start + end_w
                    word_timings.append(
                        WordTiming(
                            word=text,
                            start=adjusted_start,
                            end=adjusted_end,
                            is_subword=False,
                        )
                    )

            logger.debug(
                f"Python API output parsed into {len(word_timings)} WordTiming entries."
            )
            return word_timings
