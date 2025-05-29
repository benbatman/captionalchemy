import os
from typing import List, Dict


class SRTCaptionWriter:
    """
    Accumulates speaker-based transcript segments and writes them out
    as a standard .srt caption file.
    """

    def __init__(self) -> None:
        self._captions: List[Dict] = []

    def add_caption(self, start: float, end: float, speaker: str, text: str) -> None:
        """
        Add one caption entry.

        Args:
            start (float): Segment start time in seconds.
            end (float): Segment end time in seconds.
            speaker (str): Speaker name or ID.
            text (str): Transcribed text.
        """
        self._captions.append(
            {
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": text.strip().replace("\n", " "),
            }
        )

    def write(self, filepath: str) -> None:
        """
        Write all accumulated captions to an .srt file.

        Args:
            filepath (str): Path to output .srt file. Overwrites if exists.
        """
        # sort by start time
        entries = sorted(self._captions, key=lambda c: c["start"])

        # ensure output directory exists
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            for idx, cap in enumerate(entries, start=1):
                start_ts = self._format_timestamp(cap["start"])
                end_ts = self._format_timestamp(cap["end"])

                # write index, timing line, then speaker: text
                f.write(f"{idx}\n")
                f.write(f"{start_ts} --> {end_ts}\n")
                f.write(f"{cap['speaker']}: {cap['text']}\n\n")

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """
        Convert seconds (float) to 'HH:MM:SS,mmm' format for SRT.
        """
        hours = int(seconds // 3600)
        seconds -= hours * 3600
        minutes = int(seconds // 60)
        seconds -= minutes * 60
        secs = int(seconds)
        millis = int((seconds - secs) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
