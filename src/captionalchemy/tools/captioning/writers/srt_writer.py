import os
from typing import List, Dict, Optional


class SRTCaptionWriter:
    """
    Accumulates speaker-based transcript segments and writes them out
    as a standard .srt caption file.
    """

    def __init__(
        self,
        min_silence_duration: float = 1.0,
        min_music_duration: float = 1.0,
    ) -> None:
        self._captions: List[Dict] = []
        self.min_silence_duration = min_silence_duration
        self.min_music_duration = min_music_duration
        self.last_speaker: Optional[str] = None

    def add_caption(
        self,
        start: float,
        end: float,
        speaker: Optional[str] = None,
        text: Optional[str] = None,
        event_type: Optional[str] = None,
        label: Optional[str] = None,
    ) -> None:
        """
        Add one caption entry.

        Args:
            start (float): Segment start time in seconds.
            end (float): Segment end time in seconds.
            speaker (str): Speaker name or ID.
            text (str): Transcribed text.
            event_type (str, optional): Type of event (speech, music, silence, etc.).
            label (str, optional): Original label from audio detection.
        """
        # Determine the caption text based on event type
        if text:
            caption_text = text.strip().replace("\n", " ")
        elif event_type == "music":
            if end - start >= self.min_music_duration:
                caption_text = "[MUSIC PLAYING]"
            else:
                return
        elif event_type == "silence":
            # Only add silence captions for longer silences
            if end - start >= self.min_silence_duration:
                caption_text = "[SILENCE]"
            else:
                return  # Skip short silences
        elif label:
            # For other sound events, use the detected label
            caption_text = f"[{label.upper()}]"
        else:
            caption_text = "[AUDIO]"  # Fallback

        self._captions.append(
            {
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": caption_text,
                "event_type": event_type,
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
                # For speech events, include speaker name
                if cap["speaker"] and cap["event_type"] == "speech":
                    if self.last_speaker != cap["speaker"]:
                        self.last_speaker = cap["speaker"]
                        f.write(f"{cap['speaker']}: {cap['text']}\n\n")
                    else:
                        # If the speaker hasn't changed, just write the text
                        f.write(f"{cap['text']}\n\n")
                else:
                    # For music and other events, just show the description
                    f.write(f"{cap['text']}\n\n")

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
