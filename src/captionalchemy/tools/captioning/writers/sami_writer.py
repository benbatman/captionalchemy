import os
import html
from typing import List, Dict, Optional


class SAMICaptionWriter:
    """
    Accumulates speaker-based transcript segments and writes them out
    as a standard .smi (SAMI) caption file following the Microsoft
    Synchronized Accessible Media Interchange specification.
    """

    def __init__(
        self,
        min_silence_duration: float = 1.0,
        min_music_duration: float = 1.5,
        language: str = "en-US",
        title: str = "Generated Captions",
    ) -> None:
        self._captions: List[Dict] = []
        self.min_silence_duration = min_silence_duration
        self.min_music_duration = min_music_duration
        self.language = language
        self.title = title
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
                caption_text = "[music playing]"
            else:
                return
        elif event_type == "silence":
            # Only add silence captions for longer silences
            if end - start >= self.min_silence_duration:
                caption_text = "[silence]"
            else:
                return  # Skip short silences
        elif label:
            # For other sound events, use the detected label
            caption_text = f"[{label.lower()}]"
        else:
            caption_text = "[audio]"  # Fallback

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
        Write all accumulated captions to a .smi file.

        Args:
            filepath (str): Path to output .smi file. Overwrites if exists.
        """
        # Sort by start time
        entries = sorted(self._captions, key=lambda c: c["start"])

        # Ensure output directory exists
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            # Write SAMI document structure
            f.write("<sami>\n")
            f.write(" <head>\n")
            f.write(f"  <title>{html.escape(self.title)}</title>\n")

            # Write CSS styles
            f.write('  <style type="text/css">\n')
            f.write("   <!--\n")

            # Base paragraph style
            f.write("    p {\n")
            f.write("     font-family: Arial, sans-serif;\n")
            f.write("     font-size: 14pt;\n")
            f.write("     color: white;\n")
            f.write("     background-color: black;\n")
            f.write("     text-align: center;\n")
            f.write("     margin: 0;\n")
            f.write("     padding: 2px;\n")
            f.write("    }\n")

            # Language class
            lang_class = self.language.replace("-", "").upper()
            f.write(f"    .{lang_class} {{\n")
            f.write(f"     lang: {self.language};\n")
            f.write("    }\n")

            # Source/Speaker style for identification
            f.write("    #Source {\n")
            f.write("     color: yellow;\n")
            f.write("     font-weight: bold;\n")
            f.write("     font-size: 12pt;\n")
            f.write("    }\n")

            f.write("   -->\n")
            f.write("  </style>\n")
            f.write(" </head>\n")
            f.write(" <body>\n")

            # Process entries and add clearing syncs
            for i, cap in enumerate(entries):
                start_ms = int(cap["start"] * 1000)
                end_ms = int(cap["end"] * 1000)

                # Write sync block for the caption
                f.write(f'  <sync start="{start_ms}">\n')

                # Escape HTML entities in text
                escaped_text = html.escape(cap["text"])

                # Handle line breaks for longer text
                if len(escaped_text) > 40:
                    # Split long text into multiple lines at word boundaries
                    words = escaped_text.split()
                    lines = []
                    current_line = []
                    current_length = 0

                    for word in words:
                        if current_length + len(word) + 1 <= 40:
                            current_line.append(word)
                            current_length += len(word) + 1
                        else:
                            if current_line:
                                lines.append(" ".join(current_line))
                            current_line = [word]
                            current_length = len(word)

                    if current_line:
                        lines.append(" ".join(current_line))

                    escaped_text = "<br/>".join(lines)

                # Write paragraph with appropriate class and speaker info
                if cap["speaker"] and cap["event_type"] == "speech":
                    # Include speaker information for speech
                    if self.last_speaker != cap["speaker"]:
                        self.last_speaker = cap["speaker"]
                        f.write(
                            f"   <p class=\"{lang_class}\" id=\"Source\">{html.escape(cap['speaker'])}</p>\n"
                        )
                    f.write(f'   <p class="{lang_class}">\n')
                    f.write(f"    {escaped_text}\n")
                    f.write("   </p>\n")
                else:
                    # For music and other events
                    f.write(f'   <p class="{lang_class}">\n')
                    f.write(f"    {escaped_text}\n")
                    f.write("   </p>\n")

                f.write("  </sync>\n")

                # Add clearing sync at the end of the caption duration
                # Check if there's a gap before the next caption
                next_start_ms = None
                if i + 1 < len(entries):
                    next_start_ms = int(entries[i + 1]["start"] * 1000)

                # Add a clearing sync if there's a significant gap or this is the last caption
                if (
                    next_start_ms is None or next_start_ms - end_ms > 500
                ):  # 500ms gap threshold
                    f.write(f'  <sync start="{end_ms}">\n')
                    f.write(f'   <p class="{lang_class}">\n')
                    f.write("    &nbsp;\n")
                    f.write("   </p>\n")
                    f.write("  </sync>\n")

            f.write(" </body>\n")
            f.write("</sami>\n")

    def _escape_text(self, text: str) -> str:
        """
        Escape HTML entities and handle special characters for SAMI format.
        """
        # First escape basic HTML entities
        # Single quotes get outputted to the numeric &#x27;, not the HTML decimal &#39;
        escaped = html.escape(text, quote=False)

        # Replace other common characters that might cause issues
        escaped = escaped.replace('"', "&quot;")
        escaped = escaped.replace("'", "&#39;")

        return escaped
