from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum


class EventType(Enum):
    SPEECH = "speech"
    MUSIC = "music"
    SILENCE = "silence"
    OTHER_SOUND = "other_sound"


@dataclass
class AudioEvent:
    """
    Represents an audio event with its type and time range.
    """

    start: float
    end: float
    event_type: EventType
    speaker_id: Optional[str] = None
    confidence: Optional[float] = None
    label: Optional[str] = None
    duration: Optional[float] = None

    def __post_init__(self):
        if self.duration is None:
            self.duration = self.end - self.start
        if self.duration < 0:
            raise ValueError("End time must be greater than start time.")


def assign_speakers_to_speech_segment(
    speech_segments: List[Dict], diarization: Dict[str, Dict]
) -> List[Dict]:
    """
    Assigns speakers to speech segments based on diarization data.

    Args:
        speech_segments (List[Dict]): List of speech segments with start and end times.
        diarization (Dict[str, Dict]): Diarization data mapping speakers to their time ranges.

    Returns:
        List[Dict]: Updated speech segments with assigned speaker IDs.
    """
    speech_segments_with_diarization = []

    for segment in speech_segments:
        segment_start = segment["start"]
        segment_end = segment["end"]
        assigned_speaker = None

        max_overlap = 0.0

        # Find which speaker's time range this segment falls into
        # Looking for the speaker whose range has the maximum overlap with this segment
        for speaker_id, time_range in diarization.items():
            speaker_start = time_range["start"]
            speaker_end = time_range["end"]

            # Calculate overlap
            overlap_start = max(segment_start, speaker_start)
            overlap_end = min(segment_end, speaker_end)
            overlap = max(0, overlap_end - overlap_start)

            # If this egment is mostly within this speaker's range, assign it
            segment_duration = segment_end - segment_start
            overlap_ratio = overlap / segment_duration if segment_duration > 0 else 0

            if overlap_ratio > 0.5 and overlap > max_overlap:  # Majority overlap
                max_overlap = overlap
                assigned_speaker = speaker_id

        # Add speaker information
        segment_with_speaker = segment.copy()
        segment_with_speaker["speaker_id"] = assigned_speaker
        segment_with_speaker["duration"] = segment_end - segment_start
        speech_segments_with_diarization.append(segment_with_speaker)
