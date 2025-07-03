from typing import List, Dict, Optional
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
    speaker_name: Optional[str] = None
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
    if not diarization:
        # If no diarization data, return segments as is
        for segment in speech_segments:
            segment["speaker_id"] = None
            segment["duration"] = segment["end"] - segment["start"]
        return speech_segments

    speech_segments_with_diarization = []

    for segment in speech_segments:
        segment_start = segment["start"]
        segment_end = segment["end"]
        assigned_speaker = None

        max_overlap = 0.0

        # Find which speaker's time range this segment falls into
        # Looking for the speaker whose range has the maximum overlap with this segment
        for speaker_id, time_range in diarization.items():
            print(
                f"Checking speaker {speaker_id} for segment {segment_start} - {segment_end}"
            )
            speaker_start = time_range["start"]
            speaker_end = time_range["end"]

            # Calculate overlap
            overlap_start = max(segment_start, speaker_start)
            overlap_end = min(segment_end, speaker_end)
            overlap = max(0, overlap_end - overlap_start)

            # If this segment is mostly within this speaker's range, assign it
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

    return speech_segments_with_diarization


def identify_silence_gaps(
    speech_segments: List[Dict],
    non_speech_segments: List[Dict],
    total_audio_duration: float,
    min_silence_duration: float = 0.5,
) -> List[Dict]:
    """
    Identify periods of silence (gaps between speech and non-speech events).

    This function finds the "empty spaces" in the audio timeline - periods where
    neither speech nor other sounds are detected.

    Args:
        speech_segments: List of speech segments
        non_speech_events: List of non-speech events
        total_audio_duration: Total length of audio in seconds
        min_silence_duration: Minimum duration to consider as silence

    Returns:
        List of silence periods with start/end times
    """
    all_events = []

    for segment in speech_segments:
        all_events.append((segment["start"], segment["end"], EventType.SPEECH))

    for event in non_speech_segments:
        all_events.append((event["start"], event["end"], EventType.OTHER_SOUND))

    # Sort by start time
    all_events.sort(key=lambda x: x[0])

    silence_periods = []
    last_end_time = 0.0

    for start, end, event_type in all_events:
        # Check if there's a gap btn the last event and this one
        if start > last_end_time + min_silence_duration:
            silence_periods.append(
                {
                    "start": last_end_time,
                    "end": start,
                    "duration": start - last_end_time,
                    "event_type": EventType.SILENCE,
                }
            )

        # Update the last end time, handling overlaps
        last_end_time = max(last_end_time, end)

    # Check for silence at the very end
    if total_audio_duration > last_end_time + min_silence_duration:
        silence_periods.append(
            {
                "start": last_end_time,
                "end": total_audio_duration,
                "duration": total_audio_duration - last_end_time,
                "event_type": EventType.SILENCE,
            }
        )

    return silence_periods


def integrate_audio_segments(
    speech_segments: List[Dict],
    non_speech_segments: List[Dict],
    diarization: Dict[str, Dict],
    total_audio_duration: Optional[float] = None,
) -> List[AudioEvent]:
    """
    Create a comprehensive, chronologically ordered timeline of all audio events.

    This is the main integration function that brings together all your audio
    analysis results into a single, coherent timeline. Think of this as creating
    a detailed transcript that includes not just speech, but all audio events.

    Args:
        speech_segments: Output from get_speech_segments()
        non_speech_events: Output from detect_non_speech_segments()
        diarization_result: Output from diarize()
        total_audio_duration: Total audio length (estimated if not provided)

    Returns:
        Chronologically sorted list of AudioEvent objects
    """
    # Assign speakers to speech segments
    speech_segments_with_speakers = assign_speakers_to_speech_segment(
        speech_segments, diarization
    )

    # Estimate total duration if not provided
    if total_audio_duration is None:
        max_speech_end = (
            max([seg["end"] for seg in speech_segments]) if speech_segments else 0
        )
        max_non_speech_end = (
            max([seg["end"] for seg in non_speech_segments])
            if non_speech_segments
            else 0
        )
        total_audio_duration = max(max_speech_end, max_non_speech_end)

    # Identify silence gaps
    silence_segments = identify_silence_gaps(
        speech_segments, non_speech_segments, total_audio_duration
    )

    # Convert it all to AudioEvent objects
    audio_events = []

    # Add speech segments
    for segment in speech_segments_with_speakers:
        audio_events.append(
            AudioEvent(
                start=segment["start"],
                end=segment["end"],
                event_type=EventType.SPEECH,
                speaker_id=segment.get("speaker_id"),
                duration=segment.get("duration"),
            )
        )

    # Add non-speech segments
    for segment in non_speech_segments:
        # Map the label to appropriate event type
        if segment["label"].lower() == "music":
            event_type = EventType.MUSIC
        else:
            event_type = EventType.OTHER_SOUND

        event = AudioEvent(
            start=float(segment["start"]),
            end=float(segment["end"]),
            event_type=event_type,
            confidence=float(segment["confidence"]),
            label=segment["label"],
            duration=float(segment["duration"]),
        )
        audio_events.append(event)

    # Add silence segments
    for silence in silence_segments:
        audio_events.append(
            AudioEvent(
                start=silence["start"],
                end=silence["end"],
                event_type=EventType.SILENCE,
                duration=silence["duration"],
            )
        )

    # Sort all events by start time
    audio_events.sort(key=lambda x: x.start)

    return audio_events
