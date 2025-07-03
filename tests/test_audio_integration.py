import pytest

from captionalchemy.tools.audio_analysis.audio_segment_integration import (
    AudioEvent,
    EventType,
    assign_speakers_to_speech_segment,
    identify_silence_gaps,
)


def test_audio_event_creation_speech():
    event = AudioEvent(
        start=0.0,
        end=1.0,
        speaker_id="Speaker_00",
        event_type=EventType.SPEECH,
        confidence=0.95,
    )
    assert event.start == 0.0
    assert event.end == 1.0
    assert event.speaker_id == "Speaker_00"
    assert event.event_type == EventType.SPEECH
    assert event.confidence == 0.95


def test_audio_event_creation_music():
    """Test creating a music AudioEvent."""
    event = AudioEvent(
        start=10.0,
        end=15.0,
        event_type=EventType.MUSIC,
        label="background_music",
        confidence=0.88,
    )

    assert event.event_type == EventType.MUSIC
    assert event.label == "background_music"
    assert event.speaker_id is None


def test_audio_event_duration_auto_calculation():
    """Test that duration is automatically calculated."""
    event = AudioEvent(start=1.5, end=4.7, event_type=EventType.SILENCE)
    assert abs(event.duration - 3.2) < 1e-6


def test_audio_event_validation_invalid_timing():
    """Test that invalid timing raises ValueError."""
    with pytest.raises(ValueError, match="End time must be greater than start time"):
        AudioEvent(start=2.0, end=1.0, event_type=EventType.SPEECH)


def test_assign_speakers():
    speech_segments = [{"start": 0.0, "end": 5.0}, {"start": 10.0, "end": 15.0}]

    diarization = {
        "SPEAKER_00": {"start": 0.0, "end": 5.0},
        "SPEAKER_01": {"start": 10.0, "end": 15.0},
    }
    result = assign_speakers_to_speech_segment(speech_segments, diarization)

    assert len(result) == 2
    assert result[0]["speaker_id"] == "SPEAKER_00"
    assert result[1]["speaker_id"] == "SPEAKER_01"


def test_assign_speakers_overlapping_segments():
    """Test speaker assignment with overlapping segments."""
    speech_segments = [{"start": 2.0, "end": 7.0}]  # Overlaps both speakers

    diarization = {
        "SPEAKER_00": {"start": 0.0, "end": 5.0},  # 3 seconds overlap
        "SPEAKER_01": {"start": 4.0, "end": 10.0},  # 3 seconds overlap
    }

    result = assign_speakers_to_speech_segment(speech_segments, diarization)
    assert len(result) == 1
    assert (
        result[0]["speaker_id"] == "SPEAKER_00"
    )  # Overlapping segments should assign the first speaker


def test_assign_speakers_no_diarization():
    """Test behavior when no diarization data is provided."""
    speech_segments = [{"start": 0.0, "end": 5.0}]
    diarization = {}

    result = assign_speakers_to_speech_segment(speech_segments, diarization)

    assert len(result) == 1
    assert result[0].get("speaker_id") is None


def test_assign_speakers_duration_calculation():
    """Test that duration is calculated correctly."""
    speech_segments = [{"start": 1.5, "end": 4.7}]
    diarization = {"SPEAKER_00": {"start": 0.0, "end": 10.0}}
    result = assign_speakers_to_speech_segment(speech_segments, diarization)
    assert abs(result[0]["duration"] - 3.2) < 1e-6


def test_identify_silence_gaps():
    speech_segments = [
        {"start": 0.0, "end": 3.0},
        {"start": 5.0, "end": 8.0},  # 2-second gap
    ]
    non_speech_segments = []
    silences = identify_silence_gaps(
        speech_segments,
        non_speech_segments,
        total_audio_duration=10.0,
        min_silence_duration=0.5,
    )
    assert len(silences) == 2

    first, second = silences
    assert abs(first["start"] - 3.0) < 1e-6
    assert abs(first["end"] - 5.0) < 1e-6
    assert abs(first["duration"] - 2.0) < 1e-6
    assert first["event_type"] == EventType.SILENCE

    assert abs(second["start"] - 8.0) < 1e-6
    assert abs(second["end"] - 10.0) < 1e-6
    assert abs(second["duration"] - 2.0) < 1e-6
    assert second["event_type"] == EventType.SILENCE


def test_no_silence_when_too_short():
    speech_segments = [{"start": 0.0, "end": 1.0}, {"start": 1.4, "end": 2.0}]
    non_speech_segments = []
    silences = identify_silence_gaps(
        speech_segments,
        non_speech_segments,
        total_audio_duration=2.1,  # Ensure the final silence check is valid
        min_silence_duration=0.5,
    )
    assert len(silences) == 0
