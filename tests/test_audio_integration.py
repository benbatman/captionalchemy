import unittest

from captionalchemy.tools.audio_analysis.audio_segment_integration import (
    AudioEvent,
    EventType,
    assign_speakers_to_speech_segment,
    identify_silence_gaps,
)


class TestAudioEvent(unittest.TestCase):
    def test_audio_event_creation_speech(self):
        event = AudioEvent(
            start=0.0,
            end=1.0,
            speaker_id="Speaker_00",
            event_type=EventType.SPEECH,
            confidence=0.95,
        )
        self.assertEqual(event.start, 0.0)
        self.assertEqual(event.end, 1.0)
        self.assertEqual(event.speaker_id, "Speaker_00")
        self.assertEqual(event.event_type, EventType.SPEECH)
        self.assertEqual(event.confidence, 0.95)

    def test_audio_event_creation_music(self):
        """Test creating a music AudioEvent."""
        event = AudioEvent(
            start=10.0,
            end=15.0,
            event_type=EventType.MUSIC,
            label="background_music",
            confidence=0.88,
        )

        self.assertEqual(event.event_type, EventType.MUSIC)
        self.assertEqual(event.label, "background_music")
        self.assertIsNone(event.speaker_id)

    def test_audio_event_duration_auto_calculation(self):
        """Test that duration is automatically calculated."""
        event = AudioEvent(start=1.5, end=4.7, event_type=EventType.SILENCE)
        self.assertAlmostEqual(event.duration, 3.2, places=6)

    def test_audio_event_validation_invalid_timing(self):
        """Test that invalid timing raises ValueError."""
        with self.assertRaises(ValueError) as context:
            AudioEvent(start=2.0, end=1.0, event_type=EventType.SPEECH)

        self.assertIn(
            "End time must be greater than start time", str(context.exception)
        )


class TestSpeakerAssignment(unittest.TestCase):
    def test_assign_speakers(self):
        speech_segments = [{"start": 0.0, "end": 5.0}, {"start": 10.0, "end": 15.0}]

        diarization = {
            "SPEAKER_00": {"start": 0.0, "end": 5.0},
            "SPEAKER_01": {"start": 10.0, "end": 15.0},
        }
        result = assign_speakers_to_speech_segment(speech_segments, diarization)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["speaker_id"], "SPEAKER_00")
        self.assertEqual(result[1]["speaker_id"], "SPEAKER_01")

    def test_assign_speakers_overlapping_segments(self):
        """Test speaker assignment with overlapping segments."""
        speech_segments = [{"start": 2.0, "end": 7.0}]  # Overlaps both speakers

        diarization = {
            "SPEAKER_00": {"start": 0.0, "end": 5.0},  # 3 seconds overlap
            "SPEAKER_01": {"start": 4.0, "end": 10.0},  # 3 seconds overlap
        }

        result = assign_speakers_to_speech_segment(speech_segments, diarization)
        self.assertEqual(len(result), 1)
        self.assertEqual(
            result[0]["speaker_id"], "SPEAKER_00"
        )  # Overlapping segments should assign the first speaker

    def test_assign_speakers_no_diarization(self):
        """Test behavior when no diarization data is provided."""
        speech_segments = [{"start": 0.0, "end": 5.0}]
        diarization = {}

        result = assign_speakers_to_speech_segment(speech_segments, diarization)

        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0].get("speaker_id"))

    def test_assign_speakers_duration_calculation(self):
        """Test that duration is calculated correctly."""
        speech_segments = [{"start": 1.5, "end": 4.7}]
        diarization = {"SPEAKER_00": {"start": 0.0, "end": 10.0}}
        result = assign_speakers_to_speech_segment(speech_segments, diarization)
        self.assertAlmostEqual(result[0]["duration"], 3.2, places=6)


class TestSilenceIdentification(unittest.TestCase):
    def test_identify_silence_gaps(self):
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
        self.assertEqual(len(silences), 2)

        first, second = silences
        self.assertAlmostEqual(first["start"], 3.0)
        self.assertAlmostEqual(first["end"], 5.0)
        self.assertAlmostEqual(first["duration"], 2.0)
        self.assertAlmostEqual(first["event_type"], EventType.SILENCE)

        self.assertAlmostEqual(second["start"], 8.0)
        self.assertAlmostEqual(second["end"], 10.0)
        self.assertAlmostEqual(second["duration"], 2.0)
        self.assertAlmostEqual(second["event_type"], EventType.SILENCE)

    def test_no_silence_when_too_short(self):
        speech_segments = [{"start": 0.0, "end": 1.0}, {"start": 1.4, "end": 2.0}]
        non_speech_segments = []
        silences = identify_silence_gaps(
            speech_segments,
            non_speech_segments,
            total_audio_duration=3.0,
            min_silence_duration=0.5,
        )
        self.assertEqual(len(silences), 0)


if __name__ == "__main__":
    unittest.main()
