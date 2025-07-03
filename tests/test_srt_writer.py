import tempfile
import os

from captionalchemy.tools.captioning.writers.srt_writer import SRTCaptionWriter


def test_add_caption_text_and_silence_music():
    writer = SRTCaptionWriter()

    # Speech with text
    writer.add_caption(
        0.0, 1.0, speaker="SPEAKER_00", text="Hello world.", event_type="speech"
    )
    # Short silence (should skip)
    writer.add_caption(1.0, 1.2, speaker="SPEAKER_00", text="", event_type="silence")
    # Long silence (should add)
    writer.add_caption(1.2, 3.0, speaker="SPEAKER_00", text="", event_type="silence")
    # Music event (should add)
    writer.add_caption(3.0, 4.5, event_type="music")
    # Music event (should add)
    writer.add_caption(4.5, 6.0, event_type="music")

    caps = writer._captions
    # Expect 1 speech, 1 silence, and 1 music caption, 3 entries total
    assert len(caps) == 4
    assert caps[0]["text"] == "Hello world."
    assert caps[1]["text"] == "[SILENCE]"
    assert caps[2]["text"] == "[MUSIC PLAYING]"


def test_write_srt_file():
    writer = SRTCaptionWriter()

    # Two entries with same speaker
    writer.add_caption(
        0.0, 1.0, speaker="SPEAKER_00", text="Hello world.", event_type="speech"
    )
    writer.add_caption(
        1.0, 2.0, speaker="SPEAKER_00", text="This is a test.", event_type="speech"
    )
    # Different speaker
    writer.add_caption(
        2.0, 3.0, speaker="SPEAKER_01", text="Goodbye.", event_type="speech"
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".srt") as tmp:
        path = tmp.name

    try:
        writer.write(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().splitlines()

        # Check index and timings in first block
        assert content[0] == "1"
        assert "-->" in content[1]
        # Speaker line for first caption
        assert "SPEAKER_00: Hello world." in content[2]
        # Next blocks exist
        assert "2" in content
        assert "3" in content
    finally:
        os.remove(path)
