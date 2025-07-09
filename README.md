# CaptionAlchemy

A Python package for creating intelligent closed captions with face detection and speaker recognition.

## Features

- **Audio Transcription**: Powered by OpenAI Whisper for high-quality speech-to-text
- **Speaker Diarization**: Identifies different speakers in audio
- **Face Recognition**: Links speakers to known faces for character identification
- **Multiple Output Formats**: Supports SRT, VTT, and SAMI caption formats
- **Voice Activity Detection**: Intelligently detects speech vs non-speech segments
- **GPU Acceleration**: Automatic CUDA support when available

## Installation

```bash
pip install captionalchemy
```

If you have a GPU and want to use hardware acceleration:

```bash
pip install captionalchemy[cuda]
```

### Prerequisites

- Python 3.10+
- FFmpeg (for video/audio processing)
- CUDA-capable GPU (optional, for acceleration but is highly recommended for the diarization)
- Whisper.cpp capable (optional on MacOS)

If using Whisper.cpp on MacOS, follow installation instructions [[here](https://github.com/ggml-org/whisper.cpp?tab=readme-ov-file#core-ml-support)]
Clone the whisper repo into your working directory.

## Quick Start

1. **Set up environment variables** (create `.env` file):

   ```
   HF_AUTH_TOKEN=your_huggingface_token_here
   ```

2. **Prepare known faces** (optional, for speaker identification):
   Create `known_faces.json`:

   ```json
   [
     {
       "name": "Speaker Name",
       "image_path": "path/to/speaker/photo.jpg"
     }
   ]
   ```

3. **Generate captions**:

```bash
captionalchemy video.mp4 -f srt -o my_captions
```

or in a python script

```python
from dotenv import load_dotenv
from captionalchemy import caption

load_dotenv()

caption.run_pipeline(
    video_url_or_path="path/to/your/video.mp4",         # this can be a video URL or local file
    character_identification=False,                      # True by default
    known_faces_json="path/to/known_faces.json",
    embed_faces_json="path/to/embed_faces.json",        # name of the output file
    caption_output_path="my_captions/output",           # will write output to output.srt (or .vtt/.smi)
    caption_format="srt"
)
```

## Usage

### Basic Usage

```bash
# Generate SRT captions from video file
captionalchemy video.mp4

# Generate VTT captions from YouTube URL
captionalchemy "https://youtube.com/watch?v=VIDEO_ID" -f vtt -o output

# Disable face recognition
captionalchemy video.mp4 --no-face-id
```

### Command Line Options

```
captionalchemy VIDEO [OPTIONS]

Arguments:
  VIDEO                Video file path or URL

Options:
  -f, --format         Caption format: srt, vtt, smi (default: srt)
  -o, --output         Output file base name (default: output_captions)
  --no-face-id         Disable face recognition
  --known-faces-json   Path to known faces JSON (default: example/known_faces.json)
  --embed-faces-json   Path to face embeddings JSON (default: example/embed_faces.json)
  -v, --verbose        Enable debug logging
```

## How It Works

1. **Face Embedding**: Pre-processes known faces into embeddings
2. **Audio Extraction**: Extracts audio from video files
3. **Voice Activity Detection**: Identifies speech segments
4. **Speaker Diarization**: Separates different speakers
5. **Transcription**: Converts speech to text using Whisper
6. **Face Recognition**: Matches speakers to known faces (if enabled)
7. **Caption Generation**: Creates timestamped captions with speaker names

## Configuration

### Known Faces Setup

Create a `known_faces.json` file with speaker information:

```json
[
  {
    "name": "John Doe",
    "image_path": "photos/john_doe.jpg"
  },
  {
    "name": "Jane Smith",
    "image_path": "photos/jane_smith.png"
  }
]
```

### Environment Variables

- `HF_AUTH_TOKEN`: Hugging Face token for accessing pyannote models

## Output Examples

### SRT Format

```
1
00:00:03,254 --> 00:00:06,890
John Doe: Welcome to our presentation on quantum computing.

2
00:00:07,120 --> 00:00:10,456
Jane Smith: Thanks John. Let's start with the basics.
```

### VTT Format

```
WEBVTT

00:03.254 --> 00:06.890
John Doe: Welcome to our presentation on quantum computing.

00:07.120 --> 00:10.456
Jane Smith: Thanks John. Let's start with the basics.
```

## Development and Contributing

### Setup Development Environment

```bash
# Install in development mode
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

### Code Quality

```bash
# Linting
flake8

# Code formatting
black src/ tests/
```

## Requirements

See `requirements.txt` for the complete list of dependencies. Key packages include:

- `openai-whisper`: Speech transcription
- `pyannote.audio`: Speaker diarization
- `opencv-python`: Computer vision
- `insightface`: Face recognition
- `torch`: Deep learning framework

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## Troubleshooting

### Common Issues

- **CUDA out of memory**: Use CPU-only mode or reduce batch sizes
- **Missing models**: Ensure whisper.cpp models are downloaded
- **Face recognition errors**: Verify image paths in known_faces.json
- **Audio extraction fails**: Check that FFmpeg is installed

### Getting Help

- Check the logs with `-v` flag for detailed error information
- Ensure all dependencies are properly installed
- Verify video file format compatibility

```

```
