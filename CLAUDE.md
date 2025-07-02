# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CaptionAlchemy is a Python package for creating closed captions with face detection and recognition. It combines audio transcription, speaker diarization, and facial recognition to generate accurate subtitles with speaker identification.

## Architecture

The package follows a modular structure under `src/captionalchemy/`:

- **Main Pipeline**: `caption.py` - Core orchestration of the caption generation process
- **Audio Analysis**: `tools/audio_analysis/` - Voice activity detection, speaker diarization, and non-speech detection
- **Captioning**: `tools/captioning/` - Whisper transcription, timing analysis, and output writers (SRT, VTT, SAMI)
- **Computer Vision**: `tools/cv/` - Face embedding and recognition for speaker identification
- **Media Utils**: `tools/media_utils/` - Video download and audio extraction utilities

## Key Components

### Core Pipeline Flow

1. Embed known faces from JSON configuration
2. Download/extract audio from video
3. Run Voice Activity Detection (VAD) and speaker diarization
4. Transcribe audio segments with Whisper
5. Identify speakers using facial recognition
6. Generate timestamped captions in chosen format

### Dependencies

- **Audio Processing**: pyannote.audio, pydub, librosa, openai-whisper
- **Computer Vision**: opencv-python, insightface, onnxruntime
- **Deep Learning**: Uses CUDA when available, falls back to CPU

## Development Commands

### Installation

```bash
pip install -e .
```

### Testing

```bash
pytest
```

### Linting

```bash
flake8
mypy src/
```

### Running the Package

```bash
captionalchemy <video_path_or_url> -f srt -o output_captions
```

## Configuration

- **Environment**: Uses `.env` file for configuration (HF_AUTH_TOKEN for Hugging Face models)
- **Face Recognition**: Requires `known_faces.json`. `embed_faces.json` is an artifact generated.
- **Whisper**: Integrates with whisper.cpp for transcription (requires models in `whisper.cpp/models/`)

## External Dependencies

- Whisper.cpp integration for transcription
- Hugging Face models for audio analysis
- ONNX runtime for face recognition models

## Code Style

- Line length: 110 characters (flake8 configured)
- Python 3.10+ compatible
- Type hints encouraged
