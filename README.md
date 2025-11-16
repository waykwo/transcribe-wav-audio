# Audio Transcription with Speaker Diarization

A Python script that transcribes audio files and identifies different speakers using OpenAI's Whisper and pyannote.audio for speaker diarization.

## Features

- 🎙️ **Accurate Transcription**: Uses OpenAI's Whisper model for high-quality speech-to-text
- 👥 **Speaker Identification**: Automatically identifies and labels different speakers
- ⏱️ **Timestamps**: Includes precise timestamps for each segment
- 🧪 **Test Mode**: Process only the first N seconds for quick testing
- 🍎 **Mac Optimization**: Supports Metal (MPS) acceleration on Apple Silicon
- 📝 **Formatted Output**: Clean, readable transcript format

## Requirements

- Python 3.9+
- HuggingFace account with access token
- ffmpeg (for audio processing)

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/waykwo/transcribe-wav-audio.git
   cd transcribe-wav-audio
   ```

2. **Create and activate virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install openai-whisper pyannote.audio torch torchaudio pydub
   pip install 'huggingface_hub<1.0.0'  # Required for compatibility
   ```

4. **Install ffmpeg** (if not already installed)
   ```bash
   brew install ffmpeg
   ```

## Setup

1. **Get HuggingFace Access Token**

   - Create an account at [HuggingFace](https://huggingface.co/)
   - Generate an access token at [https://hf.co/settings/tokens](https://hf.co/settings/tokens)
   - Save your token in a text file (e.g., `hf_token.txt`)

2. **Accept Model Terms**

   You must accept the terms for these models on HuggingFace:

   - [pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://hf.co/pyannote/segmentation-3.0)

3. **Configure Token Path**

   Edit line 219 in `transcribe-meeting.py`:

   ```python
   HF_TOKEN_FILE = "/path/to/your/hf_token.txt"
   ```

4. **Prepare Audio**

   Place your audio file in the `./audio` directory:

   ```bash
   mkdir audio
   cp your-audio-file.m4a ./audio/audio.m4a
   ```

## Usage

### Basic Usage

Process entire audio file with default settings:

```bash
python transcribe-meeting.py
```

### Test Mode

Process only the first 60 seconds (recommended for initial testing):

```bash
python transcribe-meeting.py --test 60
```

### Custom Options

```bash
python transcribe-meeting.py --audio ./audio/meeting.m4a --speakers 3 --model medium --test 120
```

### Command-Line Arguments

| Argument     | Description                                                     | Default             |
| ------------ | --------------------------------------------------------------- | ------------------- |
| `--audio`    | Path to audio file                                              | `./audio/audio.m4a` |
| `--speakers` | Number of speakers                                              | `2`                 |
| `--model`    | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) | `small`             |
| `--test`     | Test mode: process only first N seconds                         | `None` (full file)  |

### Get Help

```bash
python transcribe-meeting.py --help
```

## Output

The script generates a formatted transcript in the `./output` directory with speaker labels and timestamps.

## Model Sizes

Choose a Whisper model based on your needs:

| Model    | Size    | Speed    | Quality |
| -------- | ------- | -------- | ------- |
| `tiny`   | 39 MB   | Fastest  | Basic   |
| `base`   | 74 MB   | Fast     | Good    |
| `small`  | 244 MB  | Moderate | Better  |
| `medium` | 769 MB  | Slow     | Great   |
| `large`  | 1550 MB | Slowest  | Best    |

## Project Structure

```
transcribe-wav-audio/
├── transcribe-meeting.py    # Main script
├── .gitignore               # Git ignore rules
├── README.md                # This file
├── audio/                   # Audio files (gitignored)
│   └── audio.m4a
└── output/                  # Transcripts (gitignored)
    └── audio_transcript.txt
```

## Troubleshooting

### "Could not download pipeline" Error

- Ensure you've accepted the model terms on HuggingFace
- Verify your access token is correct and has the necessary permissions

### ImportError or Version Conflicts

- Make sure `huggingface_hub` version is < 1.0.0
- Reinstall with: `pip install 'huggingface_hub<1.0.0'`

### Slow Performance

- Use a smaller Whisper model (`tiny` or `base`)
- Use test mode to process shorter segments
- On Apple Silicon Macs, ensure MPS acceleration is enabled

## License

MIT License - Feel free to use and modify for your needs.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) - Speaker diarization
