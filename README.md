# Audio Transcription with Speaker Diarization

A Python script that transcribes audio files and identifies different speakers using OpenAI's Whisper and pyannote.audio for speaker diarization.

## 🔒 Privacy First

**Complete privacy guaranteed - 100% local processing.**

- ✅ Your audio files **never leave your computer**
- ✅ Transcription runs entirely on your machine (no OpenAI API calls)
- ✅ Summarization uses local Ollama (no cloud LLM services)
- ✅ No data sent to any external servers or third parties
- ✅ Perfect for confidential meetings, sensitive conversations, or private content

**Even though this project uses "OpenAI's Whisper", it runs the open-source model locally on your machine - not through OpenAI's servers.**

## Features

- 🔐 **100% Local & Private**: All transcription, speaker ID, and AI summarization run locally - zero cloud usage
- 🎙️ **Accurate Transcription**: Uses OpenAI's Whisper model for high-quality speech-to-text
- 👥 **Speaker Identification**: Automatically identifies and labels different speakers
- 🤖 **AI-Powered Summaries**: Generate brief, medium, or detailed summaries using local Llama 3.1
- 🎵 **Multiple Audio Formats**: Supports MP3, M4A, WAV, FLAC, OGG, and more
- ⏱️ **Timestamps**: Includes precise timestamps for each segment
- 🧪 **Test Mode**: Process only the first N seconds for quick testing
- 🍎 **Mac Optimization**: Supports Metal (MPS) acceleration on Apple Silicon
- 📝 **Formatted Output**: Clean, readable transcript and summary formats

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
   pip install openai-whisper pyannote.audio torch torchaudio pydub ollama
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

**Example format:**

```
[00:00] SPEAKER_1: [Transcribed content here]
[00:15] SPEAKER_2: [Transcribed content here]
[00:32] SPEAKER_1: [Transcribed content here]
```

All transcripts are saved locally and never uploaded anywhere.

## Model Sizes

Choose a Whisper model based on your needs:

| Model    | Size    | Speed    | Quality |
| -------- | ------- | -------- | ------- |
| `tiny`   | 39 MB   | Fastest  | Basic   |
| `base`   | 74 MB   | Fast     | Good    |
| `small`  | 244 MB  | Moderate | Better  |
| `medium` | 769 MB  | Slow     | Great   |
| `large`  | 1550 MB | Slowest  | Best    |

## Transcript Summarization

After transcribing your audio, you can generate AI-powered summaries using local LLMs via Ollama.

### 🔒 Privacy: Local AI Summarization

**All summarization processing happens on your machine using Ollama:**

- ✅ Your transcripts never leave your computer
- ✅ No ChatGPT, Claude, or other cloud AI services used
- ✅ Llama 3.1 model runs entirely locally
- ✅ Perfect for confidential meeting summaries and sensitive content

### Prerequisites

1. **Install Ollama** (if not already installed)

   - Download from [ollama.com](https://ollama.com/download)
   - Or install via Homebrew: `brew install ollama`

2. **Pull Llama 3.1 model**

   ```bash
   ollama pull llama3.1
   ```

3. **Start Ollama** (if not running)
   ```bash
   ollama serve
   ```

### Usage

Generate summaries at different detail levels:

```bash
# Brief summary (3-5 bullet points)
python summarize_transcript.py output/audio_transcript.txt --detail brief

# Medium summary (main topics, decisions, action items)
python summarize_transcript.py output/audio_transcript.txt --detail medium

# Detailed summary (comprehensive with full context)
python summarize_transcript.py output/audio_transcript.txt --detail detailed

# Comprehensive summary (exhaustive report with all discussion points)
python summarize_transcript.py output/audio_transcript.txt --detail comprehensive
```

### Summary Options

| Detail Level    | Description                                                        | Best For                         |
| --------------- | ------------------------------------------------------------------ | -------------------------------- |
| `brief`         | 3-5 bullet points with critical information only                   | Quick overviews, executives      |
| `medium`        | Main topics, decisions, action items, next steps                   | Team updates, status reports     |
| `detailed`      | Comprehensive with context, rationale, concerns, deadlines         | Complete documentation           |
| `comprehensive` | Exhaustive report preserving all viewpoints and discussion details | Legal records, thorough archives |

### Output

Summaries are saved to `./output` with the format: `[original_name]_summary_[detail_level].txt`

Example filename: `meeting_transcript_summary_medium.txt`

**Example summary structure:**

```
MEETING SUMMARY (MEDIUM)
================================================================================

Main Topics:
- [Topic 1]
- [Topic 2]

Key Decisions:
- [Decision 1]
- [Decision 2]

Action Items:
- [Action with owner]

Next Steps:
- [Follow-up items]
```

🔒 **All summaries are generated and stored locally** - no data transmitted to external AI services.

## Project Structure

```
transcribe-wav-audio/
├── transcribe-meeting.py    # Main transcription script
├── summarize_transcript.py  # Summary generation script
├── .gitignore               # Git ignore rules
├── README.md                # This file
├── audio/                   # Audio files (gitignored)
│   └── audio.m4a
└── output/                  # Transcripts & summaries (gitignored)
    ├── audio_transcript.txt
    └── audio_transcript_summary_medium.txt
```

## Troubleshooting

### Transcription Issues

**"Could not download pipeline" Error**

- Ensure you've accepted the model terms on HuggingFace
- Verify your access token is correct and has the necessary permissions

**ImportError or Version Conflicts**

- Make sure `huggingface_hub` version is < 1.0.0
- Reinstall with: `pip install 'huggingface_hub<1.0.0'`

**Slow Performance**

- Use a smaller Whisper model (`tiny` or `base`)
- Use test mode to process shorter segments
- On Apple Silicon Macs, ensure MPS acceleration is enabled

### Summarization Issues

**"Failed to connect to Ollama" Error**

- Verify Ollama is installed: `which ollama`
- Check if Ollama is running: `ollama list`
- Start Ollama service: `ollama serve`

**Model Not Found**

- Pull the Llama 3.1 model: `ollama pull llama3.1`
- Verify available models: `ollama list`

**Summary Quality Issues**

- Try a different detail level (`brief`, `medium`, `detailed`)
- For very long transcripts, consider breaking into smaller segments
- Different Ollama models may produce varying results: `--model llama3.1`

## License

MIT License - Feel free to use and modify for your needs.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) - Speaker diarization
