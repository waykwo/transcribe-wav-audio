#!/usr/bin/env python3
"""
Audio transcription script with speaker diarization.
Transcribes .m4a audio files with speaker labels and timestamps.
"""

import whisper
from pyannote.audio import Pipeline
from pydub import AudioSegment
import torch
from pathlib import Path
from typing import List, Dict, Tuple
import sys
import argparse


def load_audio(audio_path: str, duration_seconds: int = None) -> str:
    """
    Load and convert audio file to WAV format if needed.

    Args:
        audio_path: Path to the audio file
        duration_seconds: Optional duration in seconds to trim audio (for testing)

    Returns:
        Path to the WAV file
    """
    print(f"Loading audio file: {audio_path}")
    audio = AudioSegment.from_file(audio_path)

    # Trim audio if duration is specified (for testing)
    if duration_seconds is not None:
        duration_ms = duration_seconds * 1000
        audio = audio[:duration_ms]
        print(f"Trimmed to {duration_seconds} seconds for testing")

    # Convert to WAV for compatibility
    wav_path = audio_path.rsplit('.', 1)[0] + '_temp.wav'
    audio.export(wav_path, format='wav')
    print(f"Converted to WAV: {wav_path}")

    return wav_path


def transcribe_audio(audio_path: str, model_size: str = "base") -> Dict:
    """
    Transcribe audio using Whisper.

    Args:
        audio_path: Path to the audio file
        model_size: Whisper model size (tiny, base, small, medium, large)

    Returns:
        Transcription result with segments and timestamps
    """
    print(f"\nLoading Whisper model ({model_size})...")
    model = whisper.load_model(model_size)

    print("Transcribing audio... This may take several minutes.")
    result = model.transcribe(audio_path, verbose=True)

    return result


def diarize_audio(audio_path: str, hf_token: str, num_speakers: int = 2) -> List[Tuple[float, float, str]]:
    """
    Perform speaker diarization using pyannote.

    Args:
        audio_path: Path to the audio file
        hf_token: Hugging Face authentication token
        num_speakers: Number of speakers in the audio

    Returns:
        List of tuples (start_time, end_time, speaker_label)
    """
    print("\nLoading speaker diarization model...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )

    # Use MPS (Metal Performance Shaders) for M-series Macs if available
    if torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))
        print("Using MPS (Metal) acceleration")

    print("Performing speaker diarization... This may take several minutes.")
    diarization = pipeline(audio_path, num_speakers=num_speakers)

    # Extract speaker segments
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append((turn.start, turn.end, speaker))

    return segments


def assign_speakers_to_transcription(
    transcription: Dict,
    diarization: List[Tuple[float, float, str]]
) -> List[Dict[str, any]]:
    """
    Combine transcription segments with speaker labels.

    Args:
        transcription: Whisper transcription result
        diarization: List of speaker segments

    Returns:
        List of segments with text, timestamps, and speaker labels
    """
    print("\nCombining transcription with speaker labels...")

    result_segments = []

    for segment in transcription['segments']:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text'].strip()

        # Find the speaker for this segment (using midpoint)
        midpoint = (start_time + end_time) / 2
        speaker = "Unknown"

        for diar_start, diar_end, diar_speaker in diarization:
            if diar_start <= midpoint <= diar_end:
                speaker = diar_speaker
                break

        result_segments.append({
            'start': start_time,
            'end': end_time,
            'speaker': speaker,
            'text': text
        })

    return result_segments


def format_timestamp(seconds: float) -> str:
    """
    Format seconds into HH:MM:SS timestamp.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def save_transcript(segments: List[Dict[str, any]], output_path: str) -> None:
    """
    Save the transcript to a text file.

    Args:
        segments: List of transcript segments with speaker labels
        output_path: Path to save the transcript
    """
    print(f"\nSaving transcript to: {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("MEETING TRANSCRIPT\n")
        f.write("=" * 80 + "\n\n")

        current_speaker = None

        for segment in segments:
            speaker = segment['speaker']
            timestamp = format_timestamp(segment['start'])
            text = segment['text']

            # Add extra line break when speaker changes
            if speaker != current_speaker:
                if current_speaker is not None:
                    f.write("\n")
                current_speaker = speaker

            f.write(f"[{timestamp}] {speaker}: {text}\n")

    print("Transcript saved successfully!")


def main():
    """Main execution function."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Transcribe audio with speaker diarization')
    parser.add_argument('--test', type=int, metavar='SECONDS',
                        help='Test mode: process only first N seconds of audio')
    parser.add_argument('--audio', type=str, default='./audio/audio.m4a',
                        help='Path to audio file (default: ./audio/audio.m4a)')
    parser.add_argument('--speakers', type=int, default=2,
                        help='Number of speakers (default: 2)')
    parser.add_argument('--model', type=str, default='small',
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper model size (default: small)')
    args = parser.parse_args()

    # Configuration
    AUDIO_FILE = args.audio
    # Path to file containing your HuggingFace token
    HF_TOKEN_FILE = "/Volumes/Encrypted/hugging-face-token.txt"

    # Read token from file
    with open(HF_TOKEN_FILE, 'r') as f:
        HF_TOKEN = f.read().strip()

    NUM_SPEAKERS = args.speakers
    WHISPER_MODEL = args.model
    TEST_DURATION = args.test  # None if not specified, otherwise number of seconds

    # Check if paths need updating
    if "path/to/your" in AUDIO_FILE:
        print("ERROR: Please update AUDIO_FILE path in the script")
        sys.exit(1)

    if "your_huggingface_token" in HF_TOKEN:
        print("ERROR: Please update HF_TOKEN in the script")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)

    # Generate output filename
    output_file = output_dir / (Path(AUDIO_FILE).stem + "_transcript.txt")

    print("Starting transcription process...")
    print(f"Audio file: {AUDIO_FILE}")
    print(f"Output file: {output_file}")
    print(f"Number of speakers: {NUM_SPEAKERS}")
    print(f"Whisper model: {WHISPER_MODEL}")
    if TEST_DURATION:
        print(f"TEST MODE: Processing only first {TEST_DURATION} seconds")
    print("-" * 80)

    # Step 1: Convert audio to WAV
    wav_path = load_audio(AUDIO_FILE, duration_seconds=TEST_DURATION)

    try:
        # Step 2: Transcribe audio
        transcription = transcribe_audio(wav_path, model_size=WHISPER_MODEL)

        # Step 3: Perform speaker diarization
        diarization = diarize_audio(
            wav_path, HF_TOKEN, num_speakers=NUM_SPEAKERS)

        # Step 4: Combine transcription with speaker labels
        segments = assign_speakers_to_transcription(transcription, diarization)

        # Step 5: Save transcript
        save_transcript(segments, output_file)

        print("\n" + "=" * 80)
        print("TRANSCRIPTION COMPLETE!")
        print(f"Total segments: {len(segments)}")
        print(f"Output file: {output_file}")

    finally:
        # Clean up temporary WAV file
        if Path(wav_path).exists():
            Path(wav_path).unlink()
            print(f"\nCleaned up temporary file: {wav_path}")


if __name__ == "__main__":
    main()
