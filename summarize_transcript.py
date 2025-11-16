#!/usr/bin/env python3
"""
Meeting Transcript Summarizer using Ollama and Llama 3.1

This script summarizes meeting transcripts using a local Llama 3.1 model via Ollama.
All processing is done locally with no cloud or API calls.

Usage:
    python summarize_transcript.py <transcript_file> --detail <level>
    
    Detail levels: brief, medium, detailed
    
Example:
    python summarize_transcript.py output/audio_transcript.txt --detail medium
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import ollama


# Detail level prompts
DETAIL_PROMPTS: Dict[str, str] = {
    "brief": (
        "Summarize this meeting transcript in 3-5 bullet points covering only "
        "the most critical information."
    ),
    "medium": (
        "Summarize this meeting with:\n"
        "1) Main topics discussed\n"
        "2) Key decisions made\n"
        "3) Action items and owners\n"
        "4) Next steps"
    ),
    "detailed": (
        "Provide a comprehensive summary including:\n"
        "- All topics discussed with context\n"
        "- All decisions and rationale\n"
        "- Concerns raised\n"
        "- Action items with owners and deadlines\n"
        "- Any follow-up needed"
    ),
}


def read_transcript(file_path: str) -> str:
    """
    Read the transcript file from disk.

    Args:
        file_path: Path to the transcript file

    Returns:
        Content of the transcript as a string

    Raises:
        FileNotFoundError: If the transcript file doesn't exist
        IOError: If there's an error reading the file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if not content.strip():
            raise ValueError(f"Transcript file is empty: {file_path}")

        return content
    except FileNotFoundError:
        raise FileNotFoundError(f"Transcript file not found: {file_path}")
    except Exception as e:
        raise IOError(f"Error reading transcript file: {e}")


def generate_summary(
    transcript: str,
    detail_level: str,
    model: str = "llama3.1"
) -> str:
    """
    Generate a summary of the transcript using Ollama with Llama 3.1.

    Args:
        transcript: The transcript text to summarize
        detail_level: Level of detail for summary (brief, medium, detailed)
        model: Ollama model to use (default: llama3.1)

    Returns:
        The generated summary as a string

    Raises:
        ValueError: If detail_level is invalid
        Exception: If there's an error communicating with Ollama
    """
    if detail_level not in DETAIL_PROMPTS:
        raise ValueError(
            f"Invalid detail level: {detail_level}. "
            f"Must be one of: {', '.join(DETAIL_PROMPTS.keys())}"
        )

    prompt = DETAIL_PROMPTS[detail_level]

    print(f"Generating {detail_level} summary using {model}...")
    print("This may take a minute depending on transcript length...")

    try:
        # Create the full prompt with transcript
        full_prompt = f"{prompt}\n\nTranscript:\n{transcript}"

        # Call Ollama API
        response = ollama.chat(
            model=model,
            messages=[
                {
                    'role': 'user',
                    'content': full_prompt,
                }
            ]
        )

        summary = response['message']['content']

        if not summary.strip():
            raise ValueError("Ollama returned an empty summary")

        return summary

    except Exception as e:
        raise Exception(f"Error generating summary with Ollama: {e}")


def save_summary(
    summary: str,
    original_file_path: str,
    detail_level: str,
    output_dir: str = "./output"
) -> str:
    """
    Save the summary to the output directory.

    Args:
        summary: The summary text to save
        original_file_path: Path to the original transcript file
        detail_level: Detail level used for the summary
        output_dir: Directory to save the summary (default: ./output)

    Returns:
        Path to the saved summary file

    Raises:
        IOError: If there's an error writing the file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate output filename
    original_name = Path(original_file_path).stem
    output_filename = f"{original_name}_summary_{detail_level}.txt"
    output_path = os.path.join(output_dir, output_filename)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"MEETING SUMMARY ({detail_level.upper()})\n")
            f.write("=" * 80)
            f.write(f"\n\nOriginal Transcript: {original_file_path}\n")
            f.write(f"Detail Level: {detail_level}\n")
            f.write(f"Model: llama3.1\n")
            f.write("=" * 80)
            f.write("\n\n")
            f.write(summary)
            f.write("\n")

        return output_path

    except Exception as e:
        raise IOError(f"Error saving summary: {e}")


def main() -> None:
    """
    Main function to handle command-line arguments and orchestrate summarization.
    """
    parser = argparse.ArgumentParser(
        description="Summarize meeting transcripts using local Ollama and Llama 3.1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python summarize_transcript.py output/audio_transcript.txt --detail brief
  python summarize_transcript.py meeting.txt --detail medium
  python summarize_transcript.py transcript.txt --detail detailed --model llama3.1

Detail Levels:
  brief    - 3-5 bullet points with critical information only
  medium   - Main topics, decisions, action items, next steps
  detailed - Comprehensive summary with all context and details
        """
    )

    parser.add_argument(
        'transcript_file',
        type=str,
        help='Path to the transcript file to summarize'
    )

    parser.add_argument(
        '--detail',
        type=str,
        choices=['brief', 'medium', 'detailed'],
        default='medium',
        help='Level of detail for the summary (default: medium)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='llama3.1',
        help='Ollama model to use (default: llama3.1)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='./output',
        help='Output directory for summary (default: ./output)'
    )

    args = parser.parse_args()

    try:
        # Read transcript
        print(f"Reading transcript from: {args.transcript_file}")
        transcript = read_transcript(args.transcript_file)
        print(f"✓ Transcript loaded ({len(transcript)} characters)")

        # Generate summary
        summary = generate_summary(
            transcript=transcript,
            detail_level=args.detail,
            model=args.model
        )
        print(f"✓ Summary generated ({len(summary)} characters)")

        # Save summary
        output_path = save_summary(
            summary=summary,
            original_file_path=args.transcript_file,
            detail_level=args.detail,
            output_dir=args.output
        )
        print(f"✓ Summary saved to: {output_path}")

        print("\n" + "=" * 80)
        print("SUMMARY PREVIEW:")
        print("=" * 80)
        print(summary[:500] + "..." if len(summary) > 500 else summary)
        print("=" * 80)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nMake sure Ollama is running and the llama3.1 model is installed:")
        print("  1. Install Ollama: https://ollama.ai")
        print("  2. Pull the model: ollama pull llama3.1")
        print("  3. Verify it's running: ollama list")
        sys.exit(1)


if __name__ == "__main__":
    main()
