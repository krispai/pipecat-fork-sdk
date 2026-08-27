#!/usr/bin/env python3
"""Standalone script to run a real audio file through a real STT service.

Builds a real `Pipeline([input_transport, stt])` (the same code path a bot
uses), feeds it a WAV file's audio, and prints the resulting transcript. Use
this to compare what your STT actually transcribes with `use_filtered_audio`
on (the default) vs. off, on your own recordings.

Usage:
    python test_stt_audiofile.py input.wav --stt whisper-local
    python test_stt_audiofile.py input.wav --stt whisper-local --filter
    python test_stt_audiofile.py input.wav --stt whisper-local --filter --no-use-filtered-audio
    python test_stt_audiofile.py input.wav --stt deepgram --filter --level 80

Requirements:
    uv add soundfile numpy
    STT provider of choice (edit `build_stt_service` to add your own):
      --stt whisper-local  uv add "pipecat-ai[whisper]"    (local model, no API key)
      --stt openai         uv add "pipecat-ai[openai]",    set OPENAI_API_KEY
      --stt deepgram       uv add "pipecat-ai[deepgram]",  set DEEPGRAM_API_KEY
    For --filter, also:
      uv add "pipecat-ai[krisp]"
      Set KRISP_VIVA_FILTER_MODEL_PATH to your .kef model file
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

try:
    import numpy as np
    import soundfile as sf  # noqa: F401
    from audio_file_utils import read_audio_file
except ImportError as e:
    print(f"Error: Missing required dependencies: {e}")
    print("Install with: uv add soundfile numpy")
    sys.exit(1)

# Add src directory to Python path for development environment
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
src_dir = project_root / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from pipecat.frames.frames import (
    ErrorFrame,
    InputAudioRawFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams
from pipecat.services.stt_service import STTService
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_transport import TransportParams


class ReadyInputTransport(BaseInputTransport):
    """Input transport that reports itself ready as soon as it starts.

    Concrete transports (Daily, WebRTC, ...) call `set_transport_ready` once
    actually connected; there is no live connection to wait on here.
    """

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.set_transport_ready(frame)


def build_stt_service(name: str, sample_rate: int, use_filtered_audio: bool) -> STTService:
    """Build the STT service to test against.

    Add your own provider here following the same pattern: import lazily (so
    running the script doesn't require every provider's SDK installed) and
    forward `use_filtered_audio`.
    """
    if name == "whisper-local":
        from pipecat.services.whisper.stt import WhisperSTTService

        return WhisperSTTService(sample_rate=sample_rate, use_filtered_audio=use_filtered_audio)
    elif name == "openai":
        from pipecat.services.openai.stt import OpenAISTTService

        return OpenAISTTService(
            api_key=os.environ["OPENAI_API_KEY"],
            sample_rate=sample_rate,
            use_filtered_audio=use_filtered_audio,
        )
    elif name == "deepgram":
        from pipecat.services.deepgram.stt import DeepgramSTTService

        return DeepgramSTTService(
            api_key=os.environ["DEEPGRAM_API_KEY"],
            sample_rate=sample_rate,
            use_filtered_audio=use_filtered_audio,
        )
    raise ValueError(f"Unknown --stt provider: {name}")


def build_audio_filter(args):
    """Build the Krisp VIVA filter, or None if --filter wasn't passed."""
    if not args.filter:
        return None

    from pipecat.audio.filters.krisp_viva_filter import KrispVivaFilter

    model_path = args.filter_model or os.getenv("KRISP_VIVA_FILTER_MODEL_PATH")
    if not model_path or not os.path.isfile(model_path):
        print("Error: --filter requires a valid --filter-model or KRISP_VIVA_FILTER_MODEL_PATH")
        sys.exit(1)

    return KrispVivaFilter(model_path=model_path, noise_suppression_level=args.level)


def chunk_audio(
    audio_data: np.ndarray, sample_rate: int, chunk_ms: int = 20
) -> list[InputAudioRawFrame]:
    """Split int16 mono audio into `InputAudioRawFrame`s at `chunk_ms` cadence."""
    chunk_size = int(sample_rate * chunk_ms / 1000)
    frames = []
    for i in range(0, len(audio_data) - len(audio_data) % chunk_size, chunk_size):
        chunk = audio_data[i : i + chunk_size]
        frames.append(
            InputAudioRawFrame(audio=chunk.tobytes(), sample_rate=sample_rate, num_channels=1)
        )
    return frames


async def transcribe_file(args) -> None:
    """Feed `args.input` through a real transport + STT pipeline and print the transcript."""
    audio_data, sample_rate = read_audio_file(args.input, verbose=args.verbose)

    audio_filter = build_audio_filter(args)
    stt = build_stt_service(args.stt, sample_rate, use_filtered_audio=args.use_filtered_audio)

    transport = ReadyInputTransport(
        TransportParams(audio_in_enabled=True, audio_in_filter=audio_filter)
    )

    audio_frames = chunk_audio(audio_data, sample_rate)
    duration_secs = len(audio_data) / sample_rate

    # Segmented/network STT only runs after VADUserStoppedSpeakingFrame and can
    # take real wall-clock time (CPU Whisper inference is roughly real-time or
    # slower). Default to generous headroom scaled to the file's own length;
    # override with --wait-secs if your provider is slower still.
    wait_secs = args.wait_secs if args.wait_secs is not None else max(5.0, duration_secs)

    # STT only ever reads frame.filtered_audio (Krisp output) when BOTH a
    # filter is configured AND use_filtered_audio is True. Any other
    # combination means STT transcribes frame.audio, i.e. the ORIGINAL,
    # untouched signal -- identical bytes whether or not a filter is even
    # configured, since the filter never modifies frame.audio.
    stt_hears_filtered = bool(audio_filter) and args.use_filtered_audio
    stt_audio_label = (
        "FILTERED (Krisp-processed)" if stt_hears_filtered else "ORIGINAL (unfiltered)"
    )

    print(f"\n{'=' * 70}")
    print(f"Krisp filter configured : {'yes' if audio_filter else 'no'}")
    print(f"STT use_filtered_audio  : {args.use_filtered_audio}")
    print(f"==> STT will transcribe : {stt_audio_label} audio")
    print(f"{'=' * 70}")
    print(
        f"\nFeeding {len(audio_frames)} chunks ({duration_secs:.2f}s), "
        f"waiting up to {wait_secs:.1f}s for STT to finish...\n"
    )

    received_down, _ = await run_test(
        Pipeline([transport, stt]),
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            *audio_frames,
            # `BaseInputTransport` buffers InputAudioRawFrames through its own
            # internal async queue/task, while VADUserStoppedSpeakingFrame (a
            # plain SystemFrame) is forwarded immediately. Without a pause
            # here it can reach the STT service before any audio has drained
            # through, triggering transcription on an empty buffer.
            SleepFrame(1.0),
            VADUserStoppedSpeakingFrame(),
            SleepFrame(wait_secs),
        ],
        expected_down_frames=None,
        pipeline_params=PipelineParams(audio_in_sample_rate=sample_rate),
    )

    if args.verbose:
        audio_frame_count = sum(1 for f in received_down if isinstance(f, InputAudioRawFrame))
        print(f"All downstream frames received ({audio_frame_count} InputAudioRawFrame elided):")
        for f in received_down:
            if not isinstance(f, InputAudioRawFrame):
                print(f"  {f}")
        print()

    for err in received_down:
        if isinstance(err, ErrorFrame):
            print(f"STT error: {err.error}")

    text = " ".join(f.text for f in received_down if isinstance(f, TranscriptionFrame))
    print(f"Transcript (STT heard {stt_audio_label} audio):")
    print(text or "(empty)")


def main():
    parser = argparse.ArgumentParser(
        description="Run an audio file through a real STT service, with or without a Krisp VIVA filter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_stt_audiofile.py sample.wav --stt whisper-local
  python test_stt_audiofile.py sample.wav --stt whisper-local --filter
  python test_stt_audiofile.py sample.wav --stt whisper-local --filter --no-use-filtered-audio
  python test_stt_audiofile.py sample.wav --stt deepgram --filter --level 80

Run the same file twice with --filter set, once with --use-filtered-audio
(the default) and once with --no-use-filtered-audio, and diff the two
transcripts to see exactly what the filter changes for your STT.
        """,
    )
    parser.add_argument("input", help="Input audio file path")
    parser.add_argument(
        "--stt",
        default="whisper-local",
        choices=["whisper-local", "openai", "deepgram"],
        help="STT provider to test against (default: whisper-local, no API key needed)",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Configure a KrispVivaFilter as the transport's audio_in_filter",
    )
    parser.add_argument(
        "--filter-model",
        default=None,
        help="Path to the Krisp NC model (.kef). Falls back to KRISP_VIVA_FILTER_MODEL_PATH.",
    )
    parser.add_argument(
        "--level", type=int, default=100, help="Noise suppression level, 0-100 (default: 100)"
    )
    parser.add_argument(
        "--use-filtered-audio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether STT transcribes the filtered signal (default, matches "
        "STTService's own default) or the original audio (--no-use-filtered-audio). "
        "Only makes a difference together with --filter.",
    )
    parser.add_argument(
        "--wait-secs",
        type=float,
        default=None,
        help="Seconds to wait after the last audio chunk before collecting the "
        "transcript, giving segmented/network STT time to finish. Defaults to "
        "max(5, audio duration in seconds); raise this if your provider is slow.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose audio-load info and frame dump"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    asyncio.run(transcribe_file(args))


if __name__ == "__main__":
    main()
