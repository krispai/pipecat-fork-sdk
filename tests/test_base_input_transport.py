#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for frame-based audio handling in :class:`BaseInputTransport`."""

import unittest
import warnings
from unittest.mock import AsyncMock

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.frames.frames import (
    FilterControlFrame,
    InputAudioRawFrame,
    InputTransportStartAudioStreamingFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_transport import TransportParams


class TestBaseInputTransportFrameAudio(unittest.IsolatedAsyncioTestCase):
    def _transport(self) -> BaseInputTransport:
        return BaseInputTransport(TransportParams(audio_in_enabled=True))

    async def test_incoming_audio_frame_routed_to_push_audio_frame(self):
        transport = self._transport()
        transport.push_audio_frame = AsyncMock()
        transport.push_frame = AsyncMock()
        frame = InputAudioRawFrame(audio=b"\x00\x00", sample_rate=16000, num_channels=1)
        await transport.process_frame(frame, FrameDirection.DOWNSTREAM)
        # Fed into the VAD path, not forwarded as a plain frame.
        transport.push_audio_frame.assert_called_once_with(frame)

    async def test_start_audio_streaming_frame_triggers_streaming(self):
        transport = self._transport()
        transport._start_audio_in_streaming = AsyncMock()
        await transport.process_frame(
            InputTransportStartAudioStreamingFrame(), FrameDirection.DOWNSTREAM
        )
        transport._start_audio_in_streaming.assert_called_once()

    async def test_start_audio_in_streaming_method_is_deprecated(self):
        transport = self._transport()
        transport._start_audio_in_streaming = AsyncMock()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            await transport.start_audio_in_streaming()
        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))
        transport._start_audio_in_streaming.assert_called_once()


class MarkingFilter(BaseAudioFilter):
    """Filter whose output is distinguishable from its input, for assertions."""

    async def start(self, sample_rate: int):
        pass

    async def stop(self):
        pass

    async def process_frame(self, frame: FilterControlFrame):
        pass

    async def filter(self, audio: bytes) -> bytes:
        return b"filtered:" + audio


class ReadyInputTransport(BaseInputTransport):
    """Input transport that reports itself ready as soon as it starts."""

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.set_transport_ready(frame)


class TestBaseInputTransportAudioFilter(unittest.IsolatedAsyncioTestCase):
    async def test_filter_output_attached_without_altering_raw_audio(self):
        """The filter's output lands on `filtered_audio`; `audio` stays raw."""
        transport = ReadyInputTransport(
            TransportParams(audio_in_enabled=True, audio_in_filter=MarkingFilter())
        )
        raw = b"\x01\x02\x03\x04"

        received_down, _ = await run_test(
            transport,
            frames_to_send=[
                InputAudioRawFrame(audio=raw, sample_rate=16000, num_channels=1),
                SleepFrame(0.1),
            ],
            expected_down_frames=[InputAudioRawFrame],
        )

        audio_frame = next(f for f in received_down if isinstance(f, InputAudioRawFrame))
        self.assertEqual(audio_frame.audio, raw)
        self.assertEqual(audio_frame.filtered_audio, b"filtered:" + raw)
        self.assertEqual(audio_frame.analysis_audio, b"filtered:" + raw)

    async def test_no_filter_leaves_filtered_audio_unset(self):
        """Without a configured filter, `filtered_audio` stays `None`."""
        transport = ReadyInputTransport(TransportParams(audio_in_enabled=True))
        raw = b"\x01\x02\x03\x04"

        received_down, _ = await run_test(
            transport,
            frames_to_send=[
                InputAudioRawFrame(audio=raw, sample_rate=16000, num_channels=1),
                SleepFrame(0.1),
            ],
            expected_down_frames=[InputAudioRawFrame],
        )

        audio_frame = next(f for f in received_down if isinstance(f, InputAudioRawFrame))
        self.assertEqual(audio_frame.audio, raw)
        self.assertIsNone(audio_frame.filtered_audio)
        self.assertEqual(audio_frame.analysis_audio, raw)


if __name__ == "__main__":
    unittest.main()
