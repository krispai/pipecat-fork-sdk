#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for STTService's `use_filtered_audio` option."""

import unittest
from collections.abc import AsyncGenerator

from pipecat.frames.frames import Frame, InputAudioRawFrame
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService, STTService

SAMPLE_RATE = 16000


class FakeSTTService(STTService):
    """Continuous STT service that records the audio it was asked to transcribe."""

    def __init__(self, **kwargs):
        kwargs.setdefault("settings", STTSettings(model=None, language=None))
        super().__init__(**kwargs)
        self.received_audio: list[bytes] = []

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        self.received_audio.append(audio)
        yield None


class FakeSegmentedSTTService(SegmentedSTTService):
    """Segmented STT service that records the audio it was asked to transcribe."""

    def __init__(self, **kwargs):
        kwargs.setdefault("settings", STTSettings(model=None, language=None))
        super().__init__(**kwargs)
        self.received_audio: list[bytes] = []

    @property
    def wants_wav_segments(self) -> bool:
        return False

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        self.received_audio.append(audio)
        yield None


def _frame(raw: bytes, filtered_audio: bytes | None = None) -> InputAudioRawFrame:
    frame = InputAudioRawFrame(audio=raw, sample_rate=SAMPLE_RATE, num_channels=1)
    frame.filtered_audio = filtered_audio
    return frame


def _make_service(cls, **kwargs):
    service = cls(sample_rate=SAMPLE_RATE, **kwargs)
    service._sample_rate = SAMPLE_RATE
    return service


class TestSTTUseFilteredAudio(unittest.IsolatedAsyncioTestCase):
    async def test_defaults_to_filtered_audio_when_present(self):
        service = _make_service(FakeSTTService)
        raw, filtered = b"\x01" * 320, b"\x02" * 320

        await service.process_audio_frame(_frame(raw, filtered), direction=None)

        assert service.received_audio == [filtered]

    async def test_uses_raw_audio_when_disabled(self):
        service = _make_service(FakeSTTService, use_filtered_audio=False)
        raw, filtered = b"\x01" * 320, b"\x02" * 320

        await service.process_audio_frame(_frame(raw, filtered), direction=None)

        assert service.received_audio == [raw]

    async def test_falls_back_to_raw_when_no_filter_configured(self):
        service = _make_service(FakeSTTService)
        raw = b"\x01" * 320

        await service.process_audio_frame(_frame(raw, filtered_audio=None), direction=None)

        assert service.received_audio == [raw]

    async def test_skips_transcription_while_filter_buffering(self):
        service = _make_service(FakeSTTService)
        raw = b"\x01" * 320

        await service.process_audio_frame(_frame(raw, filtered_audio=b""), direction=None)

        assert service.received_audio == []

    async def test_segmented_stt_buffers_filtered_audio_by_default(self):
        service = _make_service(FakeSegmentedSTTService)
        service._user_speaking = True
        raw, filtered = b"\x01" * 320, b"\x02" * 320

        await service.process_audio_frame(_frame(raw, filtered), direction=None)

        assert bytes(service._audio_buffer) == filtered


if __name__ == "__main__":
    unittest.main()
