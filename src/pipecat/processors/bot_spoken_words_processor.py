#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Processor that reports the bot's spoken words back up the pipeline."""

from pipecat.frames.frames import BotSpokenWordsFrame, Frame, TTSTextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class BotSpokenWordsProcessor(FrameProcessor):
    """Report each bot word, as it is played, to processors earlier in the pipeline.

    Place it right after the output transport. The transport releases a
    :class:`~pipecat.frames.frames.TTSTextFrame` at the moment its audio plays,
    and this processor answers each one with a
    :class:`~pipecat.frames.frames.BotSpokenWordsFrame` pushed upstream, stamped
    with the pipeline clock time at which it arrived. With a TTS that reports
    word timestamps, that is one frame per word; with one that doesn't, it is
    one frame per sentence, timed at the start of the sentence.

    All frames pass through unchanged.

    Example::

        pipeline = Pipeline([
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            BotSpokenWordsProcessor(),
            assistant_aggregator,
        ])
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Pass the frame through, reporting spoken text upstream.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSTextFrame) and direction == FrameDirection.DOWNSTREAM:
            if frame.text.strip():
                spoken = BotSpokenWordsFrame(words=[(frame.text, self.get_clock().get_time())])
                await self.push_frame(spoken, FrameDirection.UPSTREAM)

        await self.push_frame(frame, direction)
