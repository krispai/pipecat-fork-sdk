#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User turn start strategy that only applies while the bot is silent."""

from loguru import logger

from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, Frame
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start.base_user_turn_start_strategy import BaseUserTurnStartStrategy


class BotSilentUserTurnStartStrategy(BaseUserTurnStartStrategy):
    """Run another start strategy only while the bot is not speaking.

    Frames reach the wrapped strategy between a
    :class:`~pipecat.frames.frames.BotStoppedSpeakingFrame` and the next
    :class:`~pipecat.frames.frames.BotStartedSpeakingFrame`; while the bot
    speaks they are dropped, so the wrapped strategy can neither start a turn
    nor interrupt the bot. Use it when something else decides whether the user
    really interrupted — an STT service with server-side interruption
    detection, for example — and local VAD or transcription should only open
    turns in the bot's silence.

    Example::

        start=[
            ExternalUserTurnStartStrategy(),
            BotSilentUserTurnStartStrategy(VADUserTurnStartStrategy()),
        ]
    """

    def __init__(self, strategy: BaseUserTurnStartStrategy, **kwargs):
        """Initialize the strategy.

        Args:
            strategy: The start strategy to run while the bot is silent.
            **kwargs: Additional arguments passed to :class:`BaseUserTurnStartStrategy`.
        """
        super().__init__(**kwargs)
        self._strategy = strategy
        self._bot_speaking = False
        for event in ("on_push_frame", "on_broadcast_frame", "on_user_turn_started"):
            self._strategy.add_event_handler(event, self._forward(event))
        self._strategy.add_event_handler(
            "on_reset_aggregation", self._forward("on_reset_aggregation")
        )

    def _forward(self, event: str):
        async def handler(_strategy, *args, **kwargs):
            await self._call_event_handler(event, *args, **kwargs)

        return handler

    @property
    def resolves_proposed_turn_start_frames(self) -> bool:
        """Whether the wrapped strategy resolves proposed turn starts."""
        return self._strategy.resolves_proposed_turn_start_frames

    async def setup(self, setup: FrameProcessorSetup):
        """Set up this strategy and the wrapped one.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._strategy.setup(setup)

    async def cleanup(self):
        """Clean up this strategy and the wrapped one."""
        await self._strategy.cleanup()
        await super().cleanup()

    async def handle_user_turn_started(self):
        """Forward the turn start to the wrapped strategy."""
        await self._strategy.handle_user_turn_started()

    async def handle_user_turn_stopped(self):
        """Forward the turn end to the wrapped strategy."""
        await self._strategy.handle_user_turn_stopped()

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        """Track the bot's speech and forward frames while it is silent.

        Args:
            frame: The frame to be analyzed.

        Returns:
            The wrapped strategy's result, or CONTINUE while the bot speaks.
        """
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False

        if self._bot_speaking and not isinstance(frame, BotStartedSpeakingFrame):
            logger.trace(f"{self}: bot is speaking, {frame} not passed to {self._strategy}")
            return ProcessFrameResult.CONTINUE

        result = await self._strategy.process_frame(frame)
        return ProcessFrameResult.CONTINUE if result is None else result
