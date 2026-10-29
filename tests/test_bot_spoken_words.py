#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import pytest

from pipecat.frames.frames import (
    BotSpokenWordsFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ProposedUserStartedSpeakingFrame,
    TTSTextFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.processors.bot_spoken_words_processor import BotSpokenWordsProcessor
from pipecat.tests.utils import run_test
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start import (
    BotSilentUserTurnStartStrategy,
    ExternalUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)


@pytest.mark.asyncio
async def test_processor_reports_each_text_frame_upstream():
    processor = BotSpokenWordsProcessor()

    down, up = await run_test(
        processor,
        frames_to_send=[
            TTSTextFrame("Hello", "word"),
            TTSTextFrame(" ", "word"),
            TTSTextFrame("there", "word"),
        ],
        expected_down_frames=[TTSTextFrame, TTSTextFrame, TTSTextFrame],
        expected_up_frames=[BotSpokenWordsFrame, BotSpokenWordsFrame],
    )

    assert [f.words[0][0] for f in up] == ["Hello", "there"]
    assert all(f.words[0][1] > 0 for f in up)


def _started_turns(strategy):
    started = []

    async def on_user_turn_started(_strategy, params):
        started.append(params)

    strategy.add_event_handler("on_user_turn_started", on_user_turn_started)
    return started


@pytest.mark.asyncio
async def test_gate_passes_frames_while_the_bot_is_silent():
    strategy = BotSilentUserTurnStartStrategy(VADUserTurnStartStrategy())
    started = _started_turns(strategy)

    result = await strategy.process_frame(VADUserStartedSpeakingFrame())

    assert result == ProcessFrameResult.STOP
    assert len(started) == 1 and started[0].enable_interruptions


@pytest.mark.asyncio
async def test_gate_blocks_frames_while_the_bot_speaks():
    strategy = BotSilentUserTurnStartStrategy(VADUserTurnStartStrategy())
    started = _started_turns(strategy)

    await strategy.process_frame(BotStartedSpeakingFrame())
    result = await strategy.process_frame(VADUserStartedSpeakingFrame())

    assert result == ProcessFrameResult.CONTINUE
    assert started == []

    await strategy.process_frame(BotStoppedSpeakingFrame())
    result = await strategy.process_frame(VADUserStartedSpeakingFrame())

    assert result == ProcessFrameResult.STOP
    assert len(started) == 1


@pytest.mark.asyncio
async def test_gate_reports_whether_the_wrapped_strategy_resolves_proposals():
    assert not BotSilentUserTurnStartStrategy(
        VADUserTurnStartStrategy()
    ).resolves_proposed_turn_start_frames
    assert BotSilentUserTurnStartStrategy(
        ExternalUserTurnStartStrategy()
    ).resolves_proposed_turn_start_frames


@pytest.mark.asyncio
async def test_external_strategy_still_resolves_proposals_while_the_bot_speaks():
    """The ungated external strategy is how a server interruption opens the turn."""
    strategy = ExternalUserTurnStartStrategy()
    started = _started_turns(strategy)

    await strategy.process_frame(BotStartedSpeakingFrame())
    result = await strategy.process_frame(ProposedUserStartedSpeakingFrame())

    assert result == ProcessFrameResult.STOP and len(started) == 1
