#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Interruptible bot running on Krisp's cloud APIs end to end.

This example demonstrates a conversational bot with:
- Krisp STT, whose end-of-turn prediction ends user turns, whose interruption
  prediction decides when the user talks over the bot (fed the bot's words by
  ``BotSpokenWordsProcessor``), and whose voice isolation keeps background
  voices out of transcripts
- Silero VAD starting user turns while the bot is silent (optional: with
  ``USE_SILERO_VAD = False`` Krisp's first partial transcript starts them)
- Krisp TTS
- Krisp VIVA API (``KrispVivaApiFilter``) removing noise and background voices
  from incoming audio in primary mode

Voice isolation has two modes. ``primary`` (default) hears only the main
speaker: the VIVA filter cleans the audio and STT keeps only primary-speaker
words. ``all`` hears everyone: the filter is bypassed and STT labels every
voice; background words go to the log, not to the conversation. When more than
one mode is available the bot asks which one to use when a client connects and
switches with the ``set_voice_isolation`` tool; it can be changed later the same
way, by typing ``/voice <mode>`` in the client, or set up front with
``KRISP_VOICE_ISOLATION``. Only ``all`` is available while the stage all-voices
model rejects ``primary_only``.

Required environment variables:
- KRISP_API_KEY: Krisp API key (session tokens are created from it)
- OPENAI_API_KEY: OpenAI API key for LLM
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.filters.krisp_viva_api_filter import KrispVivaApiFilter
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import (
    FilterEnableFrame,
    Frame,
    LLMMessagesAppendFrame,
    LLMRunFrame,
    StartFrame,
    STTUpdateSettingsFrame,
    TTSSpeakFrame,
)
from pipecat.metrics.metrics import TurnMetricsData
from pipecat.observers.loggers.metrics_log_observer import MetricsLogObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.bot_spoken_words_processor import BotSpokenWordsProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.krisp.stt import KrispSTTService
from pipecat.services.krisp.tts import KrispTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.livekit.transport import LiveKitParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

# LOCAL ONLY (do not commit): run against Krisp staging.
KRISP_STAGE_REST = "https://stage-api.developers.krisp.ai"
KRISP_STAGE_WS = "wss://stage.streaming.krisp.ai"
KRISP_STT_MODEL = "krisp-stt-v1-preview/prometheus-3-2-agents-all-voices-pre"

# Krisp STT decides when the user interrupts the bot. The bot's words are fed
# back to it by the BotSpokenWordsProcessor at the end of the pipeline.
INTERRUPTION_PREDICTION = True
# Silero VAD opens a user turn ~200 ms after speech starts while the bot is
# silent. Without it, Krisp's first partial transcript opens the turn (about
# a second later) and there are no local false starts on noise. End of turn
# and interruptions come from Krisp either way.
USE_SILERO_VAD = os.getenv("USE_SILERO_VAD", "0") not in ("0", "false", "False")
# Whose voice the bot hears:
# - "primary": only the main speaker. The VIVA filter removes other voices and
#   noise from the audio, and STT drops any background words it still labels.
# - "all": everyone is heard. The VIVA filter is bypassed and STT labels each
#   word primary or background; background words are logged by the STT
#   service but still kept out of the conversation.
# The stage all-voices model labels every word; it does not accept the
# "primary_only" output yet, so the "primary" mode is disabled until it does.
VOICE_ISOLATION_MODE = os.getenv("KRISP_VOICE_ISOLATION", "all")

VOICE_ISOLATION_MODES: dict[str, dict] = {
    # "primary": {
    #     "voice_isolation": True,
    #     "voice_isolation_output": "primary_only",
    #     "keep_background_words": False,
    # },
    "all": {
        "voice_isolation": True,
        "voice_isolation_output": "all_voices",
        "keep_background_words": False,
    },
}
if VOICE_ISOLATION_MODE not in VOICE_ISOLATION_MODES:
    raise SystemExit(
        f"KRISP_VOICE_ISOLATION={VOICE_ISOLATION_MODE!r} is not available; "
        f"choose one of: {', '.join(VOICE_ISOLATION_MODES)}"
    )

# The filter and command processor of the current session, so the mode can
# be switched while it runs.
_audio_filter: KrispVivaApiFilter | None = None
_commands: "VoiceIsolationCommands | None" = None

# The LLM switches modes with this tool, so the user can answer the bot's
# question (or ask for a change later) by voice.
set_voice_isolation_function = FunctionSchema(
    name="set_voice_isolation",
    description=(
        "Set whose voice the assistant hears: 'primary' keeps only the main speaker, "
        "'all' hears everyone in the room. Only the modes listed in the enum are available."
    ),
    properties={
        "mode": {
            "type": "string",
            "enum": list(VOICE_ISOLATION_MODES),
            "description": "The voice isolation mode the user asked for.",
        }
    },
    required=["mode"],
)
voice_tools = ToolsSchema(standard_tools=[set_voice_isolation_function])


async def set_voice_isolation(params: FunctionCallParams):
    mode = str(params.arguments.get("mode", "")).lower()
    if mode not in VOICE_ISOLATION_MODES:
        await params.result_callback({"error": f"unknown mode {mode!r}"})
        return
    if _commands:
        # The bot's reply to the tool result is the confirmation.
        await _commands.set_mode(mode, announce=False)
    await params.result_callback({"mode": mode})


def _audio_in_filter() -> KrispVivaApiFilter:
    global _audio_filter
    _audio_filter = KrispVivaApiFilter(
        url=f"{KRISP_STAGE_WS}/voice-isolation", rest_base_url=KRISP_STAGE_REST
    )
    return _audio_filter


class VoiceIsolationCommands(FrameProcessor):
    """Apply the voice isolation mode, and switch it when the user types ``/voice <mode>``.

    Sits right after the input transport. A mode change updates the STT
    settings (which reconnects with a new handshake) and enables the VIVA
    filter for primary or bypasses it for all; the command itself never
    reaches the LLM.
    """

    def __init__(self, stt: KrispSTTService, mode: str, **kwargs):
        super().__init__(**kwargs)
        global _commands
        self._stt = stt
        self._mode = mode
        _commands = self

    @property
    def mode(self) -> str:
        return self._mode

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            await self.set_mode(self._mode, announce=False)
            return

        if isinstance(frame, LLMMessagesAppendFrame) and len(frame.messages) == 1:
            message = frame.messages[0]
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, str) and content.strip().lower().startswith("/voice"):
                words = content.split()
                mode = words[1].lower() if len(words) > 1 else ""
                if mode in VOICE_ISOLATION_MODES:
                    await self.set_mode(mode)
                else:
                    choices = ", or ".join(f"slash voice {m}" for m in VOICE_ISOLATION_MODES)
                    await self.push_frame(TTSSpeakFrame(f"Say {choices}."))
                return

        await self.push_frame(frame, direction)

    async def set_mode(self, mode: str, announce: bool = True):
        """Switch voice isolation; the bot confirms aloud unless ``announce`` is False."""
        self._mode = mode
        logger.info(f"Voice isolation mode: {mode}")
        await self.push_frame(
            STTUpdateSettingsFrame(
                delta=KrispSTTService.Settings(**VOICE_ISOLATION_MODES[mode]), service=self._stt
            )
        )
        if _audio_filter:
            await _audio_filter.process_frame(FilterEnableFrame(enable=(mode == "primary")))
        if announce:
            await self.push_frame(TTSSpeakFrame(f"Voice isolation set to {mode}."))


transport_params = {
    "eval": lambda: EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_filter=_audio_in_filter(),
    ),
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_filter=_audio_in_filter(),
    ),
    "livekit": lambda: LiveKitParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_filter=_audio_in_filter(),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_filter=_audio_in_filter(),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_filter=_audio_in_filter(),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

    stt = KrispSTTService(
        api_key=os.environ["KRISP_API_KEY"],
        url=f"{KRISP_STAGE_WS}/stt",
        settings=KrispSTTService.Settings(
            model=KRISP_STT_MODEL,
            interruption_prediction=INTERRUPTION_PREDICTION,
            **VOICE_ISOLATION_MODES[VOICE_ISOLATION_MODE],
        ),
        rest_base_url=KRISP_STAGE_REST,
    )

    tts = KrispTTSService(
        api_key=os.environ["KRISP_API_KEY"],
        url=f"{KRISP_STAGE_WS}/tts",
        rest_base_url=KRISP_STAGE_REST,
        settings=KrispTTSService.Settings(voice="Amy", language="en-US"),
    )

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            system_instruction=(
                "You are a helpful assistant in a voice conversation. Your responses will be "
                "spoken aloud, so avoid emojis, bullet points, or other formatting that can't "
                "be spoken. Respond to what the user said in a creative, helpful, and brief way. "
                "Voice isolation decides whose voice you hear: 'primary' keeps only the main "
                "speaker, 'all' hears everyone in the room. The modes available right now "
                f"are: {', '.join(VOICE_ISOLATION_MODES)} (current: {VOICE_ISOLATION_MODE}). "
                "Whenever the user picks or changes a mode, call set_voice_isolation and then "
                "confirm in one short sentence; if they ask for a mode that is not available, "
                "say so."
            ),
        ),
    )
    llm.register_function("set_voice_isolation", set_voice_isolation)

    context = LLMContext(tools=voice_tools)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer() if USE_SILERO_VAD else None,
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            VoiceIsolationCommands(stt, VOICE_ISOLATION_MODE),  # "/voice <mode>" from the client
            stt,  # STT
            user_aggregator,  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            BotSpokenWordsProcessor(),  # Bot words back to STT for interruption detection
            assistant_aggregator,  # Assistant spoken responses
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        observers=[MetricsLogObserver(include_metrics={TurnMetricsData})],
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Kick off the conversation.
        if len(VOICE_ISOLATION_MODES) > 1:
            opening = (
                "Introduce yourself in one sentence, then ask whether you should listen "
                "to the primary speaker only or to all voices in the room "
                f"(currently: {VOICE_ISOLATION_MODE})."
            )
        else:
            opening = (
                "Introduce yourself in one sentence and mention that you are listening "
                f"to {'all voices in the room' if VOICE_ISOLATION_MODE == 'all' else 'the primary speaker only'}."
            )
        context.add_message({"role": "developer", "content": opening})
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        global _commands, _audio_filter
        logger.info("Client disconnected")
        _commands = None
        _audio_filter = None
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
