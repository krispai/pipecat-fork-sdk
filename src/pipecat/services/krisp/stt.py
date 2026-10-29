#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Krisp real-time speech-to-text service implementation.

Streams 16 kHz PCM audio to Krisp's STT WebSocket API. Krisp predicts where
each user turn ends; finals for a turn are accumulated and pushed as one
``TranscriptionFrame`` when the server sends its ``turn`` message.

With ``interruption_prediction`` the server also decides when the user has
interrupted the bot. For that it needs the bot's words as they are heard: the
service forwards :class:`~pipecat.frames.frames.BotSpokenWordsFrame` (pushed
upstream by a
:class:`~pipecat.processors.bot_spoken_words_processor.BotSpokenWordsProcessor`
after the output transport) as ``agent_words``, placed on the session's audio
clock, and turns the server's ``interruption`` event into a proposed user turn
start.

With ``voice_isolation`` the server labels every word as the primary speaker's
or a background voice's; background words are left out of transcripts.
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from websockets.protocol import State

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    BotSpokenWordsFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.krisp.common import (
    KRISP_REST_BASE_URL,
    KrispServiceError,
    connect_session,
)
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.turns.user_start import (
    BaseUserTurnStartStrategy,
    BotSilentUserTurnStartStrategy,
    ExternalUserTurnStartStrategy,
)
from pipecat.turns.user_stop import ExternalUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import (
    UserTurnStrategies,
    default_user_turn_start_strategies,
)
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt
from pipecat.utils.types import NOT_GIVEN, NotGiven

KRISP_STT_URL = "wss://stage.streaming.krisp.ai/stt"

# The API accepts 16 kHz mono PCM only.
KRISP_STT_SAMPLE_RATE = 16000
_BYTES_PER_MS = KRISP_STT_SAMPLE_RATE * 2 // 1000

# An interruption the server reported that is followed by no speech at all
# (no VAD activity, no transcript) within this long was a false alarm; the
# turn it opened is closed so the bot can resume.
_FALSE_INTERRUPTION_SECS = 2.5


@dataclass
class KrispSTTSettings(STTSettings):
    """Settings for KrispSTTService.

    ``language`` is sent as a language hint: it restricts the scripts the
    transcript may use rather than selecting a language.

    Parameters:
        vocabulary: Custom vocabulary terms.
        turn_prediction: Whether the server predicts end of turn. When True,
            the server's turn events end pipecat's user turns; when False, the
            service asks the server to finalize whenever local VAD detects the
            user stopped speaking. Turn start and bot interruption are always
            left to pipecat's start strategies.
        turn_sensitivity: End-of-turn sensitivity, 0 (more sensitive) to 1.
        turn_max_silence_ms: Silence in ms after which the turn ends (400-4000).
        turn_silence_validation_ms: Silence in ms used to validate the
            turn model's decision (10-1000).
        interruption_prediction: Whether the server decides when the user has
            interrupted the bot. Requires ``turn_prediction`` and the bot's
            words, see :class:`KrispSTTService`. When True, the service
            recommends start strategies that leave the decision to the server
            while the bot speaks.
        interruption_sensitivity: Model confidence needed to report an
            interruption, 0.5 to 1; 1 disables the model and leaves only
            ``interruption_max_overlap_ms``.
        interruption_max_overlap_ms: Overlapping speech in ms after which an
            interruption is reported even if the model hasn't (300-5000, or 0
            for never).
        voice_isolation: Whether the server labels each word as the primary
            speaker's (``primary``) or a background voice's (``background``).
        voice_isolation_output: ``"all_voices"`` or ``"primary_only"``.
        keep_background_words: With ``voice_isolation``, whether words labeled
            ``background`` stay in the transcript text. They are left out by
            default.
        backchannel_max_words: With ``interruption_prediction``, speech over
            the bot that the server didn't call an interruption is held when
            it has at most this many words (a backchannel like "uh-huh" or a
            short "hello?") and delivered as the user's turn once the bot
            finishes; anything longer interrupts the bot when its turn ends.
    """

    vocabulary: list[str] | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    turn_prediction: bool | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    turn_sensitivity: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    turn_max_silence_ms: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    turn_silence_validation_ms: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    interruption_prediction: bool | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    interruption_sensitivity: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    interruption_max_overlap_ms: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    voice_isolation: bool | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    voice_isolation_output: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    keep_background_words: bool | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    backchannel_max_words: int | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class KrispSTTService(WebsocketSTTService):
    """Krisp real-time speech-to-text service.

    Pushes ``InterimTranscriptionFrame`` with the turn's finals followed by the
    latest partial hypothesis, and one ``TranscriptionFrame`` per user turn.
    With ``turn_prediction`` enabled (the default) the service recommends user
    turn strategies that keep pipecat's default start strategies (which also
    interrupt the bot) and end turns on Krisp's end-of-turn events.
    Reconnection and silence keepalive (enabled with ``keepalive_timeout``)
    come from :class:`WebsocketSTTService`.

    With ``interruption_prediction`` the server watches for the user talking
    over the bot and the service proposes a user turn start when it reports
    one. The server needs to know when the bot speaks and what it says, so the
    pipeline must end with a
    :class:`~pipecat.processors.bot_spoken_words_processor.BotSpokenWordsProcessor`
    after the output transport; its frames are forwarded as ``agent_words``.
    The recommended start strategies then let local VAD and transcription open
    turns only while the bot is silent.

    Audio at other sample rates is resampled to 16 kHz.
    """

    Settings = KrispSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        url: str = KRISP_STT_URL,
        rest_base_url: str = KRISP_REST_BASE_URL,
        settings: Settings | None = None,
        agent_words_offset_ms: int = 0,
        **kwargs,
    ):
        """Initialize the Krisp STT service.

        Args:
            api_key: Krisp master API key. It is only used to create a session
                token before each connection and never sent on the WebSocket.
            url: Krisp STT WebSocket endpoint.
            rest_base_url: Krisp REST API base URL used to create session tokens.
            settings: Runtime-updatable settings. Changing any of them
                reconnects, since they are fixed by the session handshake.
            agent_words_offset_ms: Added to every ``agent_words`` timestamp.
                The bot's words are placed on the audio clock from the time
                they were played locally; this makes up for the delay between
                the microphone and the server (audio filters, network) that
                the pipeline can't measure.
            **kwargs: Additional arguments passed to :class:`WebsocketSTTService`
                (e.g. ``keepalive_timeout``, ``ttfs_p99_latency``).
        """
        default_settings = self.Settings(
            model="krisp-realtime-stt-agents-v1",
            language=None,
            vocabulary=None,
            turn_prediction=True,
            turn_sensitivity=0.9,
            turn_max_silence_ms=2000,
            turn_silence_validation_ms=100,
            interruption_prediction=False,
            interruption_sensitivity=None,
            interruption_max_overlap_ms=None,
            voice_isolation=False,
            voice_isolation_output=None,
            keep_background_words=False,
            backchannel_max_words=2,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(settings=default_settings, **kwargs)

        self._api_key = api_key
        self._url = url
        self._rest_base_url = rest_base_url
        self._agent_words_offset_ms = agent_words_offset_ms

        self._receive_task = None
        self._input_resampler = create_stream_resampler()

        # Transcript state for the current user turn.
        self._final_text = ""
        self._partial_text = ""
        self._detected_language: Language | None = None
        self._in_turn = False

        # Where the session's audio clock stands: bytes sent on this connection
        # and the pipeline clock time of the last send. A bot word played at
        # pipeline time T is at ``bytes / 32 + (T - last send)`` ms.
        self._audio_bytes_sent = 0
        self._last_send_time_ns: int | None = None
        # Pipeline state the interruption handling depends on: whether the
        # aggregator has a user turn open, whether the bot is speaking, and
        # whether the current turn was opened by a server interruption.
        self._user_turn_open = False
        self._bot_speaking = False
        self._interrupted = False
        self._agent_words_sent = False
        self._voice_labels_checked = False
        self._vad_user_speaking = False
        self._interruption_check_task: asyncio.Task | None = None
        # Short speech over the bot, kept until the bot finishes speaking.
        self._held_text = ""
        self._held_language: Language | None = None

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.

        Returns:
            True, as Krisp STT supports metrics generation.
        """
        return True

    def service_metadata_frame(self) -> STTMetadataFrame:
        """Recommend ending user turns on Krisp's end-of-turn events.

        Turn start (and with it bot interruption) stays with pipecat's default
        start strategies. Applied unless the application passes its own
        ``user_turn_strategies``.
        """
        frame = super().service_metadata_frame()
        if self._settings.turn_prediction:
            start: list[BaseUserTurnStartStrategy] = default_user_turn_start_strategies()
            if self._settings.interruption_prediction:
                # Krisp decides interruptions: local detectors only open turns
                # while the bot is silent.
                wrapped: list[BaseUserTurnStartStrategy] = [
                    BotSilentUserTurnStartStrategy(s) for s in start
                ]
                start = [ExternalUserTurnStartStrategy(), *wrapped]
            frame.user_turn_strategies = UserTurnStrategies(
                start=start,
                stop=[ExternalUserTurnStopStrategy()],
            )
        return frame

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply a settings delta and reconnect with a new handshake.

        Args:
            delta: A :class:`STTSettings` (or ``KrispSTTService.Settings``) delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)
        if changed:
            await self._request_reconnect()
        return changed

    async def stop(self, frame: EndFrame):
        """Push the turn's finals so far, then end the session.

        Transcripts for audio the server hasn't finalized yet aren't waited for.

        Args:
            frame: The end frame.
        """
        await self._push_pending_finals()
        await super().stop(frame)

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._connect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, finalizing on VAD stop when turn prediction is off.

        Also follows the user turn and the bot's speech, and forwards the
        bot's words to the server when it predicts interruptions.

        Args:
            frame: The frame to process.
            direction: Direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self._vad_user_speaking = False
            if not self._settings.turn_prediction:
                await self._send_json({"type": "finalize"})
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._user_turn_open = True
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_turn_open = False
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._vad_user_speaking = True
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._agent_words_sent = False
            await self._deliver_held_speech()
        elif isinstance(frame, BotSpokenWordsFrame) and self._settings.interruption_prediction:
            await self._send_agent_words(frame)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Send audio to Krisp for transcription.

        Args:
            audio: Raw 16-bit mono PCM audio at the pipeline sample rate.

        Yields:
            None, since transcriptions arrive on the receive task.
        """
        if self.sample_rate != KRISP_STT_SAMPLE_RATE:
            audio = await self._input_resampler.resample(
                audio, self.sample_rate, KRISP_STT_SAMPLE_RATE
            )
        if audio and self._websocket and self._websocket.state is State.OPEN:
            try:
                await self._websocket.send(audio)
                self._audio_bytes_sent += len(audio)
                self._last_send_time_ns = self.get_clock().get_time()
            except Exception as e:
                logger.warning(f"{self}: send failed: {e}")
        yield None

    async def _send_agent_words(self, frame: BotSpokenWordsFrame):
        """Forward the bot's words, placed on the session's audio clock.

        Words reported before any audio was sent on this connection can't be
        placed and are dropped; the server only needs them while it has audio
        to compare against.
        """
        if self._last_send_time_ns is None:
            return
        stream_ms = self._audio_bytes_sent // _BYTES_PER_MS
        words = []
        for text, played_at_ns in frame.words:
            elapsed_ms = (played_at_ns - self._last_send_time_ns) // 1_000_000
            start = max(0, stream_ms + elapsed_ms + self._agent_words_offset_ms)
            words.extend({"start": start, "text": w} for w in text.split())
        if words:
            if not self._agent_words_sent:
                # Once per bot utterance: enough to see in the log that the
                # bot's words reach the server, without a line per word.
                logger.debug(f"{self}: sending agent_words, first: {words[0]}")
            self._agent_words_sent = True
            logger.trace(f"{self}: agent_words {words}")
            await self._send_json({"type": "agent_words", "words": words})

    async def _send_json(self, msg: dict):
        if not self._websocket or self._websocket.state is not State.OPEN:
            return
        try:
            await self._websocket.send(json.dumps(msg))
        except Exception as e:
            logger.warning(f"{self}: failed to send {msg.get('type')}: {e}")

    def _build_handshake(self) -> dict[str, Any]:
        turn: dict[str, Any] = {"prediction": self._settings.turn_prediction}
        optional_turn_fields = {
            "sensitivity": self._settings.turn_sensitivity,
            "max_silence": self._settings.turn_max_silence_ms,
            "silence_validation": self._settings.turn_silence_validation_ms,
        }
        turn.update({k: v for k, v in optional_turn_fields.items() if v is not None})

        # Sent even when off: the server enables interruption prediction by
        # default, and its events are only useful with the bot's words.
        interruption: dict[str, Any] = {"prediction": bool(self._settings.interruption_prediction)}
        if self._settings.interruption_prediction:
            optional_interruption_fields = {
                "sensitivity": self._settings.interruption_sensitivity,
                "max_overlap": self._settings.interruption_max_overlap_ms,
            }
            interruption.update(
                {k: v for k, v in optional_interruption_fields.items() if v is not None}
            )

        handshake: dict[str, Any] = {
            "model": self._settings.model,
            "sample_rate": KRISP_STT_SAMPLE_RATE,
            "channels": 1,
            "audio_format": "pcm_s16le",
            "turn": turn,
            "interruption": interruption,
        }
        if self._settings.voice_isolation:
            handshake["voice_isolation"] = True
            if self._settings.voice_isolation_output:
                handshake["voice_isolation_output"] = self._settings.voice_isolation_output
        if self._settings.vocabulary:
            handshake["vocabulary"] = self._settings.vocabulary
        if self._settings.language:
            handshake["language_hints"] = [str(self._settings.language)]
        return handshake

    async def _connect(self):
        await super()._connect()

        await self._connect_websocket()

        # A receive loop that has ended (e.g. after a permanent error closed the
        # socket) can't serve the new connection.
        if self._receive_task and self._receive_task.done():
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._cancel_interruption_check()
        await self._disconnect_websocket()
        self._reset_stream_state()

    async def _reconnect_websocket(self, attempt_number: int) -> bool:
        """Push the dropped connection's finals, then reconnect.

        The server's transcript state goes with the connection, so the turn's
        finals received so far are pushed as a transcription. An open turn
        stays open for the new connection's turn event to end.

        Args:
            attempt_number: Current retry attempt number for logging.

        Returns:
            True if reconnection and verification succeeded.
        """
        await self._push_pending_finals()
        return await super()._reconnect_websocket(attempt_number)

    async def _do_reconnect(self):
        """Push the turn's finals so far, then reconnect with the new settings.

        An open turn stays open across the reconnect, so the new connection's
        turn event still ends it.
        """
        await self._push_pending_finals()
        in_turn = self._in_turn
        await super()._do_reconnect()
        self._in_turn = in_turn

    async def _push_pending_finals(self):
        """Push the finals received so far, keeping an open turn open.

        The server's transcript state goes with the connection, so finals
        that haven't been pushed yet would otherwise be lost.
        """
        in_turn = self._in_turn
        await self._push_turn_transcription({})
        self._in_turn = in_turn

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug(f"Connecting to Krisp STT ({self._url}, tokens: {self._rest_base_url})")
            websocket, ready = await connect_session(
                self._websocket_connect,
                self._url,
                api_key=self._api_key,
                usage_type="stt",
                rest_base_url=self._rest_base_url,
                handshake=self._build_handshake(),
            )
            # Exposed only once ready: no audio may be sent before that. The
            # audio clock starts over with the session.
            self._audio_bytes_sent = 0
            self._last_send_time_ns = None
            self._voice_labels_checked = False
            self._websocket = websocket
            self._check_ready(ready)
            logger.info(
                f"Connected to Krisp STT (request_id={ready.get('request_id')}, "
                f"model={self._settings.model}, "
                f"interruption_prediction={bool(self._settings.interruption_prediction)}, "
                f"voice_isolation={bool(self._settings.voice_isolation)})"
            )
            await self._call_event_handler("on_connected")
        except Exception as e:
            self._websocket = None
            await self.push_error(error_msg=f"Unable to connect to Krisp STT: {e}", exception=e)

    async def _disconnect_websocket(self):
        ws = self._websocket
        try:
            if ws and ws.state is State.OPEN:
                logger.debug("Disconnecting from Krisp STT")
                await ws.send(json.dumps({"type": "end_of_audio"}))
                await ws.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing websocket: {e}", exception=e)
        finally:
            if self._websocket is ws:
                self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _check_ready(self, ready: dict):
        """Report whether the server confirmed the requested features.

        A server that runs the session without a feature says so by echoing
        it disabled in ``ready``; one that echoes nothing (the gateway only
        returns the request id) gives no confirmation either way, so only an
        explicit refusal is a warning.
        """
        if self._settings.interruption_prediction:
            interruption = ready.get("interruption")
            if interruption is None:
                logger.debug(f"{self}: server didn't confirm interruption prediction")
            elif not interruption.get("prediction"):
                logger.warning(
                    f"{self}: interruption prediction was requested but the server "
                    f"didn't enable it (ready: {interruption})"
                )
        if self._settings.voice_isolation:
            voice_isolation = ready.get("voice_isolation")
            if voice_isolation is None:
                logger.debug(f"{self}: server didn't confirm voice isolation")
            elif not voice_isolation:
                logger.warning(
                    f"{self}: voice isolation was requested but the server didn't enable it"
                )

    def _reset_stream_state(self):
        self._reset_turn_text()
        self._held_text = ""
        self._held_language = None
        self._audio_bytes_sent = 0
        self._last_send_time_ns = None

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _receive_messages(self):
        async for message in self._get_websocket():
            if isinstance(message, bytes):
                continue
            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                logger.warning(f"{self}: received non-JSON message: {message}")
                continue

            msg_type = msg.get("type")
            if msg_type == "transcript":
                await self._on_transcript(msg.get("transcript") or {})
            elif msg_type == "turn":
                await self._on_turn(msg)
            elif msg_type == "interruption":
                await self._on_interruption(msg)
            elif msg_type == "error":
                error = KrispServiceError.from_message(msg)
                await self.push_error(error_msg=f"Krisp STT error: {error}", exception=error)
                if error.is_permanent:
                    # The error marks the service unusable, so the receive loop
                    # won't reconnect; close the session client-side.
                    await self._get_websocket().close()
                    return

    async def _on_interruption(self, msg: dict):
        """Propose a user turn start on the server's interruption decision."""
        if not self._settings.interruption_prediction:
            logger.debug(f"{self}: ignoring interruption event, prediction is off")
            return
        self._in_turn = True
        self._interrupted = True
        logger.info(
            f"{self}: INTERRUPTION interrupted=True (reason={msg.get('reason')}, "
            f"confidence={msg.get('confidence')}, detected_at={msg.get('detected_at')})"
        )
        await self.broadcast_frame(ProposedUserStartedSpeakingFrame)
        await self._cancel_interruption_check()
        self._interruption_check_task = self.create_task(self._end_silent_interruption())

    async def _end_silent_interruption(self):
        await asyncio.sleep(_FALSE_INTERRUPTION_SECS)
        if self._silent_interruption():
            logger.debug(f"{self}: no speech followed the interruption, closing the turn")
            self._reset_turn_text()
            await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)

    def _silent_interruption(self) -> bool:
        return (
            self._interrupted
            and not self._vad_user_speaking
            and not (self._final_text or self._partial_text).strip()
        )

    async def _cancel_interruption_check(self):
        if self._interruption_check_task:
            await self.cancel_task(self._interruption_check_task)
            self._interruption_check_task = None

    async def _on_transcript(self, transcript: dict):
        words = transcript.get("words") or []
        if words and self._settings.voice_isolation and not self._voice_labels_checked:
            # The gateway doesn't echo the handshake, so the first words tell
            # whether the server is labelling voices.
            self._voice_labels_checked = True
            if "voice" in words[0]:
                logger.debug(f"{self}: server is labelling voices (voice isolation active)")
            else:
                logger.warning(
                    f"{self}: transcript words carry no voice labels; voice isolation is off"
                )
        if words and self._settings.voice_isolation:
            # Which speaker the server attributed each word to; words without a
            # label show as "none".
            labelled = " ".join(
                f"{w.get('text', '').strip()}({w.get('voice') or 'none'})" for w in words
            )
            kind = "final" if transcript.get("isFinal") else "partial"
            logger.debug(f"{self}: voices {kind} [{labelled}]")
        if self._settings.voice_isolation and not self._settings.keep_background_words:
            background = [w for w in words if w.get("voice") == "background"]
            if background and transcript.get("isFinal"):
                # Background speech never reaches the conversation; the log is
                # the only place it shows up.
                logger.info(
                    f"{self}: background speech dropped: "
                    f"[{''.join(w.get('text', '') for w in background).strip()}]"
                )
            words = [w for w in words if w.get("voice") != "background"]
        # Word texts carry their leading space, so concatenation is the text.
        text = "".join(w.get("text", "") for w in words)

        if transcript.get("isFinal"):
            starts = [w.get("start") for w in words if w.get("start") is not None]
            ends = [w.get("end") for w in words if w.get("end") is not None]
            logger.debug(
                f"{self}: final [{text.strip()}] "
                f"({min(starts) if starts else '?'}-{max(ends) if ends else '?'} ms), "
                f"turn text so far [{self._final_text.strip()}]"
            )
            self._final_text += text
            for word in words:
                language = _parse_language(word.get("language"))
                if language:
                    self._detected_language = language
        else:
            # Each partial replaces the previous one.
            self._partial_text = text

        current = (self._final_text + self._partial_text).strip()
        if not current:
            return

        if self._bot_speaking and not self._in_turn and not self._user_turn_open:
            logger.debug(f"{self}: user speech while the bot is speaking [{current}]")
        self._in_turn = True

        # A final shrinks the partial region, so the stale partial could repeat
        # the final's words; the next partial frame updates the interim text.
        if not transcript.get("isFinal"):
            await self.push_frame(
                InterimTranscriptionFrame(
                    current, self._user_id, time_now_iso8601(), self._language(), result=transcript
                )
            )

    async def _on_turn(self, msg: dict):
        await self._cancel_interruption_check()
        logger.debug(
            f"{self}: turn (reason={msg.get('reason')}, "
            f"finalized_until={msg.get('finalized_until')}) text [{self._final_text.strip()}]"
        )
        if self._held_text:
            self._final_text = f" {self._held_text}{self._final_text}"
            self._held_text = ""
        if (
            self._settings.interruption_prediction
            and self._bot_speaking
            and not self._user_turn_open
            and not self._interrupted
        ):
            # Speech over the bot that the server didn't call an interruption.
            # A few words ("uh-huh", "hello?") shouldn't cut the bot off, so
            # they wait for it to finish and become the user's turn then. A
            # whole sentence is an interruption the server missed, so it
            # interrupts now.
            text = self._final_text.strip()
            words = len(text.split())
            max_words = self._settings.backchannel_max_words
            if isinstance(max_words, int) and words <= max_words:
                if text:
                    logger.debug(f"{self}: holding speech over the bot until it finishes [{text}]")
                    self._held_text = text
                    self._held_language = self._language()
                self._reset_turn_text()
                return
            self._interrupted = True
            logger.info(
                f"{self}: INTERRUPTION interrupted=True (reason=client_transcript, words={words})"
            )
            await self.broadcast_frame(ProposedUserStartedSpeakingFrame)

        was_in_turn = self._in_turn
        await self._push_turn_transcription(msg)
        # A turn the pipeline opened can end with no text (a background voice
        # with voice isolation); it must still be closed.
        if (was_in_turn or self._user_turn_open) and self._settings.turn_prediction:
            await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)

    async def _deliver_held_speech(self):
        """Make the speech held during the bot's turn the user's turn now."""
        text, self._held_text = self._held_text, ""
        language, self._held_language = self._held_language, None
        if not text:
            return
        logger.debug(f"{self}: delivering speech held during the bot's turn [{text}]")
        await self.broadcast_frame(ProposedUserStartedSpeakingFrame)
        self._final_text = text
        self._detected_language = language
        await self._push_turn_transcription({})
        await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)

    def _reset_turn_text(self):
        self._final_text = ""
        self._partial_text = ""
        self._detected_language = None
        self._in_turn = False
        if self._interrupted:
            logger.info(f"{self}: interrupted=False (interrupted turn ended)")
        self._interrupted = False

    async def _push_turn_transcription(self, result: dict):
        text = self._final_text.strip()
        language = self._language()
        self._reset_turn_text()
        if not text:
            return

        # Report usage before the transcription frame so tracing can attach it
        # to the STT span the frame closes.
        await self.emit_stt_usage_metrics()
        logger.info(f"{self}: TRANSCRIPTION final=True [{text}]")
        await self.push_frame(
            TranscriptionFrame(
                text, self._user_id, time_now_iso8601(), language, result=result, finalized=True
            )
        )
        await self._handle_transcription(text, True, language)

    def _language(self) -> Language | None:
        if self._detected_language:
            return self._detected_language
        language = self._settings.language
        if isinstance(language, Language):
            return language
        return _parse_language(language if isinstance(language, str) else None)

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Record a final transcription for tracing."""
        pass


def _parse_language(code: str | None) -> Language | None:
    if not code:
        return None
    try:
        return Language(code)
    except ValueError:
        return None
