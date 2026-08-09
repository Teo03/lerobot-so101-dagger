# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DAgger rollout strategy: Human-in-the-Loop data collection.

Implements the RaC paradigm (Recovery and Correction) for interactive
imitation learning.  Alternates between autonomous policy execution and
human intervention via teleoperator.

Input is controlled via either a keyboard or foot pedal, selected by
the ``input_device`` config field. The phase-advance control cycles through:

    1. **pause_resume** — Advance AUTONOMOUS → PAUSED → CORRECTING → AUTONOMOUS.
    2. **upload**        — Push dataset to hub on demand (corrections-only mode).
    ESC (keyboard only) — Stop session.

Recording modes:
    ``record_autonomous=True``:  Sentry-like continuous recording with
        time-based episode rotation.  Both autonomous and correction
        frames are recorded; corrections tagged ``intervention=True``.
    ``record_autonomous=False``: Only correction windows are recorded.
        Each correction (start to stop) becomes one episode.

Teleoperator handover:
    On AUTONOMOUS → PAUSED, actuated teleops (those with non-empty
    ``feedback_features``, e.g. SO-101, OpenArmMini) are smoothly driven to
    the follower's measured position via ``send_feedback`` so the operator takes
    over without a jerk.  Non-actuated teleops cannot be driven,
    so on PAUSED → CORRECTING the follower is instead slid to the teleop's
    current pose before the correction begins.
"""

from __future__ import annotations

import contextlib
import enum
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event, Lock
from typing import Any

import numpy as np

from lerobot.common.control_utils import (
    follower_smooth_move_to,
    teleop_smooth_move_to,
    teleop_supports_feedback,
)
from lerobot.datasets import VideoEncodingManager
from lerobot.datasets.utils import DEFAULT_VIDEO_FILE_SIZE_IN_MB
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.keyboard_input import create_key_listener
from lerobot.utils.pedal import start_pedal_listener
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

from ..configs import DAggerKeyboardConfig, DAggerPedalConfig, DAggerStrategyConfig
from ..context import RolloutContext
from .core import RolloutStrategy, estimate_max_episode_seconds, safe_push_to_hub, send_next_action

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DAgger state machine
# ---------------------------------------------------------------------------


class DAggerPhase(enum.Enum):
    """Observable phases of a DAgger episode."""

    AUTONOMOUS = "autonomous"  # Policy driving
    PAUSED = "paused"  # Engine paused, teleop aligned, awaiting input
    CORRECTING = "correcting"  # Human driving via teleop, recording interventions


# Valid (current_phase, event) -> next_phase
_DAGGER_TRANSITIONS: dict[tuple[DAggerPhase, str], DAggerPhase] = {
    (DAggerPhase.AUTONOMOUS, "pause_resume"): DAggerPhase.PAUSED,
    (DAggerPhase.PAUSED, "pause_resume"): DAggerPhase.CORRECTING,
    (DAggerPhase.CORRECTING, "pause_resume"): DAggerPhase.AUTONOMOUS,
}


class DAggerEvents:
    """Thread-safe container for DAgger input device events.

    The keyboard/pedal threads write transition requests; the main loop
    consumes them.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._phase = DAggerPhase.AUTONOMOUS
        self._pending_transition: str | None = None

        # Session-level flags
        self.stop_recording = Event()
        self.upload_requested = Event()

    # -- Thread-safe phase access ------------------------------------------

    @property
    def phase(self) -> DAggerPhase:
        """Current phase of the DAgger state machine."""
        with self._lock:
            return self._phase

    @phase.setter
    def phase(self, value: DAggerPhase) -> None:
        with self._lock:
            self._phase = value

    def request_transition(self, event: str) -> None:
        """Request a phase transition (called from keyboard/pedal threads).

        Only enqueues the request if it corresponds to a valid transition
        from the current phase, preventing impossible state changes.
        """
        with self._lock:
            if (self._phase, event) in _DAGGER_TRANSITIONS:
                self._pending_transition = event

    def consume_transition(self) -> tuple[DAggerPhase, DAggerPhase] | None:
        """Consume a pending transition (called from main loop)."""
        with self._lock:
            if self._pending_transition is None:
                return None
            key = (self._phase, self._pending_transition)
            self._pending_transition = None
            new_phase = _DAGGER_TRANSITIONS.get(key)
            if new_phase is None:
                return None
            old_phase = self._phase
            self._phase = new_phase
            return old_phase, new_phase

    def reset(self) -> None:
        """Reset all transient state for a fresh session."""
        with self._lock:
            self._phase = DAggerPhase.AUTONOMOUS
            self._pending_transition = None
        self.upload_requested.clear()


# ---------------------------------------------------------------------------
# Input device handlers
# ---------------------------------------------------------------------------


def _init_dagger_keyboard(events: DAggerEvents, cfg: DAggerKeyboardConfig):
    """Initialise a keyboard listener for DAgger's single phase-advance control.

    Backend selection (pynput on X11 / trusted-macOS / Windows, a terminal reader on
    Wayland / headless TTY) is delegated to :func:`create_key_listener`. Returns the
    listener (exposing ``stop()``) or ``None`` when no keyboard backend is usable.
    """
    def dispatch(name: str) -> None:
        """Apply a resolved key name to the DAgger events."""
        if name == "esc":
            logger.info("Stop recording...")
            events.stop_recording.set()
            return
        if name == cfg.pause_resume:
            events.request_transition("pause_resume")
        if name == cfg.upload:
            events.upload_requested.set()

    return create_key_listener(
        dispatch,
        controls_help=(
            f"advance='{cfg.pause_resume}' (ACT→align→correct→ACT), "
            f"upload='{cfg.upload}', ESC=finish"
        ),
    )


def _init_dagger_pedal(events: DAggerEvents, cfg: DAggerPedalConfig):
    """Initialise a pedal listener with a single phase-advance pedal.

    Returns the pedal listener thread (or ``None`` if evdev is unavailable).
    """
    def on_press(code: str) -> None:
        if code == cfg.pause_resume:
            events.request_transition("pause_resume")
        if code == cfg.upload:
            events.upload_requested.set()

    logger.info("Initializing DAgger foot pedal listener (device=%s)", cfg.device_path)
    return start_pedal_listener(on_press, device_path=cfg.device_path)


# ---------------------------------------------------------------------------
# DAgger Strategy
# ---------------------------------------------------------------------------


class DAggerStrategy(RolloutStrategy):
    """Human-in-the-Loop data collection with intervention tagging.

    State machine::

        AUTONOMOUS --(key1)--> PAUSED --(key2)--> CORRECTING --(key2)--> PAUSED
                               --(key1)--> AUTONOMOUS

    Recording modes:
        ``record_autonomous=True``: Sentry-like continuous recording with
            time-based episode rotation.  Intervention frames tagged True.
        ``record_autonomous=False``: Only correction windows recorded.
            Each correction = one episode.  Upload on demand via key3.
    """

    config: DAggerStrategyConfig

    def __init__(self, config: DAggerStrategyConfig):
        super().__init__(config)
        self._listener = None
        self._pedal_thread = None
        self._events = DAggerEvents()
        self._push_executor: ThreadPoolExecutor | None = None
        self._pending_push: Future | None = None
        self._needs_push = Event()
        self._episode_lock = Lock()
        self._clutch_leader_origin: dict[str, float] | None = None
        self._clutch_follower_origin: dict[str, float] | None = None
        self._pause_hold_action: dict[str, float] | None = None
        self._leader_start_pose: dict[str, float] | None = None

    def _apply_relative_clutch(self, teleop_action: dict[str, float]) -> dict[str, float]:
        """Convert leader motion since clutch engagement into follower-space targets."""
        if self._clutch_leader_origin is None or self._clutch_follower_origin is None:
            raise RuntimeError("Relative clutch used before its origins were captured")
        adjusted: dict[str, float] = {}
        for key, value in teleop_action.items():
            if key in self._clutch_leader_origin and key in self._clutch_follower_origin:
                adjusted[key] = self._clutch_follower_origin[key] + value - self._clutch_leader_origin[key]
            else:
                adjusted[key] = value
        return adjusted

    def setup(self, ctx: RolloutContext) -> None:
        """Initialise the inference engine and input device listener."""
        self._init_engine(ctx)
        teleop = ctx.hardware.teleop
        if (
            teleop is not None
            and teleop_supports_feedback(teleop)
            and self.config.smooth_leader_to_follower_handover
            and not self.config.relative_clutch_handover
        ):
            self._leader_start_pose = {key: float(value) for key, value in teleop.get_action().items()}
            logger.info("Captured leader startup pose (%d joints)", len(self._leader_start_pose))
        self._push_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dagger-push")
        target_mb = self.config.target_video_file_size_mb or DEFAULT_VIDEO_FILE_SIZE_IN_MB
        self._episode_duration_s = estimate_max_episode_seconds(
            ctx.data.dataset_features, ctx.runtime.cfg.fps, target_size_mb=target_mb
        )

        if self.config.input_device == "keyboard":
            self._listener = _init_dagger_keyboard(self._events, self.config.keyboard)
        else:
            self._pedal_thread = _init_dagger_pedal(self._events, self.config.pedal)

        record_mode = "all frames (sentry-like)" if self.config.record_autonomous else "corrections only"
        logger.info(
            "DAgger strategy ready (input=%s, episodes=%d, record=%s, episode_duration=%.0fs)",
            self.config.input_device,
            self.config.num_episodes,
            record_mode,
            self._episode_duration_s,
        )

    def run(self, ctx: RolloutContext) -> None:
        """Run DAgger episodes with human-in-the-loop intervention."""
        if self.config.record_autonomous:
            self._run_continuous(ctx)
        else:
            self._run_corrections_only(ctx)

    def teardown(self, ctx: RolloutContext) -> None:
        """Stop listeners, finalise the dataset, and disconnect hardware."""
        play_sounds = ctx.runtime.cfg.play_sounds
        logger.info("Stopping DAgger recording")
        log_say("Stopping DAgger recording", play_sounds)

        if self._listener is not None:
            logger.info("Stopping keyboard listener")
            self._listener.stop()

        # Flush any queued/running push cleanly
        if self._push_executor is not None:
            logger.info("Shutting down push executor (waiting for pending pushes)...")
            self._push_executor.shutdown(wait=True)
            self._push_executor = None

        if ctx.data.dataset is not None:
            logger.info("Finalizing dataset...")
            ctx.data.dataset.finalize()
            if self._needs_push.is_set() and ctx.runtime.cfg.dataset and ctx.runtime.cfg.dataset.push_to_hub:
                logger.info("Pushing final dataset to hub...")
                if safe_push_to_hub(
                    ctx.data.dataset,
                    tags=ctx.runtime.cfg.dataset.tags,
                    private=ctx.runtime.cfg.dataset.private,
                ):
                    logger.info("Dataset uploaded to hub")
                    log_say("Dataset uploaded to hub", play_sounds)

        teleop = ctx.hardware.teleop
        if teleop is not None and teleop.is_connected and teleop_supports_feedback(teleop):
            try:
                teleop.disable_torque()
            except Exception as exc:
                logger.warning("Could not disable leader torque during teardown: %s", exc)

        self._teardown_hardware(
            ctx.hardware,
            return_to_initial_position=ctx.runtime.cfg.return_to_initial_position,
        )
        logger.info("DAgger strategy teardown complete")

    # ------------------------------------------------------------------
    # Continuous recording mode (record_autonomous=True)
    # ------------------------------------------------------------------

    def _run_continuous(self, ctx: RolloutContext) -> None:
        """Sentry-like continuous recording with intervention tagging.

        Episodes are auto-rotated every ``episode_time_s`` seconds and
        uploaded in the background every ``upload_every_n_episodes`` episodes.
        Both autonomous and correction frames are recorded; corrections are
        tagged with ``intervention=True``.
        """
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        teleop = ctx.hardware.teleop
        dataset = ctx.data.dataset
        events = self._events
        interpolator = self._interpolator
        features = ctx.data.dataset_features

        control_interval = interpolator.get_control_interval(cfg.fps)
        record_stride = max(1, cfg.interpolation_multiplier)
        task_str = cfg.dataset.single_task if cfg.dataset else cfg.task
        play_sounds = cfg.play_sounds

        engine.reset()
        interpolator.reset()
        events.reset()
        engine.resume()

        last_action: dict[str, Any] | None = None
        record_tick = 0
        start_time = time.perf_counter()
        episode_start = time.perf_counter()
        episodes_since_push = 0
        episode_duration_s = self._episode_duration_s
        logger.info("DAgger continuous recording started (episode_duration=%.0fs)", episode_duration_s)

        with VideoEncodingManager(dataset):
            try:
                while not events.stop_recording.is_set() and not ctx.runtime.shutdown_event.is_set():
                    loop_start = time.perf_counter()

                    if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                        logger.info("Duration limit reached (%.0fs)", cfg.duration)
                        break

                    # Process transitions
                    transition = events.consume_transition()
                    if transition is not None:
                        old_phase, new_phase = transition
                        self._apply_transition(
                            old_phase,
                            new_phase,
                            engine,
                            interpolator,
                            ctx,
                            last_action,
                        )
                        if new_phase == DAggerPhase.AUTONOMOUS:
                            last_action = None

                    phase = events.phase
                    obs = robot.get_observation()

                    # --- CORRECTING: human teleop control ---
                    # TODO(Steven): teleop runs at the same FPS as the policy. To
                    # decouple the two, sample teleop at its native rate and
                    # interpolate to the control loop's tick rate.
                    if phase == DAggerPhase.CORRECTING:
                        obs_processed = ctx.processors.robot_observation_processor(obs)
                        teleop_action = teleop.get_action()
                        if self.config.relative_clutch_handover:
                            teleop_action = self._apply_relative_clutch(teleop_action)
                        processed_teleop = ctx.processors.teleop_action_processor((teleop_action, obs))
                        robot_action_to_send = ctx.processors.robot_action_processor((processed_teleop, obs))
                        robot.send_action(robot_action_to_send)
                        last_action = robot_action_to_send
                        self._log_telemetry(obs_processed, processed_teleop, ctx.runtime)
                        if record_tick % record_stride == 0:
                            obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
                            action_frame = build_dataset_frame(features, processed_teleop, prefix=ACTION)
                            frame = {
                                **obs_frame,
                                **action_frame,
                                "task": task_str,
                                "intervention": np.array([True], dtype=bool),
                            }
                            dataset.add_frame(frame)
                        record_tick += 1

                    # --- PAUSED: hold position ---
                    elif phase == DAggerPhase.PAUSED:
                        hold_action = self._pause_hold_action or last_action
                        if hold_action:
                            robot.send_action(hold_action)

                    # --- AUTONOMOUS: policy control ---
                    else:
                        obs_processed = self._process_observation_and_notify(ctx.processors, obs)

                        if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                            continue

                        action_dict = send_next_action(obs_processed, obs, ctx, interpolator)
                        if action_dict is not None:
                            self._log_telemetry(obs_processed, action_dict, ctx.runtime)
                            last_action = ctx.processors.robot_action_processor((action_dict, obs))
                            if record_tick % record_stride == 0:
                                obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
                                action_frame = build_dataset_frame(features, action_dict, prefix=ACTION)
                                frame = {
                                    **obs_frame,
                                    **action_frame,
                                    "task": task_str,
                                    "intervention": np.array([False], dtype=bool),
                                }
                                dataset.add_frame(frame)
                            record_tick += 1

                    # Episode rotation derived from the video file-size target.
                    # Saving is deferred while a correction is ongoing so the
                    # episode boundary lands on a clean autonomous frame.
                    elapsed = time.perf_counter() - episode_start
                    if elapsed >= episode_duration_s and phase != DAggerPhase.CORRECTING:
                        with self._episode_lock:
                            dataset.save_episode()
                        episodes_since_push += 1
                        self._needs_push.set()
                        logger.info(
                            "Episode saved (total: %d, elapsed: %.1fs)",
                            dataset.num_episodes,
                            elapsed,
                        )
                        log_say(f"Episode {dataset.num_episodes} saved", play_sounds)

                        if episodes_since_push >= self.config.upload_every_n_episodes:
                            self._background_push(dataset, cfg)
                            episodes_since_push = 0

                        episode_start = time.perf_counter()

                    dt = time.perf_counter() - loop_start
                    if (sleep_t := control_interval - dt) > 0:
                        precise_sleep(sleep_t)
                    else:
                        logger.warning(
                            f"Record loop is running slower ({1 / dt:.1f} Hz) than the target FPS ({cfg.fps} Hz). Dataset frames might be dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation"
                        )

            finally:
                logger.info("DAgger continuous control loop ended — pausing engine")
                engine.pause()
                with contextlib.suppress(Exception):
                    with self._episode_lock:
                        dataset.save_episode()
                    self._needs_push.set()
                    logger.info("Final in-progress episode saved")

    # ------------------------------------------------------------------
    # Corrections-only mode (record_autonomous=False)
    # ------------------------------------------------------------------

    def _run_corrections_only(self, ctx: RolloutContext) -> None:
        """Record only human correction windows.  Each correction = one episode.

        The policy runs autonomously without recording.  When the user
        pauses and starts a correction, frames are recorded with
        ``intervention=True``.  Stopping the correction saves the episode.
        The dataset can be uploaded on demand via the upload key/pedal.
        """
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        teleop = ctx.hardware.teleop
        dataset = ctx.data.dataset
        events = self._events
        interpolator = self._interpolator
        features = ctx.data.dataset_features

        control_interval = interpolator.get_control_interval(cfg.fps)
        record_stride = max(1, cfg.interpolation_multiplier)
        task_str = cfg.dataset.single_task if cfg.dataset else cfg.task
        play_sounds = cfg.play_sounds

        engine.reset()
        interpolator.reset()
        events.reset()
        engine.resume()

        last_action: dict[str, Any] | None = None
        start_time = time.perf_counter()
        record_tick = 0
        recorded = 0
        logger.info(
            "DAgger corrections-only recording started (target: %d episodes)", self.config.num_episodes
        )

        with VideoEncodingManager(dataset):
            try:
                while (
                    recorded < self.config.num_episodes
                    and not events.stop_recording.is_set()
                    and not ctx.runtime.shutdown_event.is_set()
                ):
                    loop_start = time.perf_counter()

                    if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                        logger.info("Duration limit reached (%.0fs)", cfg.duration)
                        break

                    # Process transitions
                    transition = events.consume_transition()
                    if transition is not None:
                        old_phase, new_phase = transition
                        correction_to_autonomous = (
                            old_phase == DAggerPhase.CORRECTING
                            and new_phase == DAggerPhase.AUTONOMOUS
                        )
                        self._apply_transition(
                            old_phase,
                            new_phase,
                            engine,
                            interpolator,
                            ctx,
                            last_action,
                            defer_autonomous_resume=correction_to_autonomous,
                        )
                        if new_phase == DAggerPhase.AUTONOMOUS:
                            last_action = None

                        # Correction ended -> save episode (blocking if not streaming)
                        if correction_to_autonomous:
                            with self._episode_lock:
                                dataset.save_episode()
                            recorded += 1
                            self._needs_push.set()
                            logger.info(
                                "Correction %d/%d saved",
                                recorded,
                                self.config.num_episodes,
                            )
                            log_say(f"Correction {recorded} saved", play_sounds)
                            engine.resume()
                            logger.info("Autonomous mode resumed after correction save")

                    # On-demand upload
                    if events.upload_requested.is_set():
                        events.upload_requested.clear()
                        logger.info("Upload requested by user")
                        self._background_push(dataset, cfg)

                    phase = events.phase
                    obs = robot.get_observation()

                    # --- CORRECTING: human teleop control + recording ---
                    # TODO(Steven): teleop runs at the same FPS as the policy. To
                    # decouple the two, sample teleop at its native rate and
                    # interpolate to the control loop's tick rate.
                    if phase == DAggerPhase.CORRECTING:
                        obs_processed = ctx.processors.robot_observation_processor(obs)
                        teleop_action = teleop.get_action()
                        if self.config.relative_clutch_handover:
                            teleop_action = self._apply_relative_clutch(teleop_action)
                        processed_teleop = ctx.processors.teleop_action_processor((teleop_action, obs))
                        robot_action_to_send = ctx.processors.robot_action_processor((processed_teleop, obs))
                        robot.send_action(robot_action_to_send)
                        last_action = robot_action_to_send
                        self._log_telemetry(obs_processed, processed_teleop, ctx.runtime)

                        if record_tick % record_stride == 0:
                            obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
                            action_frame = build_dataset_frame(features, processed_teleop, prefix=ACTION)
                            dataset.add_frame(
                                {
                                    **obs_frame,
                                    **action_frame,
                                    "task": task_str,
                                    "intervention": np.array([True], dtype=bool),
                                }
                            )
                        record_tick += 1

                    # --- PAUSED: hold position ---
                    elif phase == DAggerPhase.PAUSED:
                        hold_action = self._pause_hold_action or last_action
                        if hold_action:
                            robot.send_action(hold_action)

                    # --- AUTONOMOUS: policy control (no recording) ---
                    else:
                        obs_processed = self._process_observation_and_notify(ctx.processors, obs)

                        if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                            continue

                        action_dict = send_next_action(obs_processed, obs, ctx, interpolator)
                        if action_dict is not None:
                            self._log_telemetry(obs_processed, action_dict, ctx.runtime)
                            last_action = ctx.processors.robot_action_processor((action_dict, obs))

                    dt = time.perf_counter() - loop_start
                    if (sleep_t := control_interval - dt) > 0:
                        precise_sleep(sleep_t)
                    else:
                        logger.warning(
                            f"Record loop is running slower ({1 / dt:.1f} Hz) than the target FPS ({cfg.fps} Hz). Dataset frames might be dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation"
                        )

            finally:
                logger.info("DAgger corrections-only loop ended — pausing engine")
                engine.pause()
                with contextlib.suppress(Exception):
                    with self._episode_lock:
                        dataset.save_episode()
                    self._needs_push.set()
                    logger.info("Final in-progress episode saved")

    # ------------------------------------------------------------------
    # State-machine transition side-effects
    # ------------------------------------------------------------------

    def _apply_transition(
        self,
        old_phase: DAggerPhase,
        new_phase: DAggerPhase,
        engine,
        interpolator,
        ctx: RolloutContext,
        prev_action: dict | None,
        *,
        defer_autonomous_resume: bool = False,
    ) -> None:
        """Execute side-effects for a validated phase transition, including smooth handovers.

        AUTONOMOUS -> PAUSED (actuated teleop):
            Pause the engine, then drive the leader arm to the follower's last
            commanded position so the operator takes over without a jerk.

        PAUSED -> CORRECTING (non-actuated teleop):
            Slide the follower to the teleop's current pose so the robot meets
            the operator's hand rather than jumping to it on the first frame.

        CORRECTING -> AUTONOMOUS:
            Return the leader to its startup pose, then reset and resume inference.
        """
        teleop = ctx.hardware.teleop
        robot = ctx.hardware.robot_wrapper

        logger.info("Phase transition: %s -> %s", old_phase.value, new_phase.value)
        if old_phase == DAggerPhase.AUTONOMOUS and new_phase == DAggerPhase.PAUSED:
            logger.info("Pausing engine - robot holds position")
            engine.pause()
            measured = robot.get_observation()
            self._pause_hold_action = {k: float(v) for k, v in measured.items() if k.endswith(".pos")}
            logger.info("Captured measured follower hold pose (%d joints)", len(self._pause_hold_action))

            if (
                teleop_supports_feedback(teleop)
                and not self.config.relative_clutch_handover
                and self.config.smooth_leader_to_follower_handover
            ):
                # The measured hold pose is in robot action key space.
                # send_feedback expects teleop feedback key space. For homogeneous setups (e.g. SO-101
                # leader + SO-101 follower) the keys are identical so this works. If the processor pipeline
                # does non-trivial key renaming (e.g. a rename_map on action keys), the interpolation in
                # teleop_smooth_move_to silently no-ops and the arm doesn't move.
                # Use the measured follower position rather than prev_action: policy actions can be ahead
                # of the physical arm, especially when a follower-side relative target limiter is enabled.
                logger.info("Smooth handover: moving leader arm to measured follower position")
                teleop_smooth_move_to(teleop, self._pause_hold_action)
            elif teleop_supports_feedback(teleop) and not self.config.relative_clutch_handover:
                logger.info(
                    "Manual handover: leader remains torque-free; align it to the follower, then request correction"
                )

        elif old_phase == DAggerPhase.PAUSED and new_phase == DAggerPhase.CORRECTING:
            logger.info("Entering correction mode - human teleop control")
            if self.config.relative_clutch_handover:
                if self._pause_hold_action is None:
                    measured = robot.get_observation()
                    self._pause_hold_action = {
                        k: float(v) for k, v in measured.items() if k.endswith(".pos")
                    }
                self._clutch_leader_origin = teleop.get_action()
                self._clutch_follower_origin = dict(self._pause_hold_action)
                logger.info("Relative clutch engaged at current leader/follower poses")
            elif teleop_supports_feedback(teleop) and not self.config.smooth_leader_to_follower_handover:
                leader_pose = teleop.get_action()
                differences = {
                    key: abs(float(leader_pose[key]) - follower_value)
                    for key, follower_value in (self._pause_hold_action or {}).items()
                    if key in leader_pose
                }
                too_far = {
                    key: delta
                    for key, delta in differences.items()
                    if delta > self.config.manual_handover_max_delta
                }
                if too_far:
                    self._events.phase = DAggerPhase.PAUSED
                    details = ", ".join(f"{key}={delta:.1f}" for key, delta in sorted(too_far.items()))
                    logger.warning(
                        "Correction rejected: manually align leader to follower within %.1f per joint (%s)",
                        self.config.manual_handover_max_delta,
                        details,
                    )
                    return
                logger.info(
                    "Manual handover alignment accepted (maximum joint difference %.1f)",
                    max(differences.values(), default=0.0),
                )
            if not teleop_supports_feedback(teleop) and prev_action is not None:
                logger.info("Smooth handover: sliding follower to teleop position")
                obs = robot.get_observation()
                teleop_action = teleop.get_action()
                processed = ctx.processors.teleop_action_processor((teleop_action, obs))
                target = ctx.processors.robot_action_processor((processed, obs))
                follower_smooth_move_to(robot, prev_action, target)

            # unlock the teleop for human control
            if (
                teleop_supports_feedback(teleop)
                and self.config.smooth_leader_to_follower_handover
                and not self.config.relative_clutch_handover
            ):
                teleop.disable_torque()

        elif old_phase == DAggerPhase.CORRECTING and new_phase == DAggerPhase.PAUSED:
            if (
                teleop_supports_feedback(teleop)
                and not self.config.relative_clutch_handover
                and self.config.smooth_leader_to_follower_handover
            ):
                teleop.enable_torque()
            self._clutch_leader_origin = None
            self._clutch_follower_origin = None
            measured = robot.get_observation()
            self._pause_hold_action = {k: float(v) for k, v in measured.items() if k.endswith(".pos")}

        elif new_phase == DAggerPhase.AUTONOMOUS:
            logger.info("Preparing to resume autonomous mode")
            interpolator.reset()
            engine.reset()

            # Park and release the leader before the policy starts moving the
            # follower again, so the operator's workspace is reset each cycle.
            if teleop_supports_feedback(teleop) and self.config.smooth_leader_to_follower_handover:
                if self._leader_start_pose is not None:
                    logger.info("Returning leader to startup pose before resuming autonomous mode")
                    teleop_smooth_move_to(teleop, self._leader_start_pose)
                teleop.disable_torque()
            if defer_autonomous_resume:
                logger.info("Leader parked; autonomous resume deferred until correction is saved")
            else:
                engine.resume()
                logger.info("Autonomous mode resumed")
            self._clutch_leader_origin = None
            self._clutch_follower_origin = None
            self._pause_hold_action = None

    # ------------------------------------------------------------------
    # Background push (shared by both modes)
    # ------------------------------------------------------------------

    def _background_push(self, dataset, cfg) -> None:
        """Queue a Hub push on the single-worker executor.

        The executor's max_workers=1 guarantees at most one push runs at
        a time; submitted tasks are queued rather than dropped.  Pushes
        are blocked while the operator is mid-correction to avoid
        uploading a partially-recorded episode.
        """
        if self._push_executor is None:
            return

        if self._events.phase == DAggerPhase.CORRECTING:
            logger.info("Skipping push — correction in progress")
            return

        if self._pending_push is not None and not self._pending_push.done():
            logger.info("Previous push still in progress; queueing next")

        def _push():
            try:
                with self._episode_lock:
                    if safe_push_to_hub(
                        dataset,
                        tags=cfg.dataset.tags if cfg.dataset else None,
                        private=cfg.dataset.private if cfg.dataset else False,
                    ):
                        self._needs_push.clear()
                        logger.info("Background push to hub complete")
            except Exception as e:
                logger.error("Background push failed: %s", e)

        self._pending_push = self._push_executor.submit(_push)
        logger.info("Background push task submitted")
