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

"""Minimal tests for the rollout module's public API."""

from __future__ import annotations

import dataclasses
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------


def test_rollout_top_level_imports():
    import lerobot.rollout

    for name in lerobot.rollout.__all__:
        assert hasattr(lerobot.rollout, name), f"Missing export: {name}"


def test_inference_submodule_imports():
    import lerobot.rollout.inference

    for name in lerobot.rollout.inference.__all__:
        assert hasattr(lerobot.rollout.inference, name), f"Missing export: {name}"


def test_strategies_submodule_imports():
    import lerobot.rollout.strategies

    for name in lerobot.rollout.strategies.__all__:
        assert hasattr(lerobot.rollout.strategies, name), f"Missing export: {name}"


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_strategy_config_types():
    from lerobot.rollout import (
        BaseStrategyConfig,
        DAggerStrategyConfig,
        EpisodicStrategyConfig,
        HighlightStrategyConfig,
        SentryStrategyConfig,
    )

    assert BaseStrategyConfig().type == "base"
    assert SentryStrategyConfig().type == "sentry"
    assert HighlightStrategyConfig().type == "highlight"
    assert DAggerStrategyConfig().type == "dagger"
    assert EpisodicStrategyConfig().type == "episodic"


def test_dagger_config_invalid_input_device():
    from lerobot.rollout import DAggerStrategyConfig

    with pytest.raises(ValueError, match="input_device must be 'keyboard' or 'pedal'"):
        DAggerStrategyConfig(input_device="joystick")


def test_dagger_config_defaults():
    from lerobot.rollout import DAggerStrategyConfig

    cfg = DAggerStrategyConfig()
    assert cfg.num_episodes is None
    assert cfg.record_autonomous is False
    assert cfg.input_device == "keyboard"
    assert cfg.relative_clutch_handover is False
    assert cfg.smooth_leader_to_follower_handover is True
    assert cfg.manual_handover_max_delta == 10.0


def test_dagger_config_rejects_nonpositive_manual_handover_delta():
    from lerobot.rollout import DAggerStrategyConfig

    with pytest.raises(ValueError, match="manual_handover_max_delta must be greater than zero"):
        DAggerStrategyConfig(manual_handover_max_delta=0)


def test_dagger_relative_clutch_maps_leader_deltas_into_follower_space():
    from lerobot.rollout import DAggerStrategy, DAggerStrategyConfig

    strategy = DAggerStrategy(DAggerStrategyConfig(relative_clutch_handover=True))
    strategy._clutch_leader_origin = {"elbow_flex.pos": -30.0, "gripper.pos": 20.0}
    strategy._clutch_follower_origin = {"elbow_flex.pos": -40.0, "gripper.pos": 35.0}

    mapped = strategy._apply_relative_clutch({"elbow_flex.pos": -35.0, "gripper.pos": 22.0})

    assert mapped == {"elbow_flex.pos": -45.0, "gripper.pos": 37.0}


def test_dagger_absolute_handover_uses_measured_follower_pose():
    from lerobot.rollout import DAggerStrategy, DAggerStrategyConfig
    from lerobot.rollout.strategies import DAggerPhase

    strategy = DAggerStrategy(DAggerStrategyConfig())
    ctx = MagicMock()
    teleop = ctx.hardware.teleop
    teleop.feedback_features = {"elbow_flex.pos": float}
    robot = ctx.hardware.robot_wrapper
    robot.get_observation.return_value = {
        "elbow_flex.pos": -41.0,
        "front": object(),
    }

    with patch("lerobot.rollout.strategies.dagger.teleop_smooth_move_to") as smooth_move:
        strategy._apply_transition(
            DAggerPhase.AUTONOMOUS,
            DAggerPhase.PAUSED,
            MagicMock(),
            MagicMock(),
            ctx,
            {"elbow_flex.pos": -150.0},
        )

    smooth_move.assert_called_once_with(teleop, {"elbow_flex.pos": -41.0})


def test_dagger_manual_handover_rejects_misaligned_leader_without_torque_write():
    from lerobot.rollout import DAggerStrategy, DAggerStrategyConfig
    from lerobot.rollout.strategies import DAggerPhase

    strategy = DAggerStrategy(
        DAggerStrategyConfig(
            smooth_leader_to_follower_handover=False,
            manual_handover_max_delta=10.0,
        )
    )
    strategy._events.phase = DAggerPhase.CORRECTING
    strategy._pause_hold_action = {"elbow_flex.pos": -40.0}
    ctx = MagicMock()
    teleop = ctx.hardware.teleop
    teleop.feedback_features = {"elbow_flex.pos": float}
    teleop.get_action.return_value = {"elbow_flex.pos": -65.0}

    strategy._apply_transition(
        DAggerPhase.PAUSED,
        DAggerPhase.CORRECTING,
        MagicMock(),
        MagicMock(),
        ctx,
        None,
    )

    assert strategy._events.phase == DAggerPhase.PAUSED
    teleop.enable_torque.assert_not_called()
    teleop.disable_torque.assert_not_called()


def test_dagger_manual_handover_accepts_aligned_leader_without_torque_write():
    from lerobot.rollout import DAggerStrategy, DAggerStrategyConfig
    from lerobot.rollout.strategies import DAggerPhase

    strategy = DAggerStrategy(
        DAggerStrategyConfig(
            smooth_leader_to_follower_handover=False,
            manual_handover_max_delta=10.0,
        )
    )
    strategy._events.phase = DAggerPhase.CORRECTING
    strategy._pause_hold_action = {"elbow_flex.pos": -40.0}
    ctx = MagicMock()
    teleop = ctx.hardware.teleop
    teleop.feedback_features = {"elbow_flex.pos": float}
    teleop.get_action.return_value = {"elbow_flex.pos": -43.0}

    strategy._apply_transition(
        DAggerPhase.PAUSED,
        DAggerPhase.CORRECTING,
        MagicMock(),
        MagicMock(),
        ctx,
        None,
    )

    assert strategy._events.phase == DAggerPhase.CORRECTING
    teleop.enable_torque.assert_not_called()
    teleop.disable_torque.assert_not_called()


def test_dagger_returns_leader_to_startup_pose_before_resuming_policy():
    from lerobot.rollout import DAggerStrategy, DAggerStrategyConfig
    from lerobot.rollout.strategies import DAggerPhase

    strategy = DAggerStrategy(DAggerStrategyConfig())
    strategy._leader_start_pose = {"elbow_flex.pos": -25.0}
    ctx = MagicMock()
    teleop = ctx.hardware.teleop
    teleop.feedback_features = {"elbow_flex.pos": float}
    engine = MagicMock()

    with patch("lerobot.rollout.strategies.dagger.teleop_smooth_move_to") as smooth_move:
        strategy._apply_transition(
            DAggerPhase.PAUSED,
            DAggerPhase.AUTONOMOUS,
            engine,
            MagicMock(),
            ctx,
            None,
        )

    smooth_move.assert_called_once_with(teleop, {"elbow_flex.pos": -25.0})
    teleop.disable_torque.assert_called_once_with()
    engine.resume.assert_called_once_with()


def test_dagger_can_park_leader_while_deferring_policy_resume():
    from lerobot.rollout import DAggerStrategy, DAggerStrategyConfig
    from lerobot.rollout.strategies import DAggerPhase

    strategy = DAggerStrategy(DAggerStrategyConfig())
    strategy._leader_start_pose = {"elbow_flex.pos": -25.0}
    ctx = MagicMock()
    teleop = ctx.hardware.teleop
    teleop.feedback_features = {"elbow_flex.pos": float}
    engine = MagicMock()

    with patch("lerobot.rollout.strategies.dagger.teleop_smooth_move_to") as smooth_move:
        strategy._apply_transition(
            DAggerPhase.CORRECTING,
            DAggerPhase.AUTONOMOUS,
            engine,
            MagicMock(),
            ctx,
            None,
            defer_autonomous_resume=True,
        )

    smooth_move.assert_called_once_with(teleop, {"elbow_flex.pos": -25.0})
    teleop.disable_torque.assert_called_once_with()
    engine.resume.assert_not_called()


def test_so_leader_torque_enable_retries_each_motor():
    from lerobot.teleoperators.so_leader.so_leader import TORQUE_COMM_RETRIES, SOLeader

    leader = object.__new__(SOLeader)
    leader.bus = MagicMock()
    leader.bus.motors = {"shoulder_pan": object(), "gripper": object()}

    leader.enable_torque()

    assert leader.bus.enable_torque.call_args_list == [
        (("shoulder_pan",), {"num_retry": TORQUE_COMM_RETRIES}),
        (("gripper",), {"num_retry": TORQUE_COMM_RETRIES}),
    ]


def test_so_leader_torque_enable_rolls_back_partial_success():
    from lerobot.teleoperators.so_leader.so_leader import TORQUE_COMM_RETRIES, SOLeader

    leader = object.__new__(SOLeader)
    leader.bus = MagicMock()
    leader.bus.motors = {"shoulder_pan": object(), "elbow_flex": object(), "gripper": object()}
    leader.bus.enable_torque.side_effect = [None, None, ConnectionError("no status packet")]

    with pytest.raises(ConnectionError, match="no status packet"):
        leader.enable_torque()

    assert leader.bus.disable_torque.call_args_list == [
        (("elbow_flex",), {"num_retry": TORQUE_COMM_RETRIES}),
        (("shoulder_pan",), {"num_retry": TORQUE_COMM_RETRIES}),
    ]


def test_inference_config_types():
    from lerobot.rollout import RTCInferenceConfig, SyncInferenceConfig

    assert SyncInferenceConfig().type == "sync"

    rtc = RTCInferenceConfig()
    assert rtc.type == "rtc"
    assert rtc.queue_threshold == 30
    assert rtc.rtc is not None


def test_sentry_config_defaults():
    from lerobot.rollout import SentryStrategyConfig

    cfg = SentryStrategyConfig()
    assert cfg.upload_every_n_episodes == 5
    assert cfg.target_video_file_size_mb is None


# ---------------------------------------------------------------------------
# RolloutRingBuffer
# ---------------------------------------------------------------------------


def test_ring_buffer_append_and_eviction():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=0.5, max_memory_mb=100.0, fps=10.0)
    # max_frames = 5
    for i in range(8):
        buf.append({"val": i})
    assert len(buf) == 5


def test_ring_buffer_drain():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=1.0, max_memory_mb=100.0, fps=10.0)
    for i in range(3):
        buf.append({"val": i})
    frames = buf.drain()
    assert len(frames) == 3
    assert len(buf) == 0
    assert buf.estimated_bytes == 0


def test_ring_buffer_clear():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=1.0, max_memory_mb=100.0, fps=10.0)
    buf.append({"val": 1})
    buf.clear()
    assert len(buf) == 0
    assert buf.estimated_bytes == 0


def test_ring_buffer_tensor_bytes():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=1.0, max_memory_mb=100.0, fps=10.0)
    t = torch.zeros(100, dtype=torch.float32)  # 400 bytes
    buf.append({"tensor": t})
    assert buf.estimated_bytes >= 400


# ---------------------------------------------------------------------------
# ThreadSafeRobot
# ---------------------------------------------------------------------------


def test_thread_safe_robot_delegates():
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig

    robot = MockRobot(MockRobotConfig(n_motors=3))
    robot.connect()
    wrapper = ThreadSafeRobot(robot)

    obs = wrapper.get_observation()
    assert "motor_1.pos" in obs
    assert "motor_2.pos" in obs
    assert "motor_3.pos" in obs

    action = {"motor_1.pos": 0.0, "motor_2.pos": 1.0, "motor_3.pos": 2.0}
    result = wrapper.send_action(action)
    assert result == action

    robot.disconnect()


def test_thread_safe_robot_properties():
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig

    robot = MockRobot(MockRobotConfig(n_motors=3))
    robot.connect()
    wrapper = ThreadSafeRobot(robot)

    assert wrapper.name == "mock_robot"
    assert "motor_1.pos" in wrapper.observation_features
    assert "motor_1.pos" in wrapper.action_features
    assert wrapper.is_connected is True
    assert wrapper.inner is robot

    robot.disconnect()


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------


def test_create_strategy_dispatches():
    from lerobot.rollout import (
        BaseStrategy,
        BaseStrategyConfig,
        DAggerStrategy,
        DAggerStrategyConfig,
        EpisodicStrategy,
        EpisodicStrategyConfig,
        SentryStrategy,
        SentryStrategyConfig,
        create_strategy,
    )

    assert isinstance(create_strategy(BaseStrategyConfig()), BaseStrategy)
    assert isinstance(create_strategy(SentryStrategyConfig()), SentryStrategy)
    assert isinstance(create_strategy(DAggerStrategyConfig()), DAggerStrategy)
    assert isinstance(create_strategy(EpisodicStrategyConfig()), EpisodicStrategy)


def test_create_strategy_unknown_raises():
    from lerobot.rollout import create_strategy

    cfg = MagicMock()
    cfg.type = "bogus"
    with pytest.raises(ValueError, match="Unknown strategy type"):
        create_strategy(cfg)


# ---------------------------------------------------------------------------
# Inference factory
# ---------------------------------------------------------------------------


def test_create_inference_engine_sync():
    from lerobot.rollout import SyncInferenceConfig, SyncInferenceEngine, create_inference_engine

    engine = create_inference_engine(
        SyncInferenceConfig(),
        policy=MagicMock(),
        preprocessor=MagicMock(),
        postprocessor=MagicMock(),
        robot_wrapper=MagicMock(robot_type="mock"),
        hw_features={},
        dataset_features={},
        ordered_action_keys=["k"],
        task="test",
        fps=30.0,
        device="cpu",
    )
    assert isinstance(engine, SyncInferenceEngine)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def test_estimate_max_episode_seconds_no_video():
    from lerobot.rollout.strategies import estimate_max_episode_seconds

    assert estimate_max_episode_seconds({}, fps=30.0) == 300.0


def test_estimate_max_episode_seconds_with_video():
    from lerobot.rollout.strategies import estimate_max_episode_seconds

    features = {"cam": {"dtype": "video", "shape": (480, 640, 3)}}
    result = estimate_max_episode_seconds(features, fps=30.0)
    assert result > 0
    # With a real camera, duration should differ from the fallback
    assert result != 300.0


def test_safe_push_to_hub():
    from lerobot.rollout.strategies import safe_push_to_hub

    ds = MagicMock()
    ds.num_episodes = 0
    assert safe_push_to_hub(ds) is False
    ds.push_to_hub.assert_not_called()

    ds.num_episodes = 5
    assert safe_push_to_hub(ds, tags=["test"]) is True
    ds.push_to_hub.assert_called_once_with(tags=["test"], private=False)


# ---------------------------------------------------------------------------
# DAgger state machine
# ---------------------------------------------------------------------------


def test_dagger_full_transition_cycle():
    from lerobot.rollout.strategies import DAggerEvents, DAggerPhase

    events = DAggerEvents()
    assert events.phase == DAggerPhase.AUTONOMOUS

    # AUTONOMOUS -> PAUSED
    events.request_transition("pause_resume")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.AUTONOMOUS, DAggerPhase.PAUSED)

    # PAUSED -> CORRECTING
    events.request_transition("pause_resume")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.PAUSED, DAggerPhase.CORRECTING)

    # CORRECTING -> AUTONOMOUS
    events.request_transition("pause_resume")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.CORRECTING, DAggerPhase.AUTONOMOUS)


def test_dagger_invalid_transition_ignored():
    from lerobot.rollout.strategies import DAggerEvents, DAggerPhase

    events = DAggerEvents()
    events.request_transition("correction")  # Not valid from AUTONOMOUS
    assert events.consume_transition() is None
    assert events.phase == DAggerPhase.AUTONOMOUS


def test_rerun_shutdown_does_not_block_process_exit():
    import rerun as rr

    from lerobot.utils import rerun_visualization

    release = threading.Event()

    with (
        patch.object(rr, "unregister_shutdown") as unregister_shutdown,
        patch.object(rr, "rerun_shutdown", side_effect=lambda: release.wait(timeout=1.0)),
        patch.object(rerun_visualization, "RERUN_SHUTDOWN_TIMEOUT_S", 0.01),
    ):
        start = time.perf_counter()
        rerun_visualization.shutdown_rerun()
        elapsed = time.perf_counter() - start
        release.set()

    unregister_shutdown.assert_called_once_with()
    assert elapsed < 0.2


def test_dagger_escape_requests_global_shutdown():
    from lerobot.rollout import DAggerKeyboardConfig
    from lerobot.rollout.strategies import DAggerEvents
    from lerobot.rollout.strategies.dagger import _init_dagger_keyboard

    events = DAggerEvents()
    shutdown_event = threading.Event()

    with patch("lerobot.rollout.strategies.dagger.create_key_listener") as create_listener:
        _init_dagger_keyboard(events, DAggerKeyboardConfig(), shutdown_event)
        dispatch = create_listener.call_args.args[0]
        dispatch("esc")

    assert events.stop_recording.is_set()
    assert shutdown_event.is_set()


def test_dagger_events_reset():
    from lerobot.rollout.strategies import DAggerEvents, DAggerPhase

    events = DAggerEvents()
    events.request_transition("pause_resume")
    events.consume_transition()  # -> PAUSED
    events.upload_requested.set()
    events.reset()
    assert events.phase == DAggerPhase.AUTONOMOUS
    assert not events.upload_requested.is_set()


# ---------------------------------------------------------------------------
# Context dataclass
# ---------------------------------------------------------------------------


def test_rollout_context_fields():
    from lerobot.rollout import RolloutContext

    field_names = {f.name for f in dataclasses.fields(RolloutContext)}
    assert field_names == {"runtime", "hardware", "policy", "processors", "data"}
