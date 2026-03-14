import ctypes
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Optional

import cv2 as cv
import numpy as np
import torch
from torch import nn

sys.path.insert(0, "./vision")
from bejeweled_env import BejeweledEnv, RewardConfig, TransitionConfig, ScoreConfig
from dqn import DQN, DQNConfig, ReplayBuffer


ProgressCallback = Callable[[Dict[str, float]], None]


@dataclass
class TrainingConfig:
    window_title: str
    episodes: int = 50
    max_steps: int = 200
    model_out: str = "models/bejeweled_dqn.pt"
    device: str = "cpu"
    reward_log_path: str = "training_rewards.csv"
    plot_path: str = "training_rewards.png"
    log_steps: bool = False
    debug_view: bool = False
    poll_hotkeys: bool = True
    classifier_path: str = "models/gem_classifier.pt"
    classifier_device: str = "cpu"
    transition_enabled: bool = True
    transition_pause_seconds: float = 10.0
    transition_confidence_threshold: float = 0.58
    transition_motion_threshold: float = 0.11
    transition_consecutive_frames: int = 4
    score_enabled: bool = True
    score_calibration_path: str = "score_calibration.json"
    score_templates_dir: str = "score_digits"
    score_match_threshold: float = 0.7
    score_stable_frames: int = 4
    score_stable_threshold: float = 2.0
    score_capture_interval: float = 0.1
    score_reward_scale: float = 1.0
    score_max_wait_seconds: float = 3.0
    score_debug_print: bool = False


class TrainingControl:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._pause = threading.Event()

    def request_stop(self) -> None:
        self._stop.set()

    def request_pause(self) -> None:
        self._pause.set()

    def request_resume(self) -> None:
        self._pause.clear()

    def stopped(self) -> bool:
        return self._stop.is_set()

    def paused(self) -> bool:
        return self._pause.is_set()

    def wait_if_paused(self) -> None:
        while self._pause.is_set() and not self._stop.is_set():
            time.sleep(0.05)


def select_action(q_net: DQN, state: np.ndarray, epsilon: float, n_actions: int, device: str) -> int:
    if np.random.rand() < epsilon:
        return np.random.randint(0, n_actions)
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = q_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())


def optimize(
    q_net: DQN,
    target_net: DQN,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    cfg: DQNConfig,
    device: str,
) -> float:
    states, actions, rewards, next_states, dones = replay.sample(cfg.batch_size)
    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
    next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones_t = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

    q_values = q_net(states_t).gather(1, actions_t)
    with torch.no_grad():
        next_q = target_net(next_states_t).max(1, keepdim=True)[0]
        target = rewards_t + cfg.gamma * next_q * (1.0 - dones_t)

    loss = nn.functional.smooth_l1_loss(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def save_checkpoint(
    checkpoint_path: str,
    q_net: DQN,
    target_net: DQN,
    optimizer: torch.optim.Optimizer,
    epsilon: float,
    total_steps: int,
    reward_history: list,
    cfg: TrainingConfig,
    dqn_cfg: DQNConfig,
    reward_cfg: RewardConfig,
) -> None:
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    payload = {
        "model_state_dict": q_net.state_dict(),
        "target_state_dict": target_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epsilon": epsilon,
        "total_steps": total_steps,
        "reward_history": reward_history,
        "training_config": asdict(cfg),
        "dqn_config": asdict(dqn_cfg),
        "reward_config": asdict(reward_cfg),
    }
    torch.save(payload, checkpoint_path)


def load_checkpoint(checkpoint_path: str, device: str = "cpu") -> dict:
    return torch.load(checkpoint_path, map_location=device)


def train_agent(
    window_title: str,
    episodes: int,
    max_steps: int,
    model_out: str,
    device: str = "cpu",
    reward_cfg: RewardConfig = RewardConfig(),
    dqn_cfg: DQNConfig = DQNConfig(),
    log_steps: bool = False,
    debug_view: bool = False,
    reward_log_path: str = "training_rewards.csv",
    plot_path: str = "training_rewards.png",
    classifier_path: str = "models/gem_classifier.pt",
    classifier_device: str = "cpu",
):
    cfg = TrainingConfig(
        window_title=window_title,
        episodes=episodes,
        max_steps=max_steps,
        model_out=model_out,
        device=device,
        log_steps=log_steps,
        debug_view=debug_view,
        reward_log_path=reward_log_path,
        plot_path=plot_path,
        poll_hotkeys=True,
        classifier_path=classifier_path,
        classifier_device=classifier_device,
    )
    return train_session(cfg=cfg, reward_cfg=reward_cfg, dqn_cfg=dqn_cfg)


def train_session(
    cfg: TrainingConfig,
    reward_cfg: RewardConfig = RewardConfig(),
    dqn_cfg: DQNConfig = DQNConfig(),
    control: Optional[TrainingControl] = None,
    progress_cb: Optional[ProgressCallback] = None,
    checkpoint_in: Optional[str] = None,
    save_checkpoint_path: Optional[str] = None,
) -> dict:
    env = BejeweledEnv(
        window_title=cfg.window_title,
        reward_cfg=reward_cfg,
        classifier_path=cfg.classifier_path,
        classifier_device=cfg.classifier_device,
        transition_cfg=TransitionConfig(
            enabled=cfg.transition_enabled,
            pause_seconds=cfg.transition_pause_seconds,
            confidence_threshold=cfg.transition_confidence_threshold,
            motion_threshold=cfg.transition_motion_threshold,
            consecutive_frames=cfg.transition_consecutive_frames,
        ),
        score_cfg=ScoreConfig(
            enabled=cfg.score_enabled,
            calibration_path=cfg.score_calibration_path,
            templates_dir=cfg.score_templates_dir,
            match_threshold=cfg.score_match_threshold,
            stable_frames=cfg.score_stable_frames,
            stable_threshold=cfg.score_stable_threshold,
            capture_interval=cfg.score_capture_interval,
            reward_scale=cfg.score_reward_scale,
            max_wait_seconds=cfg.score_max_wait_seconds,
            debug_print=cfg.score_debug_print,
        ),
    )
    obs = env.reset()

    q_net = DQN(obs.shape, env.action_count).to(cfg.device)
    target_net = DQN(obs.shape, env.action_count).to(cfg.device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(q_net.parameters(), lr=dqn_cfg.lr)
    replay = ReplayBuffer(dqn_cfg.replay_size)

    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.995
    total_steps = 0
    reward_history = []
    completed_episodes = 0
    last_loss = 0.0

    if checkpoint_in:
        checkpoint = load_checkpoint(checkpoint_in, cfg.device)
        q_net.load_state_dict(checkpoint["model_state_dict"])
        target_net.load_state_dict(checkpoint["target_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epsilon = float(checkpoint.get("epsilon", epsilon))
        total_steps = int(checkpoint.get("total_steps", total_steps))
        reward_history = list(checkpoint.get("reward_history", reward_history))

    get_async_key_state = None
    if cfg.poll_hotkeys:
        get_async_key_state = ctypes.windll.user32.GetAsyncKeyState

    for ep in range(1, cfg.episodes + 1):
        if control and control.stopped():
            break
        state = env.reset()
        ep_reward = 0.0
        ep_loss = 0.0
        step = 0
        attempts = 0
        max_attempts = cfg.max_steps * 10
        while step < cfg.max_steps and attempts < max_attempts:
            attempts += 1
            if control:
                control.wait_if_paused()
                if control.stopped():
                    break

            if get_async_key_state and get_async_key_state(ord("Q")) & 0x8000:
                if control:
                    control.request_stop()
                break
            if get_async_key_state and get_async_key_state(ord("P")) & 0x8000 and control:
                if control.paused():
                    control.request_resume()
                else:
                    control.request_pause()
                time.sleep(0.2)

            action = select_action(q_net, state, epsilon, env.action_count, cfg.device)
            prev_state = state
            next_state, reward, done, info = env.step(action)
            state = next_state
            skip_replay = bool(info.get("skip_replay", False))
            if not skip_replay:
                step += 1
                replay.push(prev_state, action, reward, next_state, done)
                ep_reward += reward
                total_steps += 1

                if len(replay) >= dqn_cfg.min_replay:
                    ep_loss = optimize(q_net, target_net, optimizer, replay, dqn_cfg, cfg.device)
                    last_loss = ep_loss

                if total_steps % dqn_cfg.target_update == 0:
                    target_net.load_state_dict(q_net.state_dict())
            else:
                time.sleep(0.05)

            if cfg.log_steps:
                print(f"ep={ep} step={step} reward={reward:.3f} total={ep_reward:.3f} info={info}")

            if cfg.debug_view:
                frame = env.debug_frame()
                cv.imshow("Bejeweled Debug", frame)
                cv.waitKey(1)

            if progress_cb:
                progress_cb(
                    {
                        "episode": float(ep),
                        "step": float(step),
                        "episode_reward": float(ep_reward),
                        "last_reward": float(reward),
                        "loss": float(ep_loss),
                        "epsilon": float(epsilon),
                        "total_steps": float(total_steps),
                        "match_count": float(info.get("match_count", 0)),
                        "invalid": float(1 if info.get("invalid") else 0),
                        "transition": float(1 if info.get("transition") else 0),
                        "score_diff": float(info.get("score_diff", 0)),
                    }
                )

            if done or (control and control.stopped()):
                break

        if control and control.stopped():
            break

        reward_history.append(ep_reward)
        completed_episodes += 1
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(
            f"Episode {ep}/{cfg.episodes} | reward={ep_reward:.2f} | loss={ep_loss:.4f} | eps={epsilon:.3f}"
        )

    if cfg.debug_view:
        cv.destroyAllWindows()

    os.makedirs(os.path.dirname(cfg.model_out) or ".", exist_ok=True)
    torch.save(q_net.state_dict(), cfg.model_out)
    _write_reward_log(cfg.reward_log_path, reward_history)
    _try_plot_rewards(reward_history, cfg.plot_path)

    if save_checkpoint_path:
        save_checkpoint(
            checkpoint_path=save_checkpoint_path,
            q_net=q_net,
            target_net=target_net,
            optimizer=optimizer,
            epsilon=epsilon,
            total_steps=total_steps,
            reward_history=reward_history,
            cfg=cfg,
            dqn_cfg=dqn_cfg,
            reward_cfg=reward_cfg,
        )

    return {
        "run_mode": "train",
        "model_out": cfg.model_out,
        "reward_log_path": cfg.reward_log_path,
        "plot_path": cfg.plot_path,
        "episodes_completed": completed_episodes,
        "reward_history": reward_history,
        "epsilon": epsilon,
        "total_steps": total_steps,
        "loss": last_loss,
    }


def play_session(
    cfg: TrainingConfig,
    model_in: str,
    reward_cfg: RewardConfig = RewardConfig(),
    control: Optional[TrainingControl] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> dict:
    env = BejeweledEnv(
        window_title=cfg.window_title,
        reward_cfg=reward_cfg,
        classifier_path=cfg.classifier_path,
        classifier_device=cfg.classifier_device,
        transition_cfg=TransitionConfig(
            enabled=cfg.transition_enabled,
            pause_seconds=cfg.transition_pause_seconds,
            confidence_threshold=cfg.transition_confidence_threshold,
            motion_threshold=cfg.transition_motion_threshold,
            consecutive_frames=cfg.transition_consecutive_frames,
        ),
        score_cfg=ScoreConfig(
            enabled=cfg.score_enabled,
            calibration_path=cfg.score_calibration_path,
            templates_dir=cfg.score_templates_dir,
            match_threshold=cfg.score_match_threshold,
            stable_frames=cfg.score_stable_frames,
            stable_threshold=cfg.score_stable_threshold,
            capture_interval=cfg.score_capture_interval,
            reward_scale=cfg.score_reward_scale,
            max_wait_seconds=cfg.score_max_wait_seconds,
            debug_print=cfg.score_debug_print,
        ),
    )
    obs = env.reset()
    q_net = DQN(obs.shape, env.action_count).to(cfg.device)

    payload = torch.load(model_in, map_location=cfg.device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        q_net.load_state_dict(payload["model_state_dict"])
    else:
        q_net.load_state_dict(payload)
    q_net.eval()

    get_async_key_state = None
    if cfg.poll_hotkeys:
        get_async_key_state = ctypes.windll.user32.GetAsyncKeyState

    reward_history = []
    total_steps = 0
    for ep in range(1, cfg.episodes + 1):
        if control and control.stopped():
            break
        state = env.reset()
        ep_reward = 0.0
        step = 0
        attempts = 0
        max_attempts = cfg.max_steps * 10
        while step < cfg.max_steps and attempts < max_attempts:
            attempts += 1
            if control:
                control.wait_if_paused()
                if control.stopped():
                    break

            if get_async_key_state and get_async_key_state(ord("Q")) & 0x8000:
                if control:
                    control.request_stop()
                break
            if get_async_key_state and get_async_key_state(ord("P")) & 0x8000 and control:
                if control.paused():
                    control.request_resume()
                else:
                    control.request_pause()
                time.sleep(0.2)

            action = select_action(q_net, state, epsilon=0.0, n_actions=env.action_count, device=cfg.device)
            next_state, reward, done, info = env.step(action)
            state = next_state

            if not info.get("skip_replay", False):
                step += 1
                ep_reward += reward
                total_steps += 1
            else:
                time.sleep(0.05)

            if progress_cb:
                progress_cb(
                    {
                        "episode": float(ep),
                        "step": float(step),
                        "episode_reward": float(ep_reward),
                        "last_reward": float(reward),
                        "loss": 0.0,
                        "epsilon": 0.0,
                        "total_steps": float(total_steps),
                        "match_count": float(info.get("match_count", 0)),
                        "invalid": float(1 if info.get("invalid") else 0),
                        "transition": float(1 if info.get("transition") else 0),
                        "score_diff": float(info.get("score_diff", 0)),
                    }
                )

            if done or (control and control.stopped()):
                break

        if control and control.stopped():
            break
        reward_history.append(ep_reward)
        print(f"Play Episode {ep}/{cfg.episodes} | reward={ep_reward:.2f}")

    return {
        "run_mode": "play",
        "model_out": model_in,
        "reward_log_path": "",
        "plot_path": "",
        "episodes_completed": len(reward_history),
        "reward_history": reward_history,
        "epsilon": 0.0,
        "total_steps": total_steps,
        "loss": 0.0,
    }


def _write_reward_log(path: str, rewards: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("episode,reward\n")
        for i, r in enumerate(rewards, start=1):
            f.write(f"{i},{r}\n")


def _try_plot_rewards(rewards: list, path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(rewards)
    plt.title("Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
