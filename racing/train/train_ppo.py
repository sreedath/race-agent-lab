"""PPO training for RaceTrack-v0 (student edition).

Designed to be driven from train_agent.ipynb: call `train(reward_fn, steps)`
and it returns the trained model. Interrupt-safe: pressing the notebook's
stop button (KeyboardInterrupt) keeps the partially trained in-memory model,
which you can export immediately.

Also runnable from a shell:
  python -m racing.train.train_ppo --steps 100000
(uses the sample reward function below).
"""

import argparse
import os
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from racing.env.race_env import RaceEnv

ROOT = Path(__file__).resolve().parent.parent.parent
RUNS = ROOT / "runs"
NUM_ENVS = max(2, os.cpu_count() or 2)
CHECKPOINT_EVERY = 25_000  # steps between saved checkpoints


def sample_reward(sig) -> float:
    """A reasonable starting reward: progress minus crash penalties.

    Feel free to study it, but the point of the exercise is to write your
    own in the notebook.
    """
    reward = 1.0 * sig.delta_s
    reward -= 0.02
    if sig.wall_contact:
        reward -= 0.5
    if sig.new_wall_hit:
        reward -= 3.0
    if sig.terminated:
        reward -= 30.0
    return reward


def make_env(reward_fn, rank: int):
    def _init():
        env = RaceEnv(reward_fn=reward_fn)
        env.reset(seed=1000 + rank)
        return env

    return _init


def linear_schedule(initial: float):
    def fn(progress_remaining: float) -> float:
        return initial * progress_remaining

    return fn


def train(reward_fn, steps: int, seed: int = 0, run_name: str = "agent") -> PPO:
    """Train PPO with YOUR reward function; returns the (possibly partial)
    model even if interrupted with the stop button."""
    # Fail fast on a broken reward function before spawning workers.
    probe = RaceEnv(reward_fn=reward_fn)
    probe.reset(seed=0)
    probe.step(probe.action_space.sample())

    torch.set_num_threads(2)
    RUNS.mkdir(exist_ok=True)
    vec = VecMonitor(
        SubprocVecEnv([make_env(reward_fn, i) for i in range(NUM_ENVS)])
    )
    model = PPO(
        "MlpPolicy",
        vec,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
            activation_fn=torch.nn.Tanh,
        ),
        n_steps=1024,
        batch_size=2048,
        n_epochs=10,
        learning_rate=linear_schedule(3e-4),
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.003,
        seed=seed,
        device="cpu",
        verbose=1,
    )
    checkpoint = CheckpointCallback(
        save_freq=max(CHECKPOINT_EVERY // NUM_ENVS, 1),
        save_path=str(RUNS / "checkpoints"),
        name_prefix=run_name,
    )
    try:
        model.learn(
            total_timesteps=steps, callback=checkpoint, progress_bar=False
        )
    except KeyboardInterrupt:
        print(
            "\ntraining interrupted: keeping the model trained so far "
            f"(~{model.num_timesteps} steps). You can export it now."
        )
    finally:
        vec.close()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=str(RUNS / "ppo_race"))
    args = parser.parse_args()
    model = train(sample_reward, args.steps, seed=args.seed)
    model.save(args.out)
    print(f"saved model to {args.out}.zip")


if __name__ == "__main__":
    main()
