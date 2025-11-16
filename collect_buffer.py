import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch as th
import yaml
from huggingface_sb3 import EnvironmentName
from stable_baselines3.common.callbacks import tqdm
from stable_baselines3.common.utils import set_random_seed
from tqdm import trange

import rl_zoo3.import_envs  # noqa: F401 pylint: disable=unused-import
from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.load_from_hub import download_from_hub
from rl_zoo3.utils import StoreDict, get_model_path


def downsample_mean(
    X: Union[np.ndarray, np.memmap],
    chunk_size: int = 512,
    out: Optional[Union[np.ndarray, np.memmap]] = None,
    dtype=np.float32,
):
    """
    X: (B,84,84,4) array-like (np.ndarray or np.memmap). Columns are channels last.
    chunk_size: number of items per chunk along batch dimension.
    out: optional preallocated/memmap array of shape (B,42,42,4). If None, allocates.
    dtype: dtype for computation/output.

    Returns: out (np.ndarray or np.memmap) with shape (B,42,42,4)
    """
    assert X.ndim == 4 and X.shape[1:4] == (84, 84, 4), "Expected (B,84,84,4)"
    B = X.shape[0]

    if out is None:
        out = np.empty((B, 42, 42, 4), dtype=dtype)

    print("Downsampling Images")
    for start in trange(0, B, chunk_size):
        stop = min(start + chunk_size, B)
        xb = np.asarray(X[start:stop], dtype=dtype)  # (b,84,84,4)
        # 2x2 area/mean pooling without copies
        xb = xb.reshape(stop-start, 42, 2, 42, 2, 4).mean(axis=(2, 4))
        out[start:stop] = xb  # (b,42,42,4)

    return out


def _array_copy(value: Any) -> np.ndarray:
    """
    Create a numpy array copy of the provided value. Raises a ValueError
    when the value cannot be converted into a numpy array.
    """
    try:
        return np.asarray(value).copy()
    except Exception as exc:
        raise ValueError(
            "Could not convert collected data to a numpy array. "
            "This script currently supports environments with numpy-compatible observations and actions."
        ) from exc


def _prepare_output_paths(buffer_path: str, metadata_path: Optional[str]) -> Tuple[Path, Path]:
    buffer_path = Path(buffer_path).expanduser().resolve()
    if metadata_path is not None:
        meta_path = Path(metadata_path).expanduser().resolve()
    else:
        meta_path = buffer_path.with_suffix(".json")

    if buffer_path.parent != Path.cwd():
        buffer_path.parent.mkdir(parents=True, exist_ok=True)
    if meta_path.parent != Path.cwd():
        meta_path.parent.mkdir(parents=True, exist_ok=True)
    return buffer_path, meta_path


def collect_buffer() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(description="Collect a replay-style buffer from a trained RL Zoo agent.")
    parser.add_argument("--env", help="environment ID", type=EnvironmentName, default="CartPole-v1")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("--buffer-size", help="Number of transitions to gather", default=2048, type=int)
    parser.add_argument("--buffer-path", help="Where to store the collected buffer (.npz)", default="buffer.npz", type=str)
    parser.add_argument(
        "--metadata-path",
        help="Optional metadata path (defaults to buffer_path with .json extension)",
        default="metadata.json",
        type=str,
    )
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument("--device", help="PyTorch device to use (ex: cpu, cuda...)", default="auto", type=str)
    parser.add_argument(
        "--load-best", action="store_true", default=False, help="Load best model instead of last model if available"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=int,
        help="Load checkpoint instead of last model if available, "
        "you must pass the number of timesteps corresponding to it",
    )
    parser.add_argument(
        "--load-last-checkpoint",
        action="store_true",
        default=False,
        help="Load last checkpoint instead of last model if available",
    )
    parser.add_argument(
        "--compress-obs",
        action="store_true",
        default=False,
        help="Do we compress our observations?",
    )
    parser.add_argument(
        "--norm-reward", action="store_true", default=False, help="Normalize reward if applicable (trained with VecNormalize)"
    )
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=[],
        help="Additional external Gym environment package modules to import",
    )
    parser.add_argument(
        "--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor"
    )
    parser.add_argument(
        "--custom-objects", action="store_true", default=False, help="Use custom objects to solve loading issues"
    )
    parser.add_argument(
        "-P",
        "--progress",
        action="store_true",
        default=False,
        help="if toggled, display a progress bar using tqdm and rich",
    )
    args = parser.parse_args()

    if args.buffer_size <= 0:
        raise ValueError("buffer-size must be strictly positive.")

    # Going through custom gym packages to let them register in the global registry
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_name: EnvironmentName = args.env
    algo = args.algo
    folder = args.folder

    try:
        _, model_path, log_path = get_model_path(
            args.exp_id,
            folder,
            algo,
            env_name,
            args.load_best,
            args.load_checkpoint,
            args.load_last_checkpoint,
        )
    except (AssertionError, ValueError) as err:
        if "rl-trained-agents" not in folder:
            raise err
        print("Pretrained model not found locally, trying to download it from sb3 Huggingface hub: https://huggingface.co/sb3")
        download_from_hub(
            algo=algo,
            env_name=env_name,
            exp_id=args.exp_id,
            folder=folder,
            organization="sb3",
            repo_name=None,
            force=False,
        )
        _, model_path, log_path = get_model_path(
            args.exp_id,
            folder,
            algo,
            env_name,
            args.load_best,
            args.load_checkpoint,
            args.load_last_checkpoint,
        )

    if args.verbose > 0:
        print(f"Loading {model_path}")

    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

    set_random_seed(args.seed)

    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    is_atari = ExperimentManager.is_atari(env_name.gym_id)
    is_minigrid = ExperimentManager.is_minigrid(env_name.gym_id)

    stats_path = os.path.join(log_path, env_name)
    hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    env_kwargs: Dict[str, Any] = {}
    args_path = os.path.join(log_path, env_name, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path) as file:
            loaded_args = yaml.load(file, Loader=yaml.UnsafeLoader)
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    env = create_test_env(
        env_name.gym_id,
        n_envs=args.n_envs,
        stats_path=maybe_stats_path,
        seed=args.seed,
        should_render=False,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
        vec_env_cls=ExperimentManager.default_vec_env_cls,
    )

    model_kwargs: Dict[str, Any] = dict(seed=args.seed)
    if algo in off_policy_algos:
        model_kwargs.update(dict(buffer_size=1))
        if "optimize_memory_usage" in hyperparams:
            model_kwargs.update(optimize_memory_usage=False)

    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects: Dict[str, Any] = {}
    if newer_python_version or args.custom_objects:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    if "HerReplayBuffer" in hyperparams.get("replay_buffer_class", ""):
        model_kwargs["env"] = env

    model = ALGOS[algo].load(model_path, custom_objects=custom_objects, device=args.device, **model_kwargs)

    obs = env.reset()

    stochastic = (is_atari or is_minigrid) and not args.deterministic
    deterministic = not stochastic if args.deterministic is False else True

    episode_reward = np.zeros(env.num_envs, dtype=np.float64)
    episode_length = np.zeros(env.num_envs, dtype=np.int64)
    collected_returns: List[float] = []
    collected_lengths: List[int] = []
    successes: List[bool] = []

    observations: List[np.ndarray] = []
    next_observations: List[np.ndarray] = []
    actions: List[Any] = []
    rewards: List[float] = []
    dones: List[bool] = []
    episode_starts: List[bool] = []

    lstm_states = None
    episode_start = np.ones((env.num_envs,), dtype=bool)
    n_collected = 0

    progress_bar = None
    if args.progress:
        if tqdm is None:
            raise ImportError("Please install tqdm and rich to use the progress bar")
        progress_bar = tqdm(total=args.buffer_size, desc="Collecting buffer")

    try:
        while n_collected < args.buffer_size:
            current_episode_start = episode_start.copy()
            action, lstm_states = model.predict(
                obs,  # type: ignore[arg-type]
                state=lstm_states,
                episode_start=current_episode_start,
                deterministic=deterministic,
            )
            next_obs, reward, done, infos = env.step(action)

            should_stop = False
            for env_idx in range(env.num_envs):
                observations.append(_array_copy(obs[env_idx]))
                next_observations.append(_array_copy(next_obs[env_idx]))

                action_value = action[env_idx]
                actions.append(_array_copy(action_value) if not np.isscalar(action_value) else action_value.item() if isinstance(action_value, np.generic) else action_value)  # type: ignore[arg-type]
                reward_value = float(np.asarray(reward[env_idx]))
                rewards.append(reward_value)
                done_value = bool(done[env_idx])
                dones.append(done_value)
                episode_starts.append(bool(current_episode_start[env_idx]))

                episode_reward[env_idx] += reward_value
                episode_length[env_idx] += 1

                if done_value:
                    collected_returns.append(float(episode_reward[env_idx]))
                    collected_lengths.append(int(episode_length[env_idx]))
                    episode_reward[env_idx] = 0.0
                    episode_length[env_idx] = 0
                    if infos[env_idx].get("is_success") is not None:
                        successes.append(infos[env_idx].get("is_success", False))

                n_collected += 1
                if progress_bar is not None:
                    progress_bar.update(1)

                if n_collected >= args.buffer_size:
                    should_stop = True
                    break

            episode_start = done
            obs = next_obs

            if should_stop:
                break

    except KeyboardInterrupt:
        if args.verbose > 0:
            print("Interrupted by user, saving collected samples so far.")
    finally:
        if progress_bar is not None:
            progress_bar.close()

    buffer_dir = f"buffers/{args.env}-{args.algo}-{args.buffer_size}"

    if args.compress_obs:
        buffer_dir += "_compressed"

    bpath, mpath = f"{buffer_dir}/{args.buffer_path}", f"{buffer_dir}/{args.metadata_path}"
    buffer_path, metadata_path = _prepare_output_paths(bpath, mpath)

    def _stack_or_array(values: List[Any]) -> np.ndarray:
        if len(values) == 0:
            return np.array(values)
        first = values[0]
        if isinstance(first, np.ndarray):
            return np.stack(values, axis=0)
        return np.asarray(values)

    # calculate returns
    current_return = 0
    all_returns = []
    for reward, done in zip(reversed(rewards), reversed(dones)):
        current_return = reward + model.gamma * (1 - done) * current_return
        all_returns.append(current_return)
    all_returns = all_returns[::-1]

    # add run statistics
    all_values, all_log_probs, all_entropies = [], [], []
    chunks = int(np.floor(args.buffer_size / 32))
    chunked_observations, chunked_actions = np.array_split(np.stack(observations), chunks, axis=0), np.array_split(np.stack(actions), chunks, axis=0)
    prog_bar = tqdm(total=chunks, desc="Evaluating actions")
    for obs, a in zip(chunked_observations, chunked_actions):
        obs_for_policy, _ = model.policy.obs_to_tensor(obs)
        value, log_prob, entropy = model.policy.evaluate_actions(obs_for_policy.to(model.device), th.tensor(a).to(model.device))
        all_values.append(value.cpu().detach().numpy())
        all_log_probs.append(log_prob.cpu().detach().numpy())
        # all_entropies.append(entropy.cpu().detach().numpy())
        prog_bar.update(1)
    prog_bar.close()
    all_values = np.concatenate(all_values, axis=0)
    all_log_probs = np.concatenate(all_log_probs, axis=0)
    # all_entropies = np.concatenate(all_entropies, axis=0)

    stacked_obs = _stack_or_array(observations) 
    stacked_next_obs = _stack_or_array(next_observations)
    if args.compress_obs:
        stacked_obs = downsample_mean(stacked_obs, chunk_size=256)
        stacked_next_obs = downsample_mean(stacked_next_obs, chunk_size=256)

    buffer_data = {
        "observations": stacked_obs,
        "next_observations": stacked_next_obs,
        "actions": _stack_or_array(actions),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=bool),
        "episode_starts": np.asarray(episode_starts, dtype=bool),
        "estimated_values": all_values,
        "log_probs": all_log_probs,
        "returns": all_returns,
        # "entropies": all_entropies,
    }
    np.savez_compressed(buffer_path, **buffer_data)

    metadata: Dict[str, Any] = {
        "env_id": env_name.gym_id,
        "algo": algo,
        "deterministic": deterministic,
        "n_envs": args.n_envs,
        "buffer_size_requested": args.buffer_size,
        "buffer_size_collected": int(len(rewards)),
        "mean_reward": float(np.mean(rewards)) if len(rewards) > 0 else None,
        "mean_episode_return": float(np.mean(collected_returns)) if collected_returns else None,
        "std_episode_return": float(np.std(collected_returns)) if len(collected_returns) > 1 else None,
        "mean_episode_length": float(np.mean(collected_lengths)) if collected_lengths else None,
        "success_rate": float(np.mean(successes)) if successes else None,
        "model_path": str(model_path),
    }

    with metadata_path.open("w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, indent=2)

    if args.verbose > 0:
        print(f"Saved {len(rewards)} transitions to {buffer_path}")
        print(f"Metadata written to {metadata_path}")
        if collected_returns:
            print(
                f"Mean episode return: {metadata['mean_episode_return']:.2f} "
                f"+/- {metadata['std_episode_return'] or 0.0:.2f} "
                f"over {len(collected_returns)} episodes"
            )
        if metadata["success_rate"] is not None:
            print(f"Success rate: {metadata['success_rate'] * 100:.2f}%")

    env.close()


if __name__ == "__main__":
    collect_buffer()

