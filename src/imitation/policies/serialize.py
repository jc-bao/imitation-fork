"""Load serialized policies of different types."""

# FIXME(sam): it seems like this module could mostly be replaced with a few
# torch.load() and torch.save() calls

import logging
import os
import os.path as osp
import pathlib
from typing import Callable, Type, TypeVar, Mapping, Any

from stable_baselines3.common import base_class, callbacks, policies, vec_env

from imitation.policies import base
from imitation.util import registry, video_wrapper
from imitation.data import rollout

Algorithm = TypeVar("Algorithm", bound=base_class.BaseAlgorithm)

PolicyLoaderFn = Callable[[str, vec_env.VecEnv], policies.BasePolicy]

policy_registry: registry.Registry[PolicyLoaderFn] = registry.Registry()


def load_stable_baselines_model(
    cls: Type[Algorithm],
    path: str,
    venv: vec_env.VecEnv,
    **kwargs,
) -> Algorithm:
    """Helper method to load RL models from Stable Baselines.

    Args:
        cls: Stable Baselines RL algorithm.
        path: Path to directory containing saved model data.
        venv: Environment to train on.
        kwargs: Passed through to `cls.load`.

    Raises:
        FileNotFoundError: If `path` is not a directory containing a `model.zip` file.
        FileExistsError: If `path` contains a `vec_normalize.pkl` file (unsupported).

    Returns:
        The deserialized RL algorithm.
    """
    logging.info(f"Loading Stable Baselines policy for '{cls}' from '{path}'")
    policy_dir = pathlib.Path(path)
    if not policy_dir.is_dir():
        raise FileNotFoundError(
            f"path={path} needs to be a directory containing model.zip.",
        )

    # SOMEDAY(adam): added 2022-01, can probably remove this check in 2023
    vec_normalize_path = policy_dir / "vec_normalize.pkl"
    if vec_normalize_path.exists():
        raise FileExistsError(
            "Outdated policy format: we do not support restoring normalization "
            "statistics from '{vec_normalize_path}'",
        )

    model_path = policy_dir / "model.zip"
    return cls.load(model_path, env=venv, **kwargs)


def _load_stable_baselines(
    cls: Type[base_class.BaseAlgorithm],
) -> PolicyLoaderFn:
    """Higher-order function, returning a policy loading function.

    Args:
        cls: The RL algorithm, e.g. `stable_baselines3.PPO`.

    Returns:
        A function loading policies trained via cls.
    """

    def f(path: str, venv: vec_env.VecEnv) -> policies.BasePolicy:
        """Loads a policy saved to path, for environment env."""
        model = load_stable_baselines_model(cls, path, venv)
        return getattr(model, "policy")

    return f


policy_registry.register(
    "random",
    value=registry.build_loader_fn_require_space(base.RandomPolicy),
)
policy_registry.register(
    "zero",
    value=registry.build_loader_fn_require_space(base.ZeroPolicy),
)


def _add_stable_baselines_policies(classes):
    for k, cls_name in classes.items():
        cls = registry.load_attr(cls_name)
        fn = _load_stable_baselines(cls)
        policy_registry.register(k, value=fn)


STABLE_BASELINES_CLASSES = {
    "ppo": "stable_baselines3:PPO",
    "sac": "stable_baselines3:SAC",
}
_add_stable_baselines_policies(STABLE_BASELINES_CLASSES)


def load_policy(
    policy_type: str,
    policy_path: str,
    venv: vec_env.VecEnv,
) -> policies.BasePolicy:
    """Load serialized policy.

    Args:
        policy_type: A key in `policy_registry`, e.g. `ppo`.
        policy_path: A path on disk where the policy is stored.
        venv: An environment that the policy is to be used with.

    Returns:
        The deserialized policy.
    """
    agent_loader = policy_registry.get(policy_type)
    return agent_loader(policy_path, venv)


def save_stable_model(
    output_dir: str,
    model: base_class.BaseAlgorithm,
) -> None:
    """Serialize Stable Baselines model.

    Load later with `load_policy(..., policy_path=output_dir)`.

    Args:
        output_dir: Path to the save directory.
        model: The stable baselines model.
    """
    # Save each model in new directory in case we want to add metadata or other
    # information in future. (E.g. we used to save `VecNormalize` statistics here,
    # although that is no longer necessary.)
    os.makedirs(output_dir, exist_ok=True)
    model.save(os.path.join(output_dir, "model.zip"))
    logging.info("Saved policy to %s", output_dir)


def record_n_save_video(
    output_dir: str,
    model: base_class.BaseAlgorithm,
    eval_venv: vec_env.VecEnv,
    video_kwargs: Mapping[str, Any],
) -> None:
    video_dir = osp.join(output_dir, "videos")
    video_venv = video_wrapper.VideoWrapper(
        eval_venv, directory=video_dir, **video_kwargs
    )
    sample_until = rollout.make_sample_until(min_timesteps=None, min_episodes=1)
    rollout.generate_trajectories(model.policy, video_venv, sample_until)
    assert "video.000000.mp4" in os.listdir(video_dir)
    video_path = osp.join(video_dir, "video.000000.mp4")
    model.logger.record("video", video_path)
    model.logger.log(f"Recording and saving video to {video_path} ...")


class SaveVideoCallback(callbacks.EventCallback):
    """Saves the policy using `save_n_record_video` each time it is called.

    Should be used in conjunction with `callbacks.EveryNTimesteps`
    or another event-based trigger.
    """

    def __init__(
        self,
        policy_dir: str,
        eval_venv: vec_env.VecEnv,
        video_kwargs: Mapping[str, Any],
        *args,
        **kwargs,
    ):
        """Builds SavePolicyCallback.

        Args:
            policy_dir: Directory to save checkpoints.
            eval_venv: Environment to evaluate the policy on.
            video_kwargs: Keyword arguments to pass to `video_wrapper.VideoWrapper`.
            *args: Passed through to `callbacks.EventCallback`.
            **kwargs: Passed through to `callbacks.EventCallback`.
        """
        super().__init__(*args, **kwargs)
        self.policy_dir = policy_dir
        self.eval_venv = eval_venv
        self.video_kwargs = video_kwargs

    def _on_step(self) -> bool:
        output_dir = os.path.join(self.policy_dir, f"{self.num_timesteps:012d}")
        record_n_save_video(output_dir, self.model, self.eval_venv, self.video_kwargs)

        return True


class SavePolicyCallback(callbacks.EventCallback):
    """Saves the policy using `save_stable_model` each time it is called.

    Should be used in conjunction with `callbacks.EveryNTimesteps`
    or another event-based trigger.
    """

    def __init__(
        self,
        policy_dir: str,
        *args,
        **kwargs,
    ):
        """Builds SavePolicyCallback.

        Args:
            policy_dir: Directory to save checkpoints.
            *args: Passed through to `callbacks.EventCallback`.
            **kwargs: Passed through to `callbacks.EventCallback`.
        """
        super().__init__(*args, **kwargs)
        self.policy_dir = policy_dir

    def _on_step(self) -> bool:
        output_dir = os.path.join(self.policy_dir, f"{self.num_timesteps:012d}")
        save_stable_model(output_dir, self.model)
        return True
