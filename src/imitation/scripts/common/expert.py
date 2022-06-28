import sacred

from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO
from stable_baselines3.common import policies

from imitation.policies import base

from imitation.scripts.common.common import common_ingredient, make_venv

expert_ingredient = sacred.Ingredient("expert", ingredients=[common_ingredient])


@expert_ingredient.config
def config():
    policy_type = "ppo"  # right now only ppo supported
    policy_path = None  # path to zip-file containing an expert policy
    huggingface_repo_id = None
    hugginface_orga = "ernestumorga"  # TODO(ernestum): change to official orga

    locals()  # quieten flake8


@expert_ingredient.capture
def get_huggingface_repo_id(common, huggingface_repo_id, policy_type, hugginface_orga):
    if huggingface_repo_id is not None:
        return huggingface_repo_id
    else:
        # TODO(ernestum): use naming scheme tools from the future here
        return f"{hugginface_orga}/{policy_type}-{common['env_name'].replace('/', '-')}"


@expert_ingredient.capture
def get_expert_path(common, policy_path, policy_type):
    if policy_path is not None:
        return policy_path
    else:
        # TODO(ernestum): use naming scheme tools from the future here
        model_filename = f"{policy_type}-{common['env_name'].replace('/', '-')}.zip"
        return load_from_hub(get_huggingface_repo_id(), model_filename)


@expert_ingredient.capture
def get_expert_policy(policy_type) -> policies.BasePolicy:
    env = make_venv()
    if policy_type == "ppo":
        return PPO.load(get_expert_path(), env)
    elif policy_type == "random":
        base.RandomPolicy(env.observation_space, env.action_space)
    elif policy_type == "zero":
        base.ZeroPolicy(env.observation_space, env.action_space)
