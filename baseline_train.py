"""Minimal configurable PPO baseline for HoverAviary / MultiHoverAviary.

`examples/learn.py` hardcodes a 1e7-step budget, which is impractical for
CPU-sized baseline runs. This script keeps only the training half of
learn.py (no replay, no plots) and exposes the step budget as an argument,
so baseline runs are reproducible and can be logged in BASELINE.md.

Example
-------

    $ python baseline_train.py --multiagent false --timesteps 200000
    $ python baseline_train.py --task formation --timesteps 400000

Notes
-----

Checkpoints and `evaluations.npz` land in `results/baseline-<date>/`; the
best model can be replayed afterwards with `examples/play.py`.
"""
import os
import argparse
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.envs.FormationAviary import FormationAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import str2bool

DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('one_d_rpm')
DEFAULT_AGENTS = 2


def run(task, multiagent, timesteps, output_folder='results'):
    filename = os.path.join(output_folder, 'baseline-' + datetime.now().strftime('%m.%d.%Y_%H.%M.%S'))
    os.makedirs(filename, exist_ok=True)

    if task == 'formation':
        train_env = make_vec_env(FormationAviary,
                                 env_kwargs=dict(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT),
                                 n_envs=1,
                                 seed=0
                                 )
        eval_env = FormationAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT)
        target_reward = 1050.
    elif not multiagent:
        train_env = make_vec_env(HoverAviary,
                                 env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT),
                                 n_envs=1,
                                 seed=0
                                 )
        eval_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
        target_reward = 474.
    else:
        train_env = make_vec_env(MultiHoverAviary,
                                 env_kwargs=dict(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT),
                                 n_envs=1,
                                 seed=0
                                 )
        eval_env = MultiHoverAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT)
        target_reward = 949.5

    model = PPO('MlpPolicy', train_env, verbose=1, seed=0)
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward, verbose=1)
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename + '/',
                                 log_path=filename + '/',
                                 eval_freq=1000,
                                 deterministic=True
                                 )
    model.learn(total_timesteps=int(timesteps), callback=eval_callback, log_interval=10)
    model.save(filename + '/final_model.zip')

    with np.load(filename + '/evaluations.npz') as data:
        eval_ts = data['timesteps'].tolist()
        eval_rw = data['results'][:, 0].tolist()
    print('[baseline] checkpoints in:', filename)
    print('[baseline] eval curve (timestep, reward):', list(zip(eval_ts, eval_rw)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configurable PPO baseline (hover/multi-hover/formation)')
    parser.add_argument('--task',           default='hover',      type=str,      help='Training task: hover, formation (default: hover)', metavar='', choices=['hover', 'formation'])
    parser.add_argument('--multiagent',     default=False,        type=str2bool, help='Train MultiHoverAviary instead of HoverAviary (default: False, hover task only)', metavar='')
    parser.add_argument('--timesteps',      default=200000,       type=int,      help='Total PPO training timesteps (default: 200000)', metavar='')
    parser.add_argument('--output_folder',  default='results',    type=str,      help='Folder where to save checkpoints (default: "results")', metavar='')
    ARGS = parser.parse_args()
    run(**vars(ARGS))
