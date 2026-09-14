"""Evaluate a trained formation policy headlessly and report formation metrics.

Loads a stable-baselines3 model trained on `FormationAviary`, rolls out
deterministic episodes, and reports per-episode return plus two formation
metrics: inter-drone distance error against the designed separation and
per-drone slot tracking RMSE (steady-state, skipping the take-off transient).

Example
-------

    $ python eval_formation.py --model_path results/baseline-*/best_model.zip --episodes 5
"""
import os
import glob
import argparse

import numpy as np
from stable_baselines3 import PPO

from gym_pybullet_drones.envs.FormationAviary import FormationAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_EPISODES = 5
DEFAULT_AGENTS = 2


def run(model_path, episodes=DEFAULT_EPISODES):
    kw = dict(num_drones=DEFAULT_AGENTS,
              obs=ObservationType('kin'),
              act=ActionType('one_d_rpm')
              )
    model = PPO.load(model_path)
    sep = 1.0
    returns, dist_errs, slot_rmses = [], [], [[] for i in range(DEFAULT_AGENTS)]
    for ep in range(episodes):
        env = FormationAviary(**kw)
        obs, info = env.reset(seed=ep)
        done, ret, steps = False, 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ret += float(reward)
            steps += 1
            states = np.array([env._getDroneStateVector(i) for i in range(DEFAULT_AGENTS)])
            if steps > 2*env.CTRL_FREQ:  # steady-state: skip take-off transient
                dist = np.linalg.norm(states[0][0:3]-states[1][0:3])
                dist_errs.append(abs(dist - sep))
                for j in range(DEFAULT_AGENTS):
                    slot_rmses[j].append(np.linalg.norm(states[j][0:3]-env.TARGET_POS[j, :])**2)
        env.close()
        returns.append(ret)
        print("[eval] episode %d: return %.1f | steps %d" % (ep, ret, steps))

    print("[eval] mean return: %.1f +/- %.1f (theoretical max 1200)" % (np.mean(returns), np.std(returns)))
    print("[eval] inter-drone distance error: mean %.4f m | max %.4f m" % (np.mean(dist_errs), np.max(dist_errs)))
    for j in range(DEFAULT_AGENTS):
        print("[eval] drone%d slot RMSE (t>2s): %.4f m" % (j, np.sqrt(np.mean(slot_rmses[j]))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained formation policy')
    parser.add_argument('--model_path', default=None, type=str,
                        help='Path to a stable-baselines3 model zip (default: newest results/baseline-*/best_model.zip)', metavar='')
    parser.add_argument('--episodes', default=DEFAULT_EPISODES, type=int, help='Number of evaluation episodes (default: 5)', metavar='')
    ARGS = parser.parse_args()
    path = ARGS.model_path
    if path is None:
        candidates = sorted(glob.glob(os.path.join('results', 'baseline-*', 'best_model.zip')))
        if not candidates:
            candidates = sorted(glob.glob(os.path.join('results', 'baseline-*', 'final_model.zip')))
        path = candidates[-1]
    run(path, ARGS.episodes)
