import os
from time import time

import gym

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from gym.utils.env_checker import check_env
from gym.wrappers import FlattenObservation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

from mec_moba.envs import MecMobaDQNEvn

env = MecMobaDQNEvn()
env = Monitor(env)
env = FlattenObservation(env)
check_env(env, warn=True)

model = DQN.load("logs/rl_mpl_model_374976_steps")
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5, deterministic=True)
head_line = ['epoch', 't_slot']
head_line += [f'qos_{i}' for i in range(6)]
head_line += [f'f_{i}' for i in range(10)]
head_line += ['queue', 'w_t', 'run', 'mig', 'drop', 'action_cap', 'action_op', 'action_dep', 'action_mig', 'reward']
# (self.running_qos_prob + self.facility_utilization +
# [self.queue_occupation, self.mean_queue_waiting_time, self.mean_running_time, self.migrable_ratio, self.queue_drop_rate])
with open('logs/eval_test.csv', 'w') as f:
    f.write(f"{','.join(head_line)}\n")
    t_slot = 0
    week = 0
    obs = env.reset()
    for i in range(1008 * 10):
        action, _state = model.predict(obs, deterministic=True)
        # print(env.action_id_to_human(action))
        obs, reward, done, info = env.step(action)
        f.write(f"{week},{t_slot},{','.join([str(i) for i in obs])},{','.join(str(env.action_id_to_human(action))[1:-1].split(','))},{reward}\n")
        env.render()
        t_slot += 1
        if done:
            print('week ends')
            week += 1
            t_slot = 0
            obs = env.reset()

# print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
