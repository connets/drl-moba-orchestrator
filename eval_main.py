import abc
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
import torch
import torch.nn as nn
import math
import numpy as np
import tianshou as ts

from mec_moba.envs import MecMobaDQNEvn
from mec_moba.envs.utils.stepLoggerWrapper import StepLogger


class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        layer_dim = pow(2, math.floor(math.log2(max(np.prod(state_shape), np.prod(action_shape)))))
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), layer_dim), nn.ReLU(inplace=True),
            nn.Linear(layer_dim, layer_dim), nn.ReLU(inplace=True),
            nn.Linear(layer_dim, layer_dim), nn.ReLU(inplace=True),
            nn.Linear(layer_dim, np.prod(action_shape)),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state


class TestAgent:
    def __init__(self):
        pass

    def initialize(self, env):
        pass

    @abc.abstractmethod
    def policy_type(self) -> str:
        pass

    @abc.abstractmethod
    def set_env_seed(self, env_obj, seed):
        pass

    @abc.abstractmethod
    def select_action(self, observation, env_obj: gym.Env):
        pass


class DqnAgent(TestAgent):
    def __init__(self, model_file):
        super().__init__()
        self.model_file = model_file

        # self.model.load_state_dict(state_dict=)

    def initialize(self, env):
        state_shape = env.observation_space.shape or env.observation_space.n
        action_shape = env.action_space.shape or env.action_space.n
        net = Net(state_shape, action_shape)
        self.policy = ts.policy.DQNPolicy(net, None, discount_factor=0.99, estimation_step=1, target_update_freq=2000)
        self.policy.load_state_dict(torch.load(self.model_file))
        self.policy.eval()
        self.policy.set_eps(0)

    def policy_type(self) -> str:
        return 'dqn'

    def select_action(self, observation, env_obj: gym.Env):
        action, _ = self.policy.forward(observation)
        return action


class RandomAgent(TestAgent):

    def policy_type(self) -> str:
        return 'random'

    def set_env_seed(self, env_obj, seed):
        env_obj.seed(seed)

    def select_action(self, observation, env_obj: gym.Env):
        return env_obj.action_space.sample()


def run_test(seed, agent: TestAgent, t_slot_to_test=1008, gen_requests_until=1008):
    env = MecMobaDQNEvn(reward_weights=reward_weights,
                        gen_requests_until=gen_requests_until,
                        log_match_data=True,
                        base_log_dir=f'logs/{seed}/match_logs_{agent.policy_type()}')
    # env = Monitor(env)
    env = StepLogger(env, logfile=f'logs/{seed}/eval_test_{agent.policy_type()}.csv')
    env = FlattenObservation(env)
    env.seed(seed)

    agent.initialize(env)
    collector = ts.data.Collector(agent.policy, env)
    collector.collect(n_step=t_slot_to_test)

    # with open(f'logs/{seed}/eval_test_{agent.policy_type()}.csv', 'w') as f:
    #     f.write(f"{','.join(head_line)}\n")
    #     t_slot = 0
    #     week = 0
    #     collector.collect(n_step=1)
    #     for i in range(t_slot_to_test):
    #         action = agent.select_action(buffer[0], env)
    #         obs_pre = obs
    #         # print(env.action_id_to_human(action))
    #         obs, reward, done, info = env.step(action)
    #
    #         # LOG
    #         args_to_write = [str(week), str(t_slot)]
    #         args_to_write += [str(i) for i in obs_pre]
    #         args_to_write += list(str(env.action_id_to_human(action))[1:-1].split(','))
    #         args_to_write += [str(i) for i in obs]
    #         args_to_write.append(str(reward))
    #         f.write(f"{','.join(args_to_write)}\n")
    #
    #         # env.render()
    #         t_slot += 1
    #         if done:
    #             print('week ends')
    #             week += 1
    #             t_slot = 0
    #             obs = env.reset()


seeds = [2000]

reward_weights = (0.25, 0.5, 1)  # , 0, 0, 0)

for seed in seeds:
    os.makedirs(f'logs/{seed}', exist_ok=True)

    dqn_agent = DqnAgent(model_file='logs/dqn/dqn.pth')

    run_test(seed, dqn_agent, t_slot_to_test=144 + 12 * 6, gen_requests_until=144)

    # reward_weights = (1, 2, 1, 0, 0, 0)
    # env = MecMobaDQNEvn(reward_weights=reward_weights, log_match_data=True, base_log_dir=f'logs/{seed}/match_logs')
    # env = Monitor(env)
    # env = FlattenObservation(env)
    # # check_env(env, warn=True)

    # MODIFICA PATH!!!!
    # model = DQN.load("out/dqn_icdcs22_2021_12_18_rw_param/2/saved_models/dqn_mlp_model_1048320_steps")
    # model.set_env(env)
    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5, deterministic=True)
    # model.set_random_seed(seed)

    # (self.running_qos_prob + self.facility_utilization +
    # [self.queue_occupation, self.mean_queue_waiting_time, self.mean_running_time, self.migrable_ratio, self.queue_drop_rate])
    # with open(f'logs/{seed}/eval_test.csv', 'w') as f:
    #     f.write(f"{','.join(head_line)}\n")
    #     t_slot = 0
    #     week = 0
    #     obs = env.reset()
    #     for i in range(1008 * 1):
    #         action, _ = model.predict(obs, deterministic=True)
    #         obs_pre = obs
    #         # print(env.action_id_to_human(action))
    #         obs, reward, done, info = env.step(action)
    #
    #         # LOG
    #         args_to_write = [str(week), str(t_slot)]
    #         args_to_write += [str(i) for i in obs_pre]
    #         args_to_write += list(str(env.action_id_to_human(action))[1:-1].split(','))
    #         args_to_write += [str(i) for i in obs]
    #         args_to_write.append(str(reward))
    #         f.write(f"{','.join(args_to_write)}\n")
    #
    #         # env.render()
    #         t_slot += 1
    #         if done:
    #             print('week ends')
    #             week += 1
    #             t_slot = 0
    #             obs = env.reset()

# print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
