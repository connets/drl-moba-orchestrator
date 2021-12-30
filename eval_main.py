import abc
import os
from time import time
import numpy as np
import gym

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from gym.utils.env_checker import check_env
from gym.wrappers import FlattenObservation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN, TD3
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

from mec_moba.envs import MecMobaDQNEvn
from mec_moba.envs.mec_moba_cont_act_env import MecMobaContinuosActionEvn


class TestAgent:
    def __init__(self):
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
        self.model = DQN.load(model_file)

    def policy_type(self) -> str:
        return 'dqn'

    def set_env_seed(self, env_obj, seed):
        self.model.set_env(env_obj)
        self.model.set_random_seed(seed)

    def select_action(self, observation, env_obj: gym.Env):
        action, _ = self.model.predict(observation, deterministic=True)
        return action


class TD3Agent(TestAgent):
    def __init__(self, model_file):
        super().__init__()
        self.model = TD3.load(model_file)

    def policy_type(self) -> str:
        return 'td3'

    def set_env_seed(self, env_obj, seed):
        self.model.set_env(env_obj)
        self.model.set_random_seed(seed)

    def select_action(self, observation, env_obj: gym.Env):
        action, _ = self.model.predict(observation, deterministic=True)
        return action


class RandomAgent(TestAgent):

    def policy_type(self) -> str:
        return 'random'

    def set_env_seed(self, env_obj, seed):
        env_obj.seed(seed)

    def select_action(self, observation, env_obj: gym.Env):
        return np.random.random(3)


def run_test(seed, agent: TestAgent, t_slot_to_test=1008, gen_requests_until=1008):
    env = MecMobaContinuosActionEvn(reward_weights=reward_weights,
                                    gen_requests_until=gen_requests_until,
                                    log_match_data=True,
                                    base_log_dir=f'logs/{seed}/match_logs_{agent.policy_type()}')
    env = Monitor(env)
    env = FlattenObservation(env)

    agent.set_env_seed(env, seed)

    with open(f'logs/{seed}/eval_test_{agent.policy_type()}.csv', 'w') as f:
        f.write(f"{','.join(head_line)}\n")
        t_slot = 0
        week = 0
        obs = env.reset()
        for i in range(t_slot_to_test):
            action = agent.select_action(obs, env)
            obs_pre = obs
            # print(env.action_id_to_human(action))
            obs, reward, done, info = env.step(action)

            # LOG
            args_to_write = [str(week), str(t_slot)]
            args_to_write += [str(i) for i in obs_pre]
            args_to_write += ["0"] + [str(a) for a in action]  # _id_to_human(action))[1:-1].split(','))
            args_to_write += [str(i) for i in obs]
            args_to_write.append(str(reward))
            f.write(f"{','.join(args_to_write)}\n")

            # env.render()
            t_slot += 1
            if done:
                print('week ends')
                week += 1
                t_slot = 0
                obs = env.reset()


seeds = [2000]

reward_weights = (1, 1, 1)  # , 0, 0, 0)

state_columns = [f'qos_{i}' for i in range(6)]
state_columns += [f'f_{i}' for i in range(7)]
state_columns += ['queue', 'w_t', 'run', 'mig', 'drop']

next_state_columns = [f'{c}_next' for c in state_columns]
head_line = ['epoch', 't_slot']
head_line += state_columns
head_line += ['action_cap', 'action_op', 'action_dep', 'action_mig', ]
head_line += next_state_columns + ['reward']

for seed in seeds:
    os.makedirs(f'logs/{seed}', exist_ok=True)

    rnd_agent = RandomAgent()
    run_test(seed, rnd_agent, t_slot_to_test=144 + 12 * 6, gen_requests_until=144)

    dqn_agent = TD3Agent(model_file='logs/rl_mlp_model_2_314496_steps.zip')

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
