import abc
import os

from gym.wrappers import FlattenObservation
import torch
import tianshou as ts

from mec_moba.envs import MecMobaDQNEvn
from mec_moba.envs.utils.stepLoggerWrapper import StepLogger
from mec_moba.policy_models.mlp_policy import MLPNet


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


class DqnAgent(TestAgent):
    def __init__(self, model_file, layer_dim=None, num_layers=2):
        super().__init__()
        self.model_file = model_file
        self.layer_dim = layer_dim
        self.num_layers = num_layers

        # self.model.load_state_dict(state_dict=)

    def initialize(self, env):
        state_shape = env.observation_space.shape or env.observation_space.n
        action_shape = env.action_space.shape or env.action_space.n
        net = MLPNet(state_shape, action_shape, layer_dim=self.layer_dim, num_layers=self.num_layers)
        # print(net.model)
        # print(self.layer_dim, self.num_layers, torch.load(self.model_file).keys())
        self.policy = ts.policy.DQNPolicy(net, None)  # discount_factor=0.99, estimation_step=1, target_update_freq=2000)
        self.policy.load_state_dict(torch.load(self.model_file), strict=False)
        self.policy.eval()
        self.policy.set_eps(0)

    def policy_type(self) -> str:
        return 'dqn'

    # def select_action(self, observation, env_obj: gym.Env):
    #     action, _ = self.policy.forward(observation)
    #     return action


class RandomAgent(TestAgent):

    def policy_type(self) -> str:
        return 'random'

    def initialize(self, env):
        state_shape = env.observation_space.shape or env.observation_space.n
        action_shape = env.action_space.shape or env.action_space.n
        net = MLPNet(state_shape, action_shape)
        self.policy = ts.policy.DQNPolicy(net, None, discount_factor=0.99, estimation_step=1, target_update_freq=2000)
        self.policy.eval()
        self.policy.set_eps(1.0)

    # def select_action(self, observation, env_obj: gym.Env):
    #     return env_obj.action_space.sample()


def run_test(seed, agent: TestAgent, base_log_dir, t_slot_to_test=1008, gen_requests_until=1008, match_probability_file=None):
    env = MecMobaDQNEvn(reward_weights=reward_weights,
                        gen_requests_until=gen_requests_until,
                        log_match_data=True,
                        match_probability_file=match_probability_file,
                        base_log_dir=f'{base_log_dir}/match_logs_{agent.policy_type()}')
    # env = Monitor(env)
    env = StepLogger(env, logfile=f'{base_log_dir}/eval_test_{agent.policy_type()}.csv')
    env = FlattenObservation(env)
    env.seed(seed)

    agent.initialize(env)
    collector = ts.data.Collector(agent.policy, env)
    collector.collect(n_step=t_slot_to_test)


if __name__ == '__main__':
    seeds = [1000]

    #reward_weights = (0.25, 0.5, 1)  # , 0, 0, 0)
    reward_weights = [1, 1, 0.25]

    for seed in seeds:
        os.makedirs(f'logs/{seed}', exist_ok=True)

        dqn_agent = DqnAgent(model_file='out/dqn_icdcs22_2022_01_10_6000matches_6ts_rw_weight_FULL/22/saved_models/policy-0.pth')

        run_test(seed, dqn_agent, t_slot_to_test=144 + 6 * 5,
                 gen_requests_until=144, base_log_dir=f'logs/{seed}',
                 match_probability_file=None)  # 'data/match_probability_uniform.csv')
