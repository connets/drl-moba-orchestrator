import logging

import gym
import numpy as np
from gym.spaces import Box, Discrete
import itertools

from mec_moba.environment.action_controller import DqnAction
from mec_moba.environment.environment import Environment
import mec_moba.environment.action_controller as action_controller
from mec_moba.environment.utils.logging_utils import close_all_logs


class MecMobaDQNEvn(gym.Env):

    @staticmethod
    def _create_action_space_map():
        cap_list = [80]
        op_list = [0, 25, 50, 75, 100]
        deploy_list = [0, 25, 50, 75, 100]
        migration_list = [0, 25, 50, 75, 100]
        actions = itertools.product(cap_list, op_list, deploy_list, migration_list)
        actions = filter(lambda a: a[2] + a[3] > 0, actions)
        return {a_id: DqnAction(*a) for a_id, a in enumerate(actions)}

    def __init__(self, reward_weights=None, gen_requests_until=None, match_probability_file=None, log_match_data=False, base_log_dir=None):
        super(MecMobaDQNEvn, self).__init__()
        if reward_weights is None:
            reward_weights = Environment.default_reward_weights()

        self._internal_env = Environment(reward_weights=reward_weights,
                                         gen_requests_until=gen_requests_until,
                                         match_probability_file=match_probability_file)
        logging.info(f'State features number: {self._internal_env.get_time_slot_state_n_features()}')
        self.observation_space = Box(low=0.0, high=1.0, shape=(1, self._internal_env.get_time_slot_state_n_features()))
        self._actions_dict = MecMobaDQNEvn._create_action_space_map()
        self.action_space = Discrete(len(self._actions_dict))

        # self.state_obs = self._initial_observation()
        self.t_slot = 0
        self._rng_seed = None

        self._log_match_data = log_match_data
        self._base_log_dir = base_log_dir
        assert not log_match_data or (log_match_data and base_log_dir is not None)

    def action_id_to_human(self, action):
        return self._actions_dict[action]

    def _initial_observation(self):
        return self._internal_env.get_time_slot_state().to_array()

    def _generate_and_enqueue_requests(self):
        new_match_requests = self._internal_env.match_generator.get_match_requests(self._internal_env.epoch_t_slot)
        self._internal_env.match_controller.enqueue_match_requests(new_match_requests)

    def step(self, action):
        # print(self._actions_dict[action])
        action_: DqnAction = self._actions_dict[action]
        action_result = action_controller.do_action(action_, self._internal_env)
        self._internal_env.implement_action(action_result)
        if self._log_match_data:
            self.log_all_matches_data()
        week_end = self._internal_env.inc_timeslot()
        self._internal_env.cleanup_terminated_matches(t_slot_end=False)
        self._generate_and_enqueue_requests()

        observation = self._internal_env.get_time_slot_state().to_array()
        reward, _ = self._internal_env.compute_reward(action_, action_result)
        return observation, reward, week_end, {}
        #

    def reset(self):
        self._internal_env.reset()
        self._generate_and_enqueue_requests()
        return self._initial_observation()

    def get_current_observation(self):
        return self._internal_env.get_time_slot_state().to_array()

    def log_all_matches_data(self):
        self._internal_env.log_all_matches_data(log_dir=self._base_log_dir)

    def render(self, mode="human"):
        return

    def close(self):
        close_all_logs()

    def seed(self, seed=None):
        self._rng_seed = seed
        self._internal_env.set_seed(seed)
