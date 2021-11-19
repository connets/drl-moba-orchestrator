import gym
import numpy as np
from gym.spaces import Box, Discrete
import itertools

from mec_moba.environment.action_controller import DqnAction
from mec_moba.environment.environment import Environment
import mec_moba.environment.action_controller as action_controller


class MecMobaDQNEvn(gym.Env):

    @staticmethod
    def _create_action_space_map():
        cap_list = [80]
        op_list = [0, 10, 20, 30, 40]
        deploy_list = [0, 25, 50, 75]
        migration_list = [0, 25, 50]
        actions = itertools.product(cap_list, op_list, deploy_list, migration_list)
        actions = filter(lambda a: a[2] + a[3] > 0, actions)
        return {a_id: DqnAction(*a) for a_id, a in enumerate(actions)}

    def __init__(self):
        super(MecMobaDQNEvn, self).__init__()
        self.observation_space = Box(low=0.0, high=1.0, shape=(1, 21))
        self._actions_dict = MecMobaDQNEvn._create_action_space_map()
        self.action_space = Discrete(len(self._actions_dict))

        self._internal_env = Environment()
        self.state_obs = self._initial_observation()
        self.t_slot = 0

    def action_id_to_human(self, action):
        return self._actions_dict[action]

    def _initial_observation(self):
        return self._internal_env.get_time_slot_state().to_array()

    def step(self, action):
        # print(self._actions_dict[action])
        action_: DqnAction = self._actions_dict[action]
        action_result = action_controller.do_action(action_, self._internal_env)
        self._internal_env.implement_action(action_result)
        week_end = self._internal_env.inc_timeslot()
        self._internal_env.cleanup_terminated_matches(t_slot_end=False)
        new_match_requests = self._internal_env.match_generator.get_match_requests(self._internal_env.epoch_t_slot)

        self._internal_env.match_controller.enqueue_match_requests(new_match_requests)

        observation = self._internal_env.get_time_slot_state().to_array()
        reward, _ = self._internal_env.compute_reward(action_, action_result)
        return observation, reward, week_end, {}
        #

    def reset(self):
        # print('end week')
        self._internal_env.reset()
        return self._initial_observation()

    def render(self, mode="human"):
        pass
        # env_state = np.zeros((10 + 3, 12))
        # t_slot_state = self._internal_env.get_time_slot_state()
        # for i, f_l in enumerate(t_slot_state.facility_utilization):
        #     f_l = max(0, int(round((f_l * 10))))
        #     env_state[i, :f_l - 1] = 1

        # for r in reversed(range(env_state.shape[0])):
        #     print("".join(["*" if x == 1 else " " for x in env_state[r, :]]))
