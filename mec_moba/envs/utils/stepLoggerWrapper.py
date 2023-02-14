import gym

from mec_moba.envs import MecMobaDQNEvn

state_columns = [f'qos_{i}' for i in range(6)]
state_columns += [f'f_{i}' for i in range(7)]  # TODO
state_columns += ['queue', 'w_t', 'run', 'mig', 'drop']

next_state_columns = [f'{c}_next' for c in state_columns]
head_line = ['epoch', 't_slot']
head_line += state_columns
head_line += ['action_cap', 'action_op', 'action_dep', 'action_mig', ]
head_line += next_state_columns + ['reward']


class StepLogger(gym.Wrapper):
    def __init__(self, env: MecMobaDQNEvn, logfile):
        super().__init__(env)
        self.env = env
        self.step_logger = open(logfile, 'w')
        self.step_logger.write(f"{','.join(head_line)}\n")

    def _log(self, observation, action, reward, next_observation):
        week = int((self.env._internal_env.absolute_t_slot - 1) / 1008)
        t_slot = (self.env._internal_env.absolute_t_slot - 1) % 1008
        args_to_write = [str(week), str(t_slot)]
        args_to_write += [str(i) for i in observation[0]]
        args_to_write += list(str(self.env.action_id_to_human(action))[1:-1].split(','))
        args_to_write += [str(i) for i in next_observation[0]]
        args_to_write.append(str(reward))
        self.step_logger.write(f"{','.join(args_to_write)}\n")

    def step(self, action):
        observation = self.env.get_current_observation()
        next_observation, reward, week_end, metadata = self.env.step(action)
        self._log(observation, action, reward, next_observation)

        return next_observation, reward, week_end, metadata

    def close(self):
        self.env.close()
        self.step_logger.close()
