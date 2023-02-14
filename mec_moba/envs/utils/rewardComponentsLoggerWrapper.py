import gzip

import gym

from mec_moba.envs import MecMobaDQNEvn
import gzip

# next_state_columns = [f'{c}_next' for c in state_columns]
head_line = ['epoch', 't_slot', 'reward',
             'qos', 'op_cost', 'waiting_time',
             'qos_norm', 'op_cost_norm', 'waiting_time_norm']


class RewardComponentLogger(gym.Wrapper):
    def __init__(self, env: MecMobaDQNEvn, logfile, compressed=False):
        super().__init__(env)
        self.env = env
        self._compressed = compressed
        self.step_logger = gzip.open(f'{logfile}.gz', 'w') if compressed else open(logfile, 'w')
        log_line = f"{','.join(head_line)}\n"
        if compressed:
            log_line = log_line.encode('utf8')
        self.step_logger.write(log_line)

    def _log(self, reward, rw_components, rw_components_norm):
        week = int((self.env._internal_env.absolute_t_slot - 1) / 1008)
        t_slot = (self.env._internal_env.absolute_t_slot - 1) % 1008
        args_to_write = [str(week), str(t_slot)]
        args_to_write.append(str(reward))
        args_to_write += [str(r) for r in rw_components]
        args_to_write += [str(r) for r in rw_components_norm]

        log_line = f"{','.join(args_to_write)}\n"
        if self._compressed:
            log_line = log_line.encode('utf8')
        self.step_logger.write(log_line)

    def step(self, action):
        observation = self.env.get_current_observation()
        next_observation, reward, week_end, metadata = self.env.step(action)
        self._log(reward, metadata['rw_components'], metadata['rw_components_norm'])
        del metadata['rw_components']
        del metadata['rw_components_norm']
        return next_observation, reward, week_end, metadata

    def close(self):
        self.env.close()
        self.step_logger.close()
