import time
import gym

from mec_moba.envs import MecMobaDQNEvn


class TimeLoggerWrapper(gym.Wrapper):
    def __init__(self, env, logfile):
        super().__init__(env)
        self.env = env
        self.step_logger = open(logfile, 'w')
        self.step_logger.write('excution_time\n')

    def step(self, action):
        s_time = time.time()
        #observation = self.env.get_current_observation()
        next_observation, reward, week_end, metadata = self.env.step(action)
        #self._log(observation, action, reward, next_observation)
        self.step_logger.write(f'{time.time() - s_time}\n')
        return next_observation, reward, week_end, metadata

    def close(self):
        self.env.close()
        self.step_logger.close()
