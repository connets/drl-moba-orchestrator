import os
from time import time

import gym

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from gym.utils.env_checker import check_env
from gym.wrappers import FlattenObservation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from mec_moba.envs import MecMobaDQNEvn


class MyCallBack(BaseCallback):

    def init_callback(self, model):
        super(MyCallBack, self).init_callback(model)
        self.env = model.env

    def _on_step(self) -> bool:
        # print(self.num_timesteps)
        return True


def main():
    log_dir = "./tmp/gym/{}".format(int(time()))
    os.makedirs(log_dir, exist_ok=True)

    env = MecMobaDQNEvn()
    env = FlattenObservation(env)
    #check_env(env, warn=True)

    learn_weeks = 52 * 10
    save_freq_steps = 1008*10

    checkpoint_callback = CheckpointCallback(save_freq=save_freq_steps, save_path='./logs/',
                                             name_prefix='rl_mlp_model')

    # model = DQN('MlpPolicy', env,
    #             verbose=1, learning_starts=1000,
    #             tensorboard_log="./tb_log/dqn_mec_moba_tensorboard/")

    model = PPO('MlpPolicy', env, verbose=1, n_steps=500, batch_size=50,
                vf_coef=0.5, ent_coef=0.01, tensorboard_log="./tb_log/ppo_mec_moba_tensorboard/")
    model.set_random_seed(1000)
    model.learn(total_timesteps=1008 * learn_weeks, callback=checkpoint_callback)

    obs = env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        print(action)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
