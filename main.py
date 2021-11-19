import gym

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from mec_moba.envs import MecMobaDQNEvn


class MyCallBack(BaseCallback):

    def init_callback(self, model):
        super(MyCallBack, self).init_callback(model)
        self.env = model.env

    def _on_step(self) -> bool:
        # print(self.num_timesteps)
        return True


def print_hi(name):
    env = MecMobaDQNEvn()

    learn_weeks = 52 * 10

    checkpoint_callback = CheckpointCallback(save_freq=1008, save_path='./logs/',
                                             name_prefix='rl_mpl_model')

    model = DQN('MlpPolicy', env, verbose=1, learning_starts=1008 * 2)
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
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
