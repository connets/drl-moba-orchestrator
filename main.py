import os
from time import time

import argparse

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from gym.wrappers import FlattenObservation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from mec_moba.envs import MecMobaDQNEvn

import torch, math, numpy as np
from torch import nn
import tianshou as ts


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


def parse_cli_args():
    parser = argparse.ArgumentParser(description="DRL MOBA on Stable Baseline 3")
    parser.add_argument('--train-epochs', type=int, default=52 * 10, help="Number of training weeks")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, required=False, default=None)
    dqn_grp_parser = parser.add_argument_group('DQN')
    dqn_grp_parser.add_argument('--dqn-batch-size', default=32, type=int)
    dqn_grp_parser.add_argument('--dqn-buffer-size', default=100_000, type=int)
    dqn_grp_parser.add_argument('--dqn-final-epsilon', default=0.05, type=float)
    dqn_grp_parser.add_argument('--dqn-learning-starts', default=5000)
    dqn_grp_parser.add_argument('--dqn-exploration_fraction', default=0.1)

    resume_evaluate_mtx_grp = parser.add_mutually_exclusive_group()
    resume_evaluate_mtx_grp.add_argument('--resume', action='store_true')
    resume_evaluate_mtx_grp.add_argument('--evaluate', action='store_true')

    return parser.parse_args()


def main(cli_args):
    # log_dir = "./tmp/gym/{}".format(int(time()))
    # os.makedirs(log_dir, exist_ok=True)

    env = MecMobaDQNEvn(reward_weights=(0.25, 0.5, 1))
    env = FlattenObservation(env)

    test_env = MecMobaDQNEvn(reward_weights=(0.25, 0.5, 1))
    test_env = FlattenObservation(test_env)
    # check_env(env, warn=True)

    learn_weeks = 52 * 100
    save_freq_steps = 1008 * 52

    checkpoint_callback = CheckpointCallback(save_freq=save_freq_steps, save_path='./logs/',
                                             name_prefix='rl_mlp_model_2')

    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    net = Net(state_shape, action_shape)
    optim = torch.optim.Adam(net.parameters(), lr=1e-4)

    policy = ts.policy.DQNPolicy(net, optim, discount_factor=0.9, estimation_step=1, target_update_freq=2000)

    train_collector = ts.data.Collector(policy, env, ts.data.PrioritizedReplayBuffer(100000, alpha=0.6, beta=0.2),
                                        exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_env, exploration_noise=True)

    from torch.utils.tensorboard import SummaryWriter
    from tianshou.utils import TensorboardLogger
    logdir = 'logs/dqn'
    writer = SummaryWriter(logdir)
    logger = TensorboardLogger(writer)

    def save_policy_fn(policy_obj):
        print('saving policy')
        torch.save(policy_obj.state_dict(), f'{logdir}/dqn.pth')

    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=52 * 20, step_per_epoch=1008, step_per_collect=1,
        update_per_step=4, episode_per_test=10, batch_size=64,
        train_fn=lambda epoch, env_step: policy.set_eps(0.1),
        test_fn=lambda epoch, env_step: policy.set_eps(0),
        stop_fn=lambda mean_rewards: mean_rewards >= -200,
        save_fn=save_policy_fn,
        logger=logger)
    print(f'Finished training! Use {result["duration"]}')

    # model = DDQN('MlpPolicy', env,
    #              verbose=1,
    #              learning_starts=100,
    #              buffer_size=100_000,
    #              target_update_interval=2000,
    #              #tau=0.001,
    #              exploration_fraction=0.2,
    #              exploration_final_eps=0.02,
    #              batch_size=64,
    #              # policy_kwargs={'net_arch': [64,64,64]},
    #              tensorboard_log="./tb_log/dqn_mec_moba_tensorboard/")

    # model = PPO('MlpPolicy', env, verbose=1, n_steps=500, batch_size=50,
    #             vf_coef=0.5, ent_coef=0.01, tensorboard_log="./tb_log/ppo_mec_moba_tensorboard/")

    # obs = env.reset()
    # for i in range(1000):
    #     action, _state = model.predict(obs, deterministic=True)
    #     print(action)
    #     obs, reward, done, info = env.step(action)
    #     env.render()
    #     if done:
    #         obs = env.reset()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(cli_args=parse_cli_args())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
