import argparse

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import progressbar
from gym.wrappers import FlattenObservation
from mec_moba.envs import MecMobaDQNEvn

import torch, math, numpy as np
from torch import nn
import tianshou as ts

from mec_moba.envs.utils.rewardComponentsLoggerWrapper import RewardComponentLogger


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
    parser = argparse.ArgumentParser(description="DRL MOBA on Stable Baseline 3 and tianshou")
    parser.add_argument('--train-epochs', type=int, default=52 * 10, help="Number of training weeks")
    parser.add_argument('--seed', type=int, required=False, default=None)
    dqn_grp_parser = parser.add_argument_group('DQN')
    dqn_grp_parser.add_argument('--dqn-batch-size', default=32, type=int)
    dqn_grp_parser.add_argument('--dqn-buffer-size', default=100_000, type=int)
    dqn_grp_parser.add_argument('--dqn-train-eps', default=0.1, type=float)
    # dqn_grp_parser.add_argument('--dqn-learning-starts', default=5000)
    rw_grp_parser = parser.add_argument_group('Reward function')
    rw_grp_parser.add_argument('--rw-w-qos', default=1.0, type=float)
    rw_grp_parser.add_argument('--rw-w-op', default=1.0, type=float)
    rw_grp_parser.add_argument('--rw-w-wt', default=1.0, type=float)

    return parser.parse_args()


def main(cli_args):
    logdir = 'logs/dqn-test'
    # log_dir = "./tmp/gym/{}".format(int(time()))
    # os.makedirs(log_dir, exist_ok=True)

    env = MecMobaDQNEvn(reward_weights=(0.25, 0.5, 1),return_reward_components=True)
    env = RewardComponentLogger(env, logfile=f'{logdir}/rw_components.csv', compressed=True)
    env = FlattenObservation(env)

    test_env = MecMobaDQNEvn(reward_weights=(0.25, 0.5, 1))
    #test_env = RewardComponentLogger(test_env, logfile=f'{logdir}/test_rw_components.csv')
    test_env = FlattenObservation(test_env)
    # check_env(env, warn=True)

    learn_weeks = 52 * 100
    save_freq_steps = 1008 * 52

    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    net = Net(state_shape, action_shape)
    optim = torch.optim.Adam(net.parameters(), lr=0.0005)

    policy = ts.policy.DQNPolicy(net, optim, discount_factor=0.9, estimation_step=1, target_update_freq=2000)

    train_collector = ts.data.Collector(policy, env, ts.data.PrioritizedReplayBuffer(100000, alpha=0.6, beta=0.2))
    # train_collector.collect(n_step=6)

    test_collector = ts.data.Collector(policy, test_env)
    # test_collector.collect(n_step=6)

    from torch.utils.tensorboard import SummaryWriter
    from tianshou.utils import TensorboardLogger

    writer = SummaryWriter(logdir)
    logger = TensorboardLogger(writer)

    def save_policy_fn(policy_obj, year):
        print('saving policy')
        torch.save(policy_obj.state_dict(), f'{logdir}/dqn-{year}.pth')

    # pre-collect at least 5000 transitions with random action before training
    train_collector.collect(n_episode=10, random=True)
    policy.set_eps(0.1)
    env_train_step = 0
    for year in range(20):  # total step
        print('Year ', year)
        for week in progressbar.progressbar(range(52)):
            for i in range(int(1008 / 4)):
                collect_result = train_collector.collect(n_step=4)
                env_train_step += int(collect_result["n/st"])
                update_results = policy.update(64, train_collector.buffer)
                logger.log_update_data(update_results, env_train_step)
                logger.log_train_data(collect_result, env_train_step)

        # TEST
        policy.set_eps(0)
        test_result = test_collector.collect(n_episode=10)
        logger.log_test_data(test_result, year)
        save_policy_fn(policy, year)
        if test_result['rews'].mean() >= 0:
            print(f'Finished training! Year {year}: Test mean returns: {test_result["rews"].mean()}')
            break
        else:
            print(f'Testing: year {year}: Test mean returns: {test_result["rews"].mean()}')
            # back to training eps
            policy.set_eps(0.1)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(cli_args=parse_cli_args())

