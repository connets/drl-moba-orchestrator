import os
from time import time
import sqlite3

import argparse
import yaml
from multiprocessing import Pool

from gym.wrappers import FlattenObservation
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from collections import namedtuple
import itertools

import torch, math, numpy as np
from torch import nn
import tianshou as ts

from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

from mec_moba.envs import MecMobaDQNEvn
from mec_moba.policy_models.mlp_policy import MLPNet

grid_search_params = ['buffer_size',
                      'target_update_interval',
                      'gamma',
                      'train_eps',
                      'batch_size',
                      'learning_rate',
                      'reward_weights']

run_parameter_fields_to_save = ['run_id', 'train_epochs', 'seed'] + grid_search_params
extra_run_parameter_fields = ['base_dir']  # these extras fields are needed internally but are not saved
run_parameter_fields = run_parameter_fields_to_save + extra_run_parameter_fields

sqlite_field_type_dict = {'run_id': 'text',
                          'train_epochs': 'integer',
                          'seed': 'integer',
                          'buffer_size': 'integer',
                          'target_update_interval': 'integer',
                          'gamma': 'real',
                          'train_eps': 'real',
                          'batch_size': 'integer',
                          'learning_rate': 'real',
                          'reward_weights': 'text'
                          }

assert all(map(lambda k: k in sqlite_field_type_dict, run_parameter_fields_to_save))

RunParameters = namedtuple('RunParameters', run_parameter_fields)


# DB

def table_creation_utils(db_conn):
    cur = db_conn.cursor()
    create_table_str = f'CREATE TABLE IF NOT EXISTS experiments ( {",".join(f"{f} {t}" for f, t in sqlite_field_type_dict.items())} )'
    cur.execute(create_table_str)
    db_conn.commit()
    cur.close()


def insert_all_runs(db_conn, experiments):
    cur = db_conn.cursor()

    # print(f"insert into experiments ( {','.join(['?'] * len(sqlite_field_type_dict))})")

    def convert_if_str(k, v):
        return str(v) if sqlite_field_type_dict[k] == 'text' else v

    # experiments = (e._asdict() for e in experiments)
    experiments = (tuple(convert_if_str(k, getattr(e, k)) for k in run_parameter_fields_to_save) for e in experiments)
    cur.executemany(f"insert into experiments values ( {','.join(['?'] * len(sqlite_field_type_dict))})", experiments)
    db_conn.commit()
    cur.close()


def create_save_policy_fn(model_save_dir, save_policy_id):
    def f(policy_obj):
        torch.save(policy_obj.state_dict(), f'{model_save_dir}/dqn-{next(save_policy_id)}.pth')

    return f


def training_process(run_params: RunParameters):
    # check_env(env, warn=True)
    model_save_dir = os.path.join(run_params.base_dir, run_params.run_id, 'saved_models')
    os.makedirs(model_save_dir)

    tb_log_dir = os.path.join(run_params.base_dir, 'dqn_mec_moba_tensorboard')

    learn_weeks = run_params.train_epochs

    train_env = MecMobaDQNEvn(reward_weights=run_params.reward_weights)
    train_env = FlattenObservation(train_env)

    test_env = MecMobaDQNEvn(reward_weights=run_params.reward_weights)
    test_env = FlattenObservation(test_env)

    state_shape = train_env.observation_space.shape or train_env.observation_space.n
    action_shape = train_env.action_space.shape or train_env.action_space.n
    net = MLPNet(state_shape, action_shape)
    optim = torch.optim.Adam(net.parameters(), lr=run_params.learning_rate)

    policy = ts.policy.DQNPolicy(net, optim, discount_factor=0.99, estimation_step=1, target_update_freq=run_params.target_update_interval)
    replay_buffer = ts.data.PrioritizedReplayBuffer(run_params.buffer_size, alpha=0.6, beta=0.2)

    train_collector = ts.data.Collector(policy, train_env, replay_buffer, exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_env, exploration_noise=False)

    tb_logger = TensorboardLogger(SummaryWriter(os.path.join(tb_log_dir,run_params.run_id)))

    save_policy_id = itertools.count(0)
    save_policy_fn = create_save_policy_fn(model_save_dir, save_policy_id)

    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=52 * 30, step_per_epoch=1008, step_per_collect=4,
        update_per_step=1, episode_per_test=10, batch_size=run_params.batch_size,
        train_fn=lambda epoch, env_step: policy.set_eps(run_params.train_eps),
        test_fn=lambda epoch, env_step: policy.set_eps(0),
        stop_fn=lambda mean_rewards: mean_rewards >= 0,
        save_fn=save_policy_fn,
        logger=tb_logger)
    print(f'Finished training! Use {result["duration"]}')


def _generate_run_parameter(run_id, base_dir, train_epochs, seed, grid_params_dict):
    d = {'run_id': str(run_id), 'base_dir': base_dir, 'train_epochs': train_epochs, 'seed': seed}
    d.update(**grid_params_dict)

    return RunParameters(**d)


def main():
    parser = argparse.ArgumentParser(description="DRL MOBA on Stable Baseline 3")
    parser.add_argument('conf_file', type=str, help='Grid search configurator file (.yaml)')
    parser.add_argument('experiment_tag', type=str, help='A name of this experiment setting')
    parser.add_argument('--seed', type=int, default=-1, help="Seed number. -1 or negative values means random seed ")
    parser.add_argument('-j', type=int, default=os.cpu_count() - 1, help='Number of parallel processes', dest='num_processes')
    parser.add_argument('--train-epochs', type=int, default=52 * 10, help="Number of training weeks")
    cli_args = parser.parse_args()

    # create all directories
    if " " in cli_args.experiment_tag:
        raise RuntimeError('experiment_tag argument must not contain spaces')

    base_dir = os.path.join('out', cli_args.experiment_tag)
    os.makedirs(os.path.join(base_dir, 'dqn_mec_moba_tensorboard'), exist_ok=True)
    # Create DB
    db_connection = sqlite3.connect(os.path.join('out', cli_args.experiment_tag, 'experiments.db'))
    table_creation_utils(db_connection)

    # cli read
    seed = int(time()) if cli_args.seed < 0 else cli_args.seed
    train_epochs = cli_args.train_epochs

    # Grid search
    grid_search_config = yaml.load(stream=open(cli_args.conf_file, 'r'), Loader=yaml.SafeLoader)

    # pprint.pprint(list((grid_search_config[p] for p in grid_search_params)))

    experiments = itertools.product(*(grid_search_config[p] for p in grid_search_params))
    experiments = map(lambda v: dict(zip(grid_search_params, v)), experiments)
    experiments = (_generate_run_parameter(run_id=i,
                                           base_dir=base_dir,
                                           train_epochs=train_epochs,
                                           seed=seed,
                                           grid_params_dict=e)
                   for i, e in enumerate(experiments))

    experiments = list(experiments)
    insert_all_runs(db_conn=db_connection, experiments=experiments)

    with Pool(processes=cli_args.num_processes) as pool:
        pool.map(training_process, experiments)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
