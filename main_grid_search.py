import os
from time import time
import sqlite3

import argparse
import yaml
from multiprocessing import Pool

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from gym.wrappers import FlattenObservation
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from collections import namedtuple
import itertools

from mec_moba.envs import MecMobaDQNEvn

grid_search_params = ['learning_starts',
                      'buffer_size',
                      'target_update_interval',
                      'gamma',
                      'exploration_final_eps',
                      'batch_size',
                      'learning_rate',
                      'train_freq']

run_parameter_fields_to_save = ['run_id', 'train_epochs', 'seed'] + grid_search_params
extra_run_parameter_fields = ['base_dir']  # these extras fields are needed internally but are not saved
run_parameter_fields = run_parameter_fields_to_save + extra_run_parameter_fields

sqlite_field_type_dict = {'run_id': 'text',
                          'train_epochs': 'integer',
                          'seed': 'integer',
                          'learning_starts': 'integer',
                          'buffer_size': 'integer',
                          'target_update_interval': 'integer',
                          'gamma': 'real',
                          'exploration_final_eps': 'real',
                          'batch_size': 'integer',
                          'learning_rate': 'real',
                          'train_freq': 'integer'
                          }

assert all(map(lambda k: k in sqlite_field_type_dict, run_parameter_fields_to_save))

RunParameters = namedtuple('RunParameters', run_parameter_fields)


def table_creation_utils(db_conn):
    cur = db_conn.cursor()
    create_table_str = f'CREATE TABLE IF NOT EXISTS experiments ( {",".join(f"{f} {t}" for f, t in sqlite_field_type_dict.items())} )'
    cur.execute(create_table_str)
    db_conn.commit()
    cur.close()


def insert_all_runs(db_conn, experiments):

    cur = db_conn.cursor()
    # print(f"insert into experiments ( {','.join(['?'] * len(sqlite_field_type_dict))})")
    # experiments = (e._asdict() for e in experiments)
    experiments = (tuple(getattr(e, k) for k in run_parameter_fields_to_save) for e in experiments)
    cur.executemany(f"insert into experiments values ( {','.join(['?'] * len(sqlite_field_type_dict))})", experiments)
    db_conn.commit()
    cur.close()


def training_process(run_params: RunParameters):
    env = MecMobaDQNEvn()
    env = FlattenObservation(env)
    # check_env(env, warn=True)
    model_save_dir = os.path.join(run_params.base_dir, run_params.run_id, 'saved_models')
    os.makedirs(model_save_dir)

    tb_log_dir = os.path.join(run_params.base_dir, 'dqn_mec_moba_tensorboard')

    learn_weeks = run_params.train_epochs
    save_freq_steps = 1008 * 10

    checkpoint_callback = CheckpointCallback(save_freq=save_freq_steps, save_path=model_save_dir,
                                             name_prefix='dqn_mlp_model')

    model = DQN('MlpPolicy', env,
                verbose=1,
                learning_starts=run_params.learning_starts,
                buffer_size=run_params.buffer_size,
                target_update_interval=run_params.target_update_interval,
                gamma=run_params.gamma,
                exploration_final_eps=run_params.exploration_final_eps,
                batch_size=run_params.batch_size,
                train_freq=run_params.train_freq,
                learning_rate=run_params.learning_rate,
                tensorboard_log=tb_log_dir)

    model.set_random_seed(run_params.seed)
    model.learn(total_timesteps=1008 * learn_weeks, callback=checkpoint_callback, tb_log_name=run_params.run_id)


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
