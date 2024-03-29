import os
from time import time
import sqlite3

import argparse
import yaml
from multiprocessing import Pool

from gym.wrappers import FlattenObservation
from mec_moba.envs.utils.rewardComponentsLoggerWrapper import RewardComponentLogger
from collections import namedtuple
import itertools

import torch, math
import tianshou as ts

from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

from mec_moba.envs import MecMobaDQNEvn
from mec_moba.policy_models.mlp_policy import MLPNet

grid_search_params = ['buffer_size',
                      'target_update_interval',
                      'train_each_n_step',
                      'gamma',
                      'train_eps',
                      'batch_size',
                      'learning_rate',
                      'reward_weights',
                      'layer_dim',
                      'num_layers', ]

run_parameter_fields_to_save = ['run_id', 'train_years', 'seed'] + grid_search_params
extra_run_parameter_fields = ['base_dir', 'match_prob_file']  # these extras fields are needed internally but are not saved
run_parameter_fields = run_parameter_fields_to_save + extra_run_parameter_fields

sqlite_field_type_dict = {'run_id': 'text',
                          'train_years': 'integer',
                          'seed': 'integer',
                          'buffer_size': 'integer',
                          'target_update_interval': 'integer',
                          'train_each_n_step': 'integer',
                          'gamma': 'real',
                          'train_eps': 'real',
                          'batch_size': 'integer',
                          'learning_rate': 'real',
                          'reward_weights': 'text',
                          'layer_dim': 'integer',
                          'num_layers': 'integer',
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


class Experiment:
    def __init__(self, run_params: RunParameters):
        self.run_params = run_params
        self.current_training_year = 0
        self.env_train_step = 0

        # INITIALIZATION
        self.model_save_dir = os.path.join(run_params.base_dir, run_params.run_id, 'saved_models')
        os.makedirs(self.model_save_dir)

        tb_log_dir = os.path.join(run_params.base_dir, 'dqn_mec_moba_tensorboard')

        train_env = MecMobaDQNEvn(reward_weights=run_params.reward_weights,
                                  match_probability_file=run_params.match_prob_file,
                                  return_reward_components=True)
        train_env = RewardComponentLogger(train_env,
                                          logfile=os.path.join(run_params.base_dir, run_params.run_id,'train_reward_log.csv'),
                                          compressed=True)
        train_env = FlattenObservation(train_env)

        test_env = MecMobaDQNEvn(reward_weights=run_params.reward_weights)
        test_env = FlattenObservation(test_env)

        state_shape = train_env.observation_space.shape or train_env.observation_space.n
        action_shape = train_env.action_space.shape or train_env.action_space.n
        net = MLPNet(state_shape, action_shape, run_params.layer_dim, run_params.num_layers)
        optim = torch.optim.Adam(net.parameters(), lr=run_params.learning_rate)

        self.policy = ts.policy.DQNPolicy(net, optim, discount_factor=run_params.gamma,
                                          estimation_step=1, target_update_freq=run_params.target_update_interval)
        replay_buffer = ts.data.PrioritizedReplayBuffer(run_params.buffer_size, alpha=0.6, beta=0.2)

        self.train_collector = ts.data.Collector(self.policy, train_env, replay_buffer)
        self.test_collector = ts.data.Collector(self.policy, test_env)
        self.tb_logger = TensorboardLogger(SummaryWriter(os.path.join(tb_log_dir, run_params.run_id)))
        print(f'Created run id {run_params.run_id} object\n{run_params}')

    def is_done(self):
        return self.current_training_year >= self.run_params.train_years

    def save_policy_fn(self):
        torch.save(self.policy.state_dict(), f'{self.model_save_dir}/policy-{self.current_training_year}.pth')

    def train_epoch(self):
        # for sanity
        if self.is_done():
            return
        # if this is the first epoch collect 10 weeks before starting the training
        if self.current_training_year == 0:
            self.train_collector.collect(n_episode=10, random=True)

        s_time = time()

        self.policy.set_eps(self.run_params.train_eps)
        for week in range(52):
            for i in range(int(1008 / self.run_params.train_each_n_step)):
                collect_result = self.train_collector.collect(n_step=self.run_params.train_each_n_step)
                self.env_train_step += int(collect_result["n/st"])
                update_results = self.policy.update(self.run_params.batch_size, self.train_collector.buffer)
                self.tb_logger.log_update_data(update_results, self.env_train_step)
                self.tb_logger.log_train_data(collect_result, self.env_train_step)

        print(f'RUN {self.run_params.run_id} - Training year {self.current_training_year} done in {round(time() - s_time, 1)} s')
        # TEST
        s_time = time()
        self.policy.set_eps(0)
        test_result = self.test_collector.collect(n_episode=10)
        self.tb_logger.log_test_data(test_result, self.current_training_year)
        self.save_policy_fn()
        print(f'RUN {self.run_params.run_id} - Testing year {self.current_training_year}: Test mean returns: {test_result["rews"].mean()} in {round(time() - s_time, 1)} s')

        self.current_training_year += 1


def training_processes(experiments_group):
    experiments_obj = [Experiment(e) for e in experiments_group if e is not None]
    experiment_to_do_idx = 0
    while not all([e.is_done() for e in experiments_obj]):
        experiments_obj[experiment_to_do_idx].train_epoch()
        experiment_to_do_idx = (experiment_to_do_idx + 1) % len(experiments_obj)


def _generate_run_parameter(run_id, base_dir, match_prob_file, train_years, seed, grid_params_dict):
    d = {'run_id': str(run_id),
         'base_dir': base_dir,
         'train_years': train_years,
         'seed': seed,
         'match_prob_file': match_prob_file}
    d.update(**grid_params_dict)

    return RunParameters(**d)


def main():
    parser = argparse.ArgumentParser(description="DRL MOBA on Stable Baseline 3")
    parser.add_argument('conf_file', type=str, help='Grid search configurator file (.yaml)')
    parser.add_argument('experiment_tag', type=str, help='A name of this experiment setting')
    parser.add_argument('--seed', type=int, default=-1, help="Seed number. -1 or negative values means random seed ")
    parser.add_argument('--match-probability-file', default=None)
    parser.add_argument('-j', type=int, default=os.cpu_count() - 1, help='Number of parallel processes', dest='num_processes')
    parser.add_argument('--train-years', type=int, default=10, help="Number of training weeks")
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
    train_years = cli_args.train_years

    # Grid search
    grid_search_config = yaml.load(stream=open(cli_args.conf_file, 'r'), Loader=yaml.SafeLoader)

    # pprint.pprint(list((grid_search_config[p] for p in grid_search_params)))

    experiments = itertools.product(*(grid_search_config[p] for p in grid_search_params))
    experiments = map(lambda v: dict(zip(grid_search_params, v)), experiments)
    experiments = (_generate_run_parameter(run_id=i,
                                           base_dir=base_dir,
                                           match_prob_file=cli_args.match_probability_file,
                                           train_years=train_years,
                                           seed=seed,
                                           grid_params_dict=e)
                   for i, e in enumerate(experiments))

    experiments = list(experiments)
    insert_all_runs(db_conn=db_connection, experiments=experiments)

    def grouper(iterable, n, fillvalue=None):
        args = [iter(iterable)] * n
        return itertools.zip_longest(*args, fillvalue=fillvalue)

    grp_experiments = grouper(experiments, n=math.ceil(len(experiments) / cli_args.num_processes))

    with Pool(processes=cli_args.num_processes) as pool:
        pool.map(training_processes, grp_experiments)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
