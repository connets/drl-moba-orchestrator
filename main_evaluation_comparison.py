import argparse
import itertools
import os
import sqlite3
import ray

import numpy as np
import re

import pandas as pd
from gym.wrappers import FlattenObservation

from mec_moba.envs import MecMobaDQNEvn
from mec_moba.envs.utils.stepLoggerWrapper import StepLogger
from optimal_main import compute_optimal_solution
from mec_moba.policy_models.mip_online import compute_online_mip_solution
from mec_moba.policy_models.best_fit import compute_best_fit_solution
from eval_main import DqnAgent, RandomAgent, TestAgent
from collections import namedtuple
import tianshou as ts

RunPolicyInfo = namedtuple('RunPolicyInfo', ['run_id', 'training_year', 'policy_filepath'])


def get_reward_weights_from_run_id(run_id, experiment_tag):
    db = sqlite3.connect(os.path.join('out', experiment_tag, 'experiments.db'))
    cur = db.cursor().execute(f"SELECT reward_weights FROM main.experiments WHERE run_id = {run_id}")
    return tuple(map(float, cur.fetchone()[0][1:-1].split(',')))


def create_DQN_agent(run_info: RunPolicyInfo, experiment_tag):
    db = sqlite3.connect(os.path.join('out', experiment_tag, 'experiments.db'))
    run_train_parameters = pd.read_sql(f"SELECT * FROM main.experiments WHERE run_id = {run_info.run_id}", db)
    if 'layer_dim' in run_train_parameters.columns and 'num_layers' in run_train_parameters.columns:
        return DqnAgent(model_file=run_info.policy_filepath,
                        layer_dim=run_train_parameters.iloc[0].layer_dim,
                        num_layers=run_train_parameters.iloc[0].num_layers)

    return DqnAgent(model_file=run_info.policy_filepath)


@ray.remote
def compute_optimal_solution_wrapper(seed, evaluation_t_slot, num_weekly_matches,
                                     base_log_dir, match_probability_file, max_threads,
                                     skip_done):
    compute_optimal_solution(seed,
                             evaluation_t_slot=evaluation_t_slot,
                             base_log_dir=base_log_dir,
                             n_games_per_epoch=num_weekly_matches,
                             match_probability_file=match_probability_file,
                             max_threads=max_threads,
                             skip_done=skip_done)


@ray.remote
def compute_online_mip_solution_wrapper(seed, evaluation_t_slot, num_weekly_matches,
                                        base_log_dir, match_probability_file, max_threads,
                                        max_look_ahead_scheduler, cost_prediction, skip_done):
    compute_online_mip_solution(seed=seed,
                                evaluation_t_slot=evaluation_t_slot,
                                base_log_dir=base_log_dir,
                                n_games_per_epoch=num_weekly_matches,
                                match_probability_file=match_probability_file,
                                max_threads=max_threads,
                                cost_prediction=cost_prediction,
                                max_look_ahead_scheduler=max_look_ahead_scheduler,
                                skip_done=skip_done)


@ray.remote
def compute_best_fit_solution_wrapper(seed, evaluation_t_slot, num_weekly_matches,
                                      base_log_dir, match_probability_file, skip_done):
    compute_best_fit_solution(seed=seed,
                              evaluation_t_slot=evaluation_t_slot,
                              base_log_dir=base_log_dir,
                              n_games_per_epoch=num_weekly_matches,
                              match_probability_file=match_probability_file, skip_done=skip_done)


@ray.remote
def run_test(seed, agent: TestAgent, reward_weights, match_probability_file, base_log_dir,
             num_weekly_matches, t_slot_to_test=1008, gen_requests_until=1008,
             skip_done=False):
    # IF skip_done == True and log file exists --> return immediately
    if skip_done and os.path.exists(f'{base_log_dir}/eval_test_{agent.policy_type()}.csv'):
        return

    os.makedirs(base_log_dir, exist_ok=True)

    env = MecMobaDQNEvn(reward_weights=reward_weights,
                        gen_requests_until=gen_requests_until,
                        log_match_data=True,
                        match_probability_file=match_probability_file,
                        n_games_per_epoch=num_weekly_matches,
                        base_log_dir=base_log_dir)
    # env = Monitor(env)
    env = StepLogger(env, logfile=f'{base_log_dir}/eval_test_{agent.policy_type()}.csv')
    env = FlattenObservation(env)
    env.seed(seed)

    agent.initialize(env)
    collector = ts.data.Collector(agent.policy, env)
    collector.collect(n_step=t_slot_to_test)
    env.close()


def evaluate_dqn_policies_and_random(seed, experiment_tag, evaluation_t_slot, base_log_dir,
                                     num_weekly_matches, match_probability_file, eval_random_policy,
                                     rnd_repeat=20, skip_done=False):
    run_saved_model_dir_pattern = re.compile(r".*[/\\](?P<run_id>\d+)[/\\]saved_models$")
    saved_policy_pattern = re.compile(r"policy-(?P<year>\d+)\.pth")

    def to_run_policy_info(root, dirs, files):
        run_id = run_saved_model_dir_pattern.match(root).groupdict()['run_id']
        files = filter(lambda f: saved_policy_pattern.match(f), files)
        return (RunPolicyInfo(run_id=run_id,
                              training_year=saved_policy_pattern.match(f).groupdict()['year'],
                              policy_filepath=os.path.join(root, f))
                for f in files)

    run_policy_dirs = os.walk(os.path.join('out', experiment_tag))
    run_policy_dirs = filter(lambda e: run_saved_model_dir_pattern.match(e[0]), run_policy_dirs)
    run_policy_dirs = map(lambda e: to_run_policy_info(*e), run_policy_dirs)
    run_policy_dirs = list(itertools.chain.from_iterable(run_policy_dirs))

    unique_run_ids = set((run.run_id for run in run_policy_dirs))

    remote_ids = []

    # for run in progressbar.progressbar(run_policy_dirs, prefix='DQN policy '):
    for run in run_policy_dirs:
        remote_ids.append(run_test.remote(seed=seed,
                                          agent=create_DQN_agent(run, experiment_tag),  # DqnAgent(model_file=run.policy_filepath),
                                          t_slot_to_test=evaluation_t_slot + 6 * 5,
                                          reward_weights=get_reward_weights_from_run_id(run.run_id, experiment_tag),
                                          num_weekly_matches=num_weekly_matches,
                                          gen_requests_until=evaluation_t_slot,
                                          match_probability_file=match_probability_file,
                                          base_log_dir=os.path.join(base_log_dir, str(seed), 'dqn', f"{run.run_id}_{run.training_year}"),
                                          skip_done=skip_done))

        # RANDOM
    # for run_id, rnd_run in progressbar.progressbar(list(itertools.product(unique_run_ids, range(repeat))), prefix='Random policy '):
    if eval_random_policy:
        print('Random policy evaluation')
        for run_id, rnd_run in itertools.product(unique_run_ids, range(rnd_repeat)):
            remote_ids.append(run_test.remote(seed=seed,
                                              agent=RandomAgent(),
                                              t_slot_to_test=evaluation_t_slot + 6 * 5,
                                              reward_weights=get_reward_weights_from_run_id(run_id, experiment_tag),
                                              match_probability_file=match_probability_file,
                                              gen_requests_until=evaluation_t_slot,
                                              base_log_dir=os.path.join(base_log_dir, str(seed), 'rnd', f"{run_id}_{rnd_run}"),
                                              skip_done=skip_done))
    return remote_ids


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def run_comparison_main():
    parser = argparse.ArgumentParser(description="DRL MOBA Policy comparative evaluation")
    parser.add_argument('experiment_tag', type=str, help='A name of the training experiment setting')
    parser.add_argument('--evaluation-tag', type=str, default='', help='A name of this evaluation setting')
    parser.add_argument('--num-scenarios', type=int, default=10, help="Number of scenarios ")
    parser.add_argument('--rnd-repeats', type=int, default=10, help="Number of random repetitions")
    parser.add_argument('-j', type=int, default=os.cpu_count() - 4, help='Number of parallel processes', dest='num_processes')
    parser.add_argument('-g', type=int, default=4, help='Number of parallel gurobi processes', dest='num_gurobi_processes')
    parser.add_argument('--test-t-slot', type=int, default=144, help="Number of testing time slots")
    parser.add_argument('--seeds-file', default=None)
    parser.add_argument('--random-policy', action='store_true')
    parser.add_argument('--mip-online', action='store_true')
    parser.add_argument('--best-fit', action='store_true')
    parser.add_argument('--match-probability-file', default=None)
    parser.add_argument('--num-weekly-matches', type=int, default=6000)
    parser.add_argument('--mip-online-max_look_ahead_scheduler', type=int, default=6, help="Number of look ahead time slots")
    parser.add_argument('--mip-online-cost_prediction', action='store_true', help="whether or not assignment cost are 'predicted'")
    parser.add_argument('--skip-done', action='store_true')
    cli_args = parser.parse_args()

    num_ray_processes = cli_args.num_processes
    num_gurobi_processes = cli_args.num_gurobi_processes

    ray.init(num_cpus=num_ray_processes)


    #print(cli_args)
    evaluation_tag = f'eval-{cli_args.experiment_tag}-{cli_args.evaluation_tag}' if cli_args.evaluation_tag != '' else f'eval-{cli_args.experiment_tag}'
    base_log_dir = os.path.join('out_eval', evaluation_tag)
    os.makedirs(base_log_dir, exist_ok=True)

    rnd = np.random.default_rng()
    if cli_args.seeds_file is None:
        seeds = (rnd.integers(low=0, high=2 ** 32 - 1, size=cli_args.num_scenarios))
    else:
        seeds = pd.read_csv(cli_args.seeds_file, names=['seed'])['seed'].unique()

    remote_ids = []
    for seed in seeds:
        # EVALUATE POLICIES SOLUTION and EVALUATE RANDOM SOLUTION
        remote_ids.extend(evaluate_dqn_policies_and_random(seed,
                                                           experiment_tag=cli_args.experiment_tag,
                                                           evaluation_t_slot=cli_args.test_t_slot,
                                                           base_log_dir=base_log_dir,
                                                           num_weekly_matches=cli_args.num_weekly_matches,
                                                           match_probability_file=cli_args.match_probability_file,
                                                           rnd_repeat=cli_args.rnd_repeats,
                                                           eval_random_policy=cli_args.random_policy,
                                                           skip_done=cli_args.skip_done))
        # EVALUATE MIP ONLINE
        if cli_args.mip_online:
            remote_ids.append(compute_online_mip_solution_wrapper.remote(seed,
                                                                         evaluation_t_slot=cli_args.test_t_slot,
                                                                         num_weekly_matches=cli_args.num_weekly_matches,
                                                                         match_probability_file=cli_args.match_probability_file,
                                                                         base_log_dir=base_log_dir,
                                                                         max_look_ahead_scheduler=cli_args.mip_online_max_look_ahead_scheduler,
                                                                         cost_prediction=cli_args.mip_online_cost_prediction,
                                                                         max_threads=4, skip_done=cli_args.skip_done))

        if cli_args.best_fit:
            remote_ids.append(compute_best_fit_solution_wrapper.remote(seed, evaluation_t_slot=cli_args.test_t_slot,
                                                                       num_weekly_matches=cli_args.num_weekly_matches,
                                                                       match_probability_file=cli_args.match_probability_file,
                                                                       base_log_dir=base_log_dir,
                                                                       skip_done=cli_args.skip_done))

        # wait until finish
        ray.get(remote_ids)

        max_gurobi_threads = int(num_ray_processes / num_gurobi_processes)

        for seed_group in grouper(seeds, num_gurobi_processes):
            # COMPUTE OPTIMAL SOLUTION
            remote_ids = [compute_optimal_solution_wrapper.remote(seed,
                                                                  evaluation_t_slot=cli_args.test_t_slot,
                                                                  num_weekly_matches=cli_args.num_weekly_matches,
                                                                  match_probability_file=cli_args.match_probability_file,
                                                                  base_log_dir=base_log_dir, max_threads=max_gurobi_threads,
                                                                  skip_done=cli_args.skip_done)
                          for seed in seed_group if seed is not None]
            # wait until finish
            ray.get(remote_ids)


if __name__ == "__main__":
    run_comparison_main()
