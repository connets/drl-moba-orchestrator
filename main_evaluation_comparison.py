import argparse
import itertools
import os
import sqlite3

import numpy as np
import re

import pandas as pd
import progressbar
from gym.wrappers import FlattenObservation

from mec_moba.envs import MecMobaDQNEvn
from mec_moba.envs.utils.stepLoggerWrapper import StepLogger
from optimal_main import compute_optimal_solution
from eval_main import DqnAgent, RandomAgent, TestAgent
from collections import namedtuple
import tianshou as ts

RunPolicyInfo = namedtuple('RunPolicyInfo', ['run_id', 'training_year', 'policy_filepath'])


def get_reward_weights_from_run_id(run_id, experiment_tag):
    db = sqlite3.connect(os.path.join('out', experiment_tag, 'experiments.db'))
    cur = db.cursor().execute(f"SELECT reward_weights FROM main.experiments WHERE run_id = {run_id}")
    return tuple(map(float, cur.fetchone()[0][1:-1].split(',')))


def run_test(seed, agent: TestAgent, reward_weights, base_log_dir, t_slot_to_test=1008, gen_requests_until=1008):
    # print(reward_weights, base_log_dir)
    os.makedirs(base_log_dir, exist_ok=True)

    env = MecMobaDQNEvn(reward_weights=reward_weights,
                        gen_requests_until=gen_requests_until,
                        log_match_data=True,
                        base_log_dir=base_log_dir)
    # env = Monitor(env)
    env = StepLogger(env, logfile=f'{base_log_dir}/eval_test_{agent.policy_type()}.csv')
    env = FlattenObservation(env)
    env.seed(seed)

    agent.initialize(env)
    collector = ts.data.Collector(agent.policy, env)
    collector.collect(n_step=t_slot_to_test)


def evaluate_dqn_policies_and_random(seed, experiment_tag, evaluation_t_slot, base_log_dir, repeat=20):
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

    for run in progressbar.progressbar(run_policy_dirs, prefix='DQN policy '):
        run_test(seed=seed,
                 agent=DqnAgent(model_file=run.policy_filepath),
                 t_slot_to_test=evaluation_t_slot + 12 * 6,
                 reward_weights=get_reward_weights_from_run_id(run.run_id, experiment_tag),
                 gen_requests_until=evaluation_t_slot,
                 base_log_dir=os.path.join(base_log_dir, str(seed), 'dqn', f"{run.run_id}_{run.training_year}"))

        # RANDOM
    for run_id, rnd_run in progressbar.progressbar(list(itertools.product(unique_run_ids, range(repeat))), prefix='Random policy '):
        run_test(seed=seed,
                 agent=RandomAgent(),
                 t_slot_to_test=evaluation_t_slot + 12 * 6,
                 reward_weights=get_reward_weights_from_run_id(run_id, experiment_tag),
                 gen_requests_until=evaluation_t_slot,
                 base_log_dir=os.path.join(base_log_dir, str(seed), 'rnd', f"{run_id}_{rnd_run}"))


def run_comparison_main():
    parser = argparse.ArgumentParser(description="DRL MOBA Policy comparative evaluation")
    parser.add_argument('experiment_tag', type=str, help='A name of the training experiment setting')
    parser.add_argument('evaluation_tag', type=str, help='A name of this evaluation setting')
    parser.add_argument('--num-scenarios', type=int, default=10, help="Number of scenarios ")
    parser.add_argument('--rnd-repeats', type=int, default=10, help="Number of random repetitions")
    # parser.add_argument('-j', type=int, default=os.cpu_count() - 1, help='Number of parallel processes', dest='num_processes')
    parser.add_argument('--test-t-slot', type=int, default=144, help="Number of testing time slots")
    parser.add_argument('--seeds-file', default=None)
    cli_args = parser.parse_args()

    base_log_dir = os.path.join('out_eval', cli_args.evaluation_tag)
    os.makedirs(base_log_dir, exist_ok=True)

    rnd = np.random.default_rng()
    if cli_args.seeds_file is None:
        seeds = (rnd.integers(low=0, high=2 ** 32 - 1, size=cli_args.num_scenarios))
    else:
        seeds = pd.read_csv(cli_args.seeds_file, names=['seed'])['seed'].unique()

    for seed in seeds:
        # COMPUTE OPTIMAL SOLUTION
        compute_optimal_solution(seed,
                                 evaluation_t_slot=cli_args.test_t_slot,
                                 base_log_dir=base_log_dir)
        # EVALUATE POLICIES SOLUTION and EVALUATE RANDOM SOLUTION
        evaluate_dqn_policies_and_random(seed,
                                         experiment_tag=cli_args.experiment_tag,
                                         evaluation_t_slot=cli_args.test_t_slot,
                                         base_log_dir=base_log_dir,
                                         repeat=cli_args.rnd_repeats)


if __name__ == "__main__":
    run_comparison_main()
