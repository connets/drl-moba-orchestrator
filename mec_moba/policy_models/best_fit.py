import os

import numpy as np
from typing import List, Tuple

from mec_moba.environment.utils.delay_exctractor import *
import gurobipy as gp
import csv
from collections import namedtuple

from mec_moba.environment.matches import GameGenerator, Game

log_policy_name = 'best_fit'


def compute_assignment_cost(game, delay_dict, t, f):
    rtt = max(delay_dict[t, bs, f] for bs in game.get_base_stations())
    return (5 - game.compute_QoS(rtt)) / 5


def schedule_game_best_fit(game: Game, request_t_slot, facility_occupation_mat, delay_dict, max_facility_capacity) -> Tuple[int, int]:
    # assignment costs are available only for request t_slot and we assume no prediction,
    # so the costs for future t_slot are the same
    scheduling_t_slot = request_t_slot
    num_facilities = facility_occupation_mat.shape[0]
    scheduled = False

    sorted_facilities = sorted(range(num_facilities), key=lambda f: compute_assignment_cost(game, delay_dict, request_t_slot, f))

    while not scheduled:
        for f in sorted_facilities:
            # check if there is sufficient capacity for all the game duration
            potential_slots = facility_occupation_mat[f, scheduling_t_slot:scheduling_t_slot + game.get_duration()]
            potential_slots = map(lambda occ: occ < max_facility_capacity, potential_slots)
            if all(potential_slots):
                return f, scheduling_t_slot

        scheduling_t_slot += 1  # try in the next time-slot

    raise Exception('No assignment is possible')


def compute_best_fit_solution(seed, match_probability_file=None,
                              evaluation_t_slot=144, n_games_per_epoch=6000,
                              num_facilities=7, max_facility_capacity=12,
                              base_log_dir='logs', skip_done=False):
    if skip_done and os.path.exists(f'{base_log_dir}/{seed}/{log_policy_name}'):
        return

    facility_occupation_mat = np.zeros((num_facilities, evaluation_t_slot * 2))

    game_generator = GameGenerator(match_probability_file=match_probability_file, n_games_per_epoch=n_games_per_epoch)
    game_generator.set_seed(seed)
    game_generator.generate_epoch_matches()

    # extract_delay()
    info_physical_net = extract_delay()
    delay_dict = info_physical_net['delays']

    # log files
    os.makedirs(f'{base_log_dir}/{seed}/{log_policy_name}', exist_ok=True)
    fd = open(f'{base_log_dir}/{seed}/{log_policy_name}/match_data.csv', 'w', newline='')
    match_data_log = csv.writer(fd)
    match_data_log.writerow(['t_slot', 'match_id', 'facility_id'])

    for t in range(evaluation_t_slot):
        t_slot_game_requests = game_generator.get_match_requests(t)
        for g in t_slot_game_requests:
            facility, sched_t_slot = schedule_game_best_fit(g, request_t_slot=t,
                                                            facility_occupation_mat=facility_occupation_mat,
                                                            delay_dict=delay_dict,
                                                            max_facility_capacity=max_facility_capacity)

            # LOG
            for g_t in range(g.get_duration()):
                match_data_log.writerow([sched_t_slot + g_t, g.id, facility])
            # UPDATE OCCUPATION
            for ut in range(sched_t_slot, sched_t_slot + g.get_duration()):
                facility_occupation_mat[facility, ut] += 1
