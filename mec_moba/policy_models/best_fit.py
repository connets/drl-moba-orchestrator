import itertools
import os
import time

import numpy as np
from typing import Tuple

from mec_moba.environment.utils.delay_exctractor import *
import csv

from mec_moba.environment.matches import GameGenerator, Game

log_policy_name = 'best_fit'


def compute_assignment_cost(game, delay_dict, t, f):
    rtt = max(delay_dict[t, bs, f] for bs in game.get_base_stations())
    return (5 - game.compute_QoS(rtt)) / 5


def compute_object_function(game, delay_dict, request_t, sched_t, f, facility_occupation_mat, op_limit=8):
    overprovisioning_cost = sum(
        map(lambda occ: max(occ - op_limit, 0), facility_occupation_mat[f, sched_t: sched_t + game.get_duration()]))
    assignment_cost = compute_assignment_cost(game, delay_dict, request_t, f)
    scheduling_cost = sched_t - request_t
    return assignment_cost + overprovisioning_cost + scheduling_cost


def schedule_game_best_fit(game: Game,
                           request_t_slot,
                           facility_occupation_mat,
                           delay_dict,
                           max_facility_capacity,
                           max_look_ahead_scheduler) -> Tuple[int, int]:
    # assignment costs are available only for request t_slot and we assume no prediction,
    # so the costs for future t_slot are the same
    scheduling_t_slot = request_t_slot
    num_facilities = facility_occupation_mat.shape[0]
    scheduled = False

    while not scheduled:
        sorted_potential_solutions = itertools.product(range(num_facilities),
                                                       range(scheduling_t_slot,
                                                             scheduling_t_slot + max_look_ahead_scheduler))
        sorted_potential_solutions = sorted(sorted_potential_solutions,
                                            key=lambda e: compute_object_function(game,
                                                                                  delay_dict,
                                                                                  request_t_slot,
                                                                                  e[1],
                                                                                  e[0],
                                                                                  facility_occupation_mat))
        for f, t in sorted_potential_solutions:
            # check if there is sufficient capacity for all the game duration
            potential_slots = facility_occupation_mat[f, t:t + game.get_duration()]
            potential_slots = map(lambda occ: occ < max_facility_capacity, potential_slots)
            if all(potential_slots):
                return f, t

        #print('Solution not found! go next!')
        scheduling_t_slot += max_look_ahead_scheduler  # try in the next time-slot

    raise Exception('No assignment is possible')


def compute_best_fit_solution(seed, match_probability_file=None,
                              evaluation_t_slot=144, n_games_per_epoch=6000,
                              num_facilities=7, max_facility_capacity=12, max_look_ahead_scheduler=6,
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

    fd_timelog = open(f'{base_log_dir}/{seed}/{log_policy_name}/time_log.csv', 'w', newline='')
    fd_timelog.write('excution_time\n')

    for t in range(evaluation_t_slot):
        t_slot_game_requests = game_generator.get_match_requests(t)
        s_time = time.time()
        for g in t_slot_game_requests:

            facility, sched_t_slot = schedule_game_best_fit(g, request_t_slot=t,
                                                            facility_occupation_mat=facility_occupation_mat,
                                                            delay_dict=delay_dict,
                                                            max_facility_capacity=max_facility_capacity,
                                                            max_look_ahead_scheduler=max_look_ahead_scheduler)

            # LOG
            for g_t in range(g.get_duration()):
                match_data_log.writerow([sched_t_slot + g_t, g.id, facility])
            # UPDATE OCCUPATION
            for ut in range(sched_t_slot, sched_t_slot + g.get_duration()):
                facility_occupation_mat[facility, ut] += 1

        fd_timelog.write(f'{time.time() - s_time}\n')
