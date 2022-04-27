import os
from typing import List

import numpy as np
from mec_moba.environment.utils.delay_exctractor import *
import gurobipy as gp
import csv
from collections import namedtuple

from mec_moba.environment.matches import GameGenerator

# RunPolicyInfo = namedtuple('RunPolicyInfo', ['run_id', 'training_year', 'policy_filepath'])
NewMatches = namedtuple('NetMatches', ['match_id', 'match_obj'])
ScheduledMatches = namedtuple('ScheduledMatches', ['match_id', 'match_obj', 'scheduled_t_slot', 'facilities_list'])
RunningMatches = namedtuple('RunningMatches', ['match_id', 'match_obj', 'scheduled_t_slot', 'facilities_list'])


def solve_t_slot_mip_instance(current_t_slot,
                              new_matches, scheduled_matches, running_matches,
                              delay_dict,
                              max_look_ahead_scheduler,
                              migrate_running: bool = False,
                              cost_prediction: bool = False,
                              num_facilities=7, match_duration=6):
    N = len(new_matches)
    assignable_matches = [m.match_obj for m in new_matches]
    if migrate_running:
        N += len(running_matches)
        assignable_matches.extend(m.match_obj for m in running_matches)

    F = num_facilities
    T = max_look_ahead_scheduler + match_duration  # check it

    # cost matrix
    cost_c = np.zeros((N, F, T))
    for n in range(N):
        for f in range(F):
            for t in range(T):
                cost_t_slot = current_t_slot + t if cost_prediction else current_t_slot
                # t_slot instead of t because in the online version we don't know future costs,
                # so we assume that costs do not change over time
                rtt = max(delay_dict[cost_t_slot, bs, f] for bs in assignable_matches[n].get_base_stations())
                cost_c[n, f, t] = (5 - assignable_matches[n].compute_QoS(rtt)) / 5

    # scheduling costs only for new matches
    delta = match_duration
    w = np.zeros((N, T))
    for m in range(N):
        for t in range(T):
            w[m, t] = t  # max(0, t - request_time_sigma[n])

    # reserved capacity
    R = np.zeros((F, T))
    for m in scheduled_matches:
        relative_sched_t = m.scheduled_t_slot - current_t_slot
        for f, t in zip(m.facilities_list, range(relative_sched_t, min(relative_sched_t + match_duration, T))):
            R[f, t] += 1  # unity of capacity that will be used by future match
    # print(len(running_matches), len(scheduled_matches), N, R)
    # if running instance are not allowed to migrate they use capacity which is not available to other instances
    for m in running_matches:
        time_to_go = match_duration - (current_t_slot - m.scheduled_t_slot)
        for f, t in zip(m.facilities_list, range(0, time_to_go)):
            R[f, t] += 1

    # print(len(running_matches), len(scheduled_matches), N, R)
    m = gp.Model("DQN_online")
    m.Params.LogToConsole = 0
    m.Params.MIPGap = 0.005
    m.Params.TimeLimit = 3000
    # m.Params.Threads =

    # Create variables
    x = m.addMVar((N, F, T), vtype=gp.GRB.BINARY, name="X")
    v = m.addMVar((F, T), lb=0, vtype=gp.GRB.INTEGER, name="v")
    y = m.addMVar((N, T), vtype=gp.GRB.BINARY, name="Y")
    s = m.addMVar((N, T), vtype=gp.GRB.BINARY, name="S")

    # Set objective
    m.setObjective(sum(sum(sum([cost_c[i, j, t] * x[i, j, t] for j in range(F)]) for i in range(N)) for t in range(T))
                   + sum(sum([1 * v[j, t] for j in range(F)]) for t in range(T))
                   + sum(sum([1 / 5 * y[i, t] for i in range(N)]) for t in range(T))
                   + sum(sum(w[i, t] * s[i, t] for i in range(N)) for t in range(T)),
                   gp.GRB.MINIMIZE)

    # Set constrains

    #
    m.addConstrs(
        (sum([x[i, j, t] for j in range(F)]) == sum([s[i, _t] for _t in range(max(0, t - delta + 1), t + 1)])
         for i in range(N) for t in range(T)),
        name='c0')

    # m.addConstrs(
    #     (sum([x[i, j, t] for j in range(F) for t in range(T)]) == delta for i in range(N)), name='c_a')

    m.addConstrs((sum([s[i, t] for t in range(0, T - delta)]) == 1 for i in range(N)),
                 name='c1')

    # m.addConstrs((sum([s[i, t] for t in range(0, request_time_sigma[i])]) == 0 for i in range(N)),
    #              name='c1_2')
    #
    # m.addConstrs((sum([s[i, t] for t in range(T - delta, T)]) == 0 for i in range(N)),
    #              name='c1_3')

    m.addConstrs((x[i, j, t + 1] - x[i, j, t] - (1 - sum(x[i, j_, t] for j_ in range(F))) <= y[i, t] for i in range(N) for j in range(F) for t in range(T - 1)),
                 name='c2')
    # m.addConstrs((x[i, j, t + 1] - x[i, j, t] <= y[i, t] for i in range(N) for j in range(F) for t in range(T - 1)),
    #              name='c2')

    #
    m.addConstrs(((sum([x[i, j, t] for i in range(N)]) <= 8 + v[j, t] - R[j, t]) for j in range(F) for t in range(T)),
                 name='c3')

    m.addConstrs(((v[j, t] <= 4) for j in range(F) for t in range(T)),
                 name='c4')

    m.update()
    m.optimize()

    if not hasattr(m, 'objVal'):
        print('no solution gurobi, there is a None Type', current_t_slot)

    else:
        # return scheduling plan
        new_matches_scheduled = []
        for i in range(N):
            for t in range(T):
                if round(s[i][t].x) > 0:
                    schedule_t_slot = t + current_t_slot  # to absolute
                    facilities_list = []
                    for tt in range(t, t + match_duration):
                        selected_f = []  # for test
                        for j in range(F):
                            if round(x[i][j][tt].x) > 0:
                                # print(i,j,tt)
                                selected_f.append(j)
                        assert len(selected_f) == 1
                        facilities_list.extend(selected_f)
                    assert len(facilities_list) == match_duration
                    break
            new_matches_scheduled.append(ScheduledMatches(match_id=new_matches[i].match_id,
                                                          match_obj=new_matches[i].match_obj,
                                                          scheduled_t_slot=schedule_t_slot,
                                                          facilities_list=facilities_list))
        return new_matches_scheduled
        # writer.writerow([t, j, i, x[i][j][t].x])


def compute_online_mip_solution(seed, match_probability_file=None,
                                evaluation_t_slot=144, n_games_per_epoch=6000,
                                base_log_dir='logs', max_threads=0,
                                cost_prediction=False,
                                max_look_ahead_scheduler=6, skip_done=False):
    log_policy_name = 'mip_online_cost_prediction' if cost_prediction else 'mip_online'

    if skip_done and os.path.exists(f'{base_log_dir}/{seed}/{log_policy_name}'):
        # print(seed, log_policy_name, 'has already done')
        return

    T_SLOTS = evaluation_t_slot
    T = T_SLOTS + 6  # * 12
    F = 7
    MATCH_DURATION = 6
    SEED = seed
    MAX_LOOK_AHEAD_T_SLOTS = max_look_ahead_scheduler

    game_generator = GameGenerator(match_probability_file=match_probability_file, n_games_per_epoch=n_games_per_epoch)
    game_generator.set_seed(SEED)
    game_generator.generate_epoch_matches()

    # extract_delay()
    info_physical_net = extract_delay()
    delay_dict = info_physical_net['delays']

    # log files
    os.makedirs(f'{base_log_dir}/{SEED}/{log_policy_name}', exist_ok=True)
    fd = open(f'{base_log_dir}/{SEED}/{log_policy_name}/match_data.csv', 'w', newline='')
    match_data_log = csv.writer(fd)
    match_data_log.writerow(['t_slot', 'match_id', 'facility_id'])

    scheduled_matches: List[ScheduledMatches] = list()
    running_matches: List[RunningMatches] = list()
    for t in range(T):
        if t < T_SLOTS:
            new_match_requests = [NewMatches(g.id, g) for g in game_generator.get_match_requests(t)]
            new_matches_scheduled = solve_t_slot_mip_instance(current_t_slot=t,
                                                              new_matches=new_match_requests,
                                                              scheduled_matches=scheduled_matches,
                                                              running_matches=running_matches,
                                                              delay_dict=delay_dict,
                                                              cost_prediction=cost_prediction,
                                                              max_look_ahead_scheduler=max_look_ahead_scheduler)
            for m in new_matches_scheduled:
                for i, f in enumerate(m.facilities_list):
                    match_data_log.writerow([m.scheduled_t_slot + i, m.match_id, f])

            scheduled_matches += new_matches_scheduled

        tmp_scheduled_matches = []
        # scheduled --> running
        for m in scheduled_matches:
            if m.scheduled_t_slot == t:
                running_matches.append(RunningMatches(*m))
                # print(m, RunningMatches(*m))
            else:
                tmp_scheduled_matches.append(m)
        scheduled_matches = tmp_scheduled_matches

        # running --> next step
        running_matches = [m._replace(facilities_list=m.facilities_list[1:] if len(m.facilities_list) > 1 else []) for m in running_matches]

        # running --> finished
        running_matches = list(filter(lambda m: len(m.facilities_list) > 0, running_matches))
        # print(len(running_matches), running_matches[0])

    fd.close()
