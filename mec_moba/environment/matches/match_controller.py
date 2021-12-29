from __future__ import annotations

import typing
from heapq import *
from typing import Iterable, List
import math

import mec_moba.environment.utils.utils as utils

# if typing.TYPE_CHECKING:
from mec_moba.environment.matches import Game

MAX_MATCH_QUEUE_SIZE_PARAM = 'max_queue_size'
MAX_WAITING_TIME_VALUE_PARAM = 'max_waiting_time'

defaults = {MAX_MATCH_QUEUE_SIZE_PARAM: 100,
            MAX_WAITING_TIME_VALUE_PARAM: 4}


class MatchController:

    def __init__(self, environment, physical_net_ctl):
        # coda di attesa

        self.environment = environment
        self.physical_net_ctl = physical_net_ctl

        self.queue: List[Game] = list()
        self.drop_ratio = 0
        self.running: List = list()

        self._initial_status()

        # lunghezza massima della coda
        self.max_wait_size = defaults[MAX_MATCH_QUEUE_SIZE_PARAM]

    def _initial_status(self):
        self.queue: List[Game] = list()
        self.drop_ratio = 0
        self.running: List = list()

    # manda la lsita delle partite da migrare
    def get_migrate_and_blocked_list(self, d):
        migrable_vnf = []
        for match in self.get_games():
            current_qos = match.get_QoS()
            for j in range(self.physical_net_ctl.n_mec):
                # >= perchè in casi di sovraffollamento di una mec e le altre vuote potrebbe aver senso migrare
                # per distribuire il peso
                if min([match.compute_QoS(self.physical_net_ctl.get_rtt(bs, j)) for bs in
                        match.get_base_stations()]) > current_qos and j != match.get_facility_id():
                    migrable_vnf.append(match)
                    break

        to_ret = nsmallest(math.ceil(len(migrable_vnf) * d / 100), migrable_vnf, key=lambda v: v.get_QoS())

        return to_ret, [i for i in self.running if i not in to_ret]

    def get_migrable_ratio(self):
        if len(self.running) == 0:
            return 0
        migrable, _ = self.get_migrate_and_blocked_list(100)
        return len(migrable) / len(self.running)

    # manda la lista delle partite da deployare
    def get_deploy_list(self, deploy_value):
        current_t_slot = self.environment.absolute_t_slot
        return nsmallest(math.ceil(len(self.queue) * deploy_value / 100),
                         self.queue, key=lambda match: - match.get_queue_waiting_time(current_t_slot))

    def deploy(self, match, facility_id):
        rtt = self._get_match_facility_rtt(match, facility_id)
        match.deploy(facility_id, rtt, self.environment.absolute_t_slot)
        self.queue.remove(match)
        self.running.append(match)

    def migrate(self, match, old_facility_id, new_facility_id):
        new_rtt = self._get_match_facility_rtt(match, new_facility_id)
        match.migrate(new_facility_id, new_rtt)

    def _get_match_facility_rtt(self, match, facility_id):
        return max([self.physical_net_ctl.get_rtt(k, facility_id) for k in match.get_base_stations()])

    def update_qos(self):
        for game in self.running:
            game.update_QoS(self._get_match_facility_rtt(game, game.get_facility_id()))

    def cleanup_terminated_matches(self, t_slot_end: bool):
        current_t_slot = self.environment.absolute_t_slot
        next_t_slot_running = []
        for match in self.running:
            if match.is_terminated(current_t_slot, t_slot_end):
                self.physical_net_ctl.terminate_match(match, match.get_facility_id())
            else:
                next_t_slot_running.append(match)

        # assert all(map(lambda m: not m.is_terminated(current_t_slot, t_slot_end), next_t_slot_running))
        # for m in self.running:
        #     if m not in next_t_slot_running:
        #         assert current_t_slot - m.deploy_time >= m.duration

        self.running = next_t_slot_running

    def get_actions_infos(self, c, d):
        return [self.get_migrate_and_blocked_list(c), self.get_deploy_list(d)]

    def get_games(self):
        return sorted(self.running, key=lambda match: match.get_QoS())

    def get_games_per_mec(self, i):
        return [j for j in self.running if i.get_facility_id() == i]

    def enqueue_match_requests(self, match_requests: List[Game]):
        current_t_slot = self.environment.absolute_t_slot
        dropped_match_requests = 0
        for match in match_requests:
            if len(self.queue) < self.max_wait_size:
                self.queue.append(match)  # += [i]
                match.enqueue(current_t_slot)
            else:
                dropped_match_requests += 1

        # assert all(map(lambda m: m in self.queue, match_requests))

        self.drop_ratio = dropped_match_requests / len(match_requests) if len(match_requests) > 0 else 0

    def get_dummy_qos_state(self) -> List[float]:
        return [0.0] * Game.get_num_qos_level()

    def get_qos_state(self) -> List[float]:
        tmp_qos = self.get_dummy_qos_state()
        if len(self.running) == 0:
            return tmp_qos

        for match in self.running:
            tmp_qos[match.get_QoS()] += 1

        return list(map(lambda a: a / len(self.running), tmp_qos))

    def get_mean_qos_running_instances(self) -> float:
        if len(self.running) == 0:
            return -1

        return sum([5 - m.get_QoS() for m in self.running]) / len(self.running)

    def get_mean_running_time_state(self) -> float:
        current_t_slot = self.environment.absolute_t_slot
        return sum([r.get_time_running(current_t_slot) / r.get_duration() for r in self.running]) / len(self.running) if len(self.running) > 0 else 0

    def get_queue_occupation_state(self):
        return len(self.queue) / self.max_wait_size

    def get_mean_queue_waiting_time(self):
        current_t_slot = self.environment.absolute_t_slot
        ret_value = map(lambda e: min(defaults[MAX_WAITING_TIME_VALUE_PARAM], utils.exp_lin(e.get_queue_waiting_time(current_t_slot) - e.get_max_wait())), self.queue)
        return sum(ret_value) / len(self.queue) / defaults[MAX_WAITING_TIME_VALUE_PARAM] if len(self.queue) > 0 else 0

    def get_queue_drop_rate(self):
        return self.drop_ratio

    def change_epoch(self):
        self._initial_status()
