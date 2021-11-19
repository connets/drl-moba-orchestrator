from __future__ import annotations

import typing
from heapq import *
from typing import Iterable, List

import mec_moba.environment.utils.utils as utils

# if typing.TYPE_CHECKING:
from mec_moba.environment.matches import Game

MAX_MATCH_QUEUE_SIZE_PARAM = 'max_queue_size'
MAX_WAITING_TIME_VALUE_PARAM = 'max_waiting_time'

defaults = {MAX_MATCH_QUEUE_SIZE_PARAM: 100,
            MAX_WAITING_TIME_VALUE_PARAM: 4}


class MatchController:

    # @staticmethod
    # def get_module_config_name() -> str:
    #     return 'match_controller'
    #
    # @staticmethod
    # def get_module_config_options() -> Iterable[ConfigOption]:
    #     return [ConfigOption(name=MAX_MATCH_QUEUE_SIZE_PARAM, default_value=100, help_string='Max match queue size'),
    #             ConfigOption(name=MAX_WAITING_TIME_VALUE_PARAM, default_value=4, cli_type=int)]

    def __init__(self, environment, physical_net_ctl):
        # coda di attesa
        self.queue: List[Game] = list()
        self.drop_ratio = 0
        self.environment = environment
        self.physical_net_ctl = physical_net_ctl

        # coda running
        self.running: List = list()

        # lunghezza massima della coda
        self.max_wait_size = defaults[MAX_MATCH_QUEUE_SIZE_PARAM]
        self.n_migration = 0

    # manda la lsita delle partite da migrare
    def get_migrate_and_blocked_list(self, d):
        migrable_vnf = []
        for match in self.get_games():
            current_qos = match.get_QoS()
            for j in range(self.physical_net_ctl.n_mec):
                # >= perchè in casi di sovraffollamento di una mec e le altre vuote potrebbe aver senso migrare
                # per distribuire il peso
                if min([match.compute_QoS(self.physical_net_ctl.get_rtt(bs, j)) for bs in
                        match.get_base_stations()]) >= current_qos and j != match.get_facility_id():
                    migrable_vnf.append(match)
                    break
        # count = len(migrable_vnf)
        # self.migrable_vnf = sorted(self.migrable_vnf, key=lambda v: v.get_QoS())
        # return count / len(self.running)

        to_ret = nsmallest(round(len(migrable_vnf) * d / 100), migrable_vnf, key=lambda v: v.get_QoS())

        return to_ret, [i for i in self.running if i not in to_ret]

    def get_migrable_ratio(self):
        if len(self.running) == 0:
            return 0
        migrable, _ = self.get_migrate_and_blocked_list(100)
        return len(migrable) / len(self.running)

    # manda la lista delle partite da deployare
    def get_deploy_list(self, deploy_value):
        current_t_slot = self.environment.absolute_t_slot
        return nsmallest(round(len(self.queue) * deploy_value / 100),
                         self.queue, key=lambda match: - match.get_queue_waiting_time(current_t_slot))

    def deploy(self, match, facility_id):
        rtt = self._get_match_facility_rtt(match, facility_id)
        match.deploy(facility_id, rtt, self.environment.absolute_t_slot)
        self.queue.remove(match)
        self.running.append(match)

    def migrate(self, match, old_facility_id, new_facility_id):
        new_rtt = self._get_match_facility_rtt(match, new_facility_id)
        match.migrate(new_facility_id, new_rtt)

    # def remove_match(self, n: int):
    #     try:
    #         tmp = self.running.pop(n)
    #         self.physical_net_ctl.set_mec_status(tmp.get_facility_id(), -1)
    #         return True
    #     except IndexError:
    #         print("match controller index error")
    #         return False

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

    def get_mean_running_time_state(self) -> float:
        current_t_slot = self.environment.absolute_t_slot
        return sum([r.get_time_running(current_t_slot) / r.get_duration() for r in self.running]) / len(self.running) if len(self.running) > 0 else 0

    def get_queue_occupation_state(self):
        return len(self.queue) / self.max_wait_size

    def get_mean_queue_waiting_time(self):
        current_t_slot = self.environment.absolute_t_slot
        return sum(map(lambda e: min(4, utils.exp_lin(e.get_queue_waiting_time(current_t_slot) - e.get_max_wait())), self.queue), ) / len(self.queue) if len(self.queue) > 0 else 0

    def get_queue_drop_rate(self):
        return self.drop_ratio

    # def get_migrations(self):
    #     to_ret = self.n_migration
    #     self.n_migration = 0
    #     return to_ret

    def _save(self) -> typing.Dict[str, typing.Any]:
        return {self.__class__.__name__: {'queue': self.queue,
                                          'running': self.running}}

    def _restore(self, saved_objects: typing.Dict):
        self.queue = saved_objects[self.__class__.__name__]['queue']
        self.running = saved_objects[self.__class__.__name__]['running']

        # deploy running in physical network to restore everything
        for match in self.running:
            self.physical_net_ctl.deploy(match, match.get_facility_id)
