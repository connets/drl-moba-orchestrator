# stotituire bs con giocatore. giocatore avrà la location (BS)

import itertools
import math
from typing import Optional


class Game(object):
    NORMAL_TYPE = 0
    PREMIUM_TYPE = 1
    _qos_levels = {NORMAL_TYPE: [(5, 0), (10, 1), (15, 2), (20, 3), (25, 4), (math.inf, 5)],
                   PREMIUM_TYPE: [(10, 0), (20, 1), (30, 2), (40, 3), (50, 4), (math.inf, 5)]}

    game_id = itertools.count(0)

    def __init__(self, game_id, duration, resources_cost, group: list, max_wait, queue_abandon_time=-1, game_type=NORMAL_TYPE):
        # identificativo unico
        self.game_id: int = game_id
        # durata di una partita
        self.duration = duration
        self.resources_cost = resources_cost
        # gruppo di Users class
        self.group = group
        self.mec_facility_id: Optional[int] = None
        self.qos = -1  # error if it has never set

        # masismo tmepo di attes
        self.max_wait = max_wait
        self.enqueue_time = None

        self.current_wait = 0

        # tempo passato
        self.deploy_time = None
        # self.time_running = 0

        self.queue_abandon_time = queue_abandon_time

        self.game_type = game_type

    @property
    def id(self):
        return self.game_id

    def get_id(self):
        return self.game_id

    # def increase_wait(self, t=1):
    #     self.current_wait += t

    def exit_queue(self):
        if self.queue_abandon_time < 0:
            return False
        return self.max_wait < self.current_wait and self.queue_abandon_time < abs(self.max_wait - self.current_wait)

    # def set_facility(self, f):
    #     self.mec = f

    def get_facility_id(self):
        return self.mec_facility_id

    def get_base_stations(self):
        return [user.get_bs() for user in self.group]

    def enqueue(self, t_slot):
        self.enqueue_time = t_slot  # environment.get_env_absolute_t_slot_time()

    def deploy(self, facility_id: int, rtt: float, t_slot: int):
        self.deploy_time = t_slot  # environment.get_env_absolute_t_slot_time()
        self.mec_facility_id = facility_id
        self.qos = self.compute_QoS(rtt)

    def migrate(self, new_facility_id: int, new_rtt: float):
        self.mec_facility_id = new_facility_id
        self.qos = self.compute_QoS(new_rtt)

    def get_queue_waiting_time(self, t_slot: int):
        return t_slot - self.enqueue_time

    def get_complain_waiting_time(self, t_slot: int):
        return t_slot - self.enqueue_time - self.max_wait

    def get_max_wait(self):
        return self.max_wait

    # def increase_time_running(self, t=1):
    #     self.time_running += t

    def get_time_running(self, t_slot):
        return t_slot - self.deploy_time

    def get_duration(self):
        return self.duration

    def get_QoS(self):
        return self.qos

    def update_QoS(self, rtt):
        self.qos = self.compute_QoS(rtt)

    def get_resource_cost(self):
        return self.resources_cost

    def is_terminated(self, t_slot, t_slot_end: bool):
        """
        :param t_slot:
        :param t_slot_end: if True means that time slot is going to end, so the match will be terminated at the end of this time slot
        :return:
        """
        t_slot_time = 1 if t_slot_end else 0
        return t_slot - self.deploy_time + t_slot_time >= self.duration

    # # calcolo basato su FPS
    # def calculate_QoS(self, rtt):
    #     self.QoS = self.compute_QoS(rtt)

    def compute_QoS(self, rtt_ms):
        fps = round(1000 / rtt_ms)
        fps_qos = Game._qos_levels[self.game_type]
        qos = next(itertools.dropwhile(lambda e: e[0] < fps, fps_qos))[1]
        return qos

    @staticmethod
    def get_num_qos_level():
        return 6  # TODO from config

    def get_users_id(self):
        return [i.get_base_stations() for i in self.group]

    def __eq__(self, other):
        return self.game_id == other.get_id()

    def __repr__(self):
        return f'{self.game_id} BSs: {self.group}'
