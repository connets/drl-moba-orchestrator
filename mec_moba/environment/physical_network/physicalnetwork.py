from __future__ import annotations

import typing
from typing import Iterable, List

from mec_moba.environment.utils.delay_exctractor import *
import os
from mec_moba.environment.physical_network.mecfacility import MecFacility

if typing.TYPE_CHECKING:
    from mec_moba.environment import Environment

NUM_FACILITIES_PARAM = 'num_facilities'
FACILITY_CAPACITY_PARAM = 'capacity'


class PhysicalNetwork:

    def __init__(self, environment: Environment):
        self.environment = environment
        self.mec_op = 0
        if not os.path.isfile('data/delay_dict.pkl'):
            extract_delay()
        info_physical_net = pickle.load(open('data/delay_dict.pkl', 'rb'))
        self.delay_dict = info_physical_net['delays']

        self.n_bs, self.n_mec = info_physical_net['n_bs'], info_physical_net['n_mec']
        facility_capacity = 10 #get_config_value(PhysicalNetwork.get_module_config_name(), FACILITY_CAPACITY_PARAM)
        self._mec_facilities = {int(m): MecFacility(int(m), facility_capacity) for m in range(self.n_mec)}

    def get_num_mec_facilities(self):
        return self.n_mec

    def get_num_base_stations(self):
        return self.n_bs

    def get_mec_facilities(self) -> List[MecFacility]:
        return list(self._mec_facilities.values())

    def get_mec_capacities(self):
        return [facility.capacity for facility in self._mec_facilities.values()]

    def deploy(self, match, facility_id):
        self._mec_facilities[facility_id].deploy(match)

    def migrate(self, match, old_facility_id, new_facility_id):
        self._mec_facilities[old_facility_id].undeploy(match)
        self._mec_facilities[new_facility_id].deploy(match)

    def terminate_match(self, match, facility_id):
        self._mec_facilities[facility_id].undeploy(match)

    def get_rtt(self, bs, mec):
        return self.delay_dict[(self.environment.epoch_t_slot, bs, mec)] * 1.5

    def get_all_facilities_occupation(self, normalized):
        return [facility.get_facility_occupation(normalized=normalized) for facility in self._mec_facilities.values()]

    def get_dummy_facility_utilization(self):
        return [0.0] * self.n_mec
