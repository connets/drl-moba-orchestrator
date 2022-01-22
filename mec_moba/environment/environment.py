from __future__ import annotations

import typing
from typing import Sized
from collections import namedtuple
import numpy as np

from mec_moba.environment.matches import GameGenerator, MatchController
from mec_moba.environment.physical_network.physicalnetwork import PhysicalNetwork

import mec_moba.environment.utils.utils as utils
import mec_moba.environment.utils.logging_utils as logging_utils

VALIDATE_ACTION_PARAM = 'validate_action'

if typing.TYPE_CHECKING:
    from mec_moba.environment.action_controller import DqnAction, ActionResultsInstructions

MatchLogRecord = namedtuple('MatchLogRecord', field_names=['t_slot', 'match_id', 'facility_id'])


def load_single_env(instance, generic_dict):
    for keys, values in generic_dict.items():
        setattr(instance, keys, values)


class TimeSlotSnapshot:
    @staticmethod
    def dummy_state(environment):
        """Used also as initial state"""
        return TimeSlotSnapshot(running_qos_prob=environment.match_controller.get_dummy_qos_state(),
                                facility_utilization=environment.physical_network.get_dummy_facility_utilization(),
                                queue_occupation=0,
                                mean_queue_waiting_time=0,
                                mean_running_time=0,
                                migrable_ratio=0,
                                queue_drop_rate=0)

    def __init__(self,
                 running_qos_prob: typing.List[float],
                 facility_utilization: typing.List[float],
                 queue_occupation: float,
                 mean_queue_waiting_time: float,
                 mean_running_time: float,
                 migrable_ratio: float,
                 queue_drop_rate: float):
        self.running_qos_prob = running_qos_prob
        self.facility_utilization = facility_utilization
        self.queue_occupation = queue_occupation
        self.mean_queue_waiting_time = mean_queue_waiting_time
        self.mean_running_time = mean_running_time
        self.migrable_ratio = migrable_ratio
        self.queue_drop_rate = queue_drop_rate

    @property
    def num_features(self) -> int:
        return len(self.to_array()[0])

    # @property
    def to_array(self) -> Sized:
        a = np.array(self.running_qos_prob + self.facility_utilization +
                     [self.queue_occupation, self.mean_queue_waiting_time, self.mean_running_time, self.migrable_ratio, self.queue_drop_rate])
        return a[np.newaxis, :]

    def to_log_format(self):
        to_log = self.running_qos_prob + self.facility_utilization
        to_log += [self.queue_occupation, self.mean_queue_waiting_time, self.mean_running_time, self.migrable_ratio, self.queue_drop_rate]
        return to_log

    # ----------------------------------


# REWARD COMPONENTS FUNCTIONS
# ----------------------------------
def _queue_occupation_reward_cmp(environment, action: DqnAction, act_res_inst: ActionResultsInstructions):
    facility_occ = environment.physical_network.get_all_facilities_occupation(normalized=True)
    queue_rew_factor = 1
    if action.is_no_action():
        if sum(facility_occ) == 0 and len(environment.match_controller.queue) == 0:
            return 0
        if len(environment.match_controller.queue) > 0 and sum(facility_occ) < 1.2 * environment.physical_network.n_mec:  # TODO remove hardcoded
            queue_rew_factor = (1.2 * environment.physical_network.n_mec) / max(sum(facility_occ), 1)

    return - queue_rew_factor * environment.match_controller.get_queue_occupation_state()


def load_balancing_reward_cmp(environment, action: DqnAction, act_res_inst: ActionResultsInstructions):
    facility_occ = environment.physical_network.get_all_facilities_occupation(normalized=True)
    # calcolo il gini, se son tutti 0 allora gini = 0, metto in negativo perhcè valori > 0 vanno a indicare disuguaglianza,
    # quindi puniscono reward
    if sum(facility_occ) > 0:
        load_balance = - utils.gini([max(0.0, f - 0.8) for f in facility_occ])
    else:
        load_balance = 0
    return load_balance


def overprovisioning_reward_cmp(environment, action: DqnAction, act_res_inst: ActionResultsInstructions):
    return - action.over_provisioning_value / 100
    # if action.is_no_action():
    #     return 0
    #
    # facility_occ = environment.physical_network.get_all_facilities_occupation(normalized=True)
    # op_min_values = [sum([(action.capacity_value + op) / 100
    #                       for _ in range(environment.physical_network.n_mec)]) - sum(facility_occ)
    #                  for op in [0, 10, 20, 30, 40]]
    #
    # _, op_min_val_feasible = sorted(filter(lambda e: e[0] >= 0, zip(op_min_values, [0, 10, 20, 30, 40])))[0]
    # # op_cost = (op_val - op_min_val_feasible) / 100 * self.Physical_net.n_mec
    # op_val = action.over_provisioning_value
    # op_cost = (op_val - op_min_val_feasible)  # / environment.physical_network.n_mec if op_val > 0 else 0
    # op_cost /= 40
    # if op_cost > 0:
    #     # bonus if v_j from PLI results have brought to a better solution
    #     op_cost -= sum((v_j / 4) - op_min_val_feasible / 40 for v_j in act_res_inst.get_facilities_used_op_levels()) / environment.physical_network.n_mec
    #     # op_cost = max(0, op_cost)  # Dealing with too good optimization
    #
    # return -op_cost


def queue_waiting_time(environment, action: DqnAction, act_res_inst: ActionResultsInstructions):
    return - environment.match_controller.get_mean_queue_waiting_time()


def free_capacity_reward_cmp(environment, action: DqnAction, act_res_inst: ActionResultsInstructions):
    facility_occ = environment.physical_network.get_all_facilities_occupation(normalized=True)
    available_free_cap = sum(facility_occ) / environment.physical_network.n_mec
    return - max(0, available_free_cap)


def queue_drop_ratio_reward_cmp(environment, action: DqnAction, act_res_inst: ActionResultsInstructions):
    return - environment.match_controller.get_queue_drop_rate()


def mean_qos_reward_cmp(environment, action: DqnAction, act_res_inst: ActionResultsInstructions):
    qos_mean = environment.match_controller.get_mean_qos_running_instances()
    if qos_mean < 0:
        return None
    return - qos_mean / 5


# reward_comp = {'queue_occ': (_queue_occupation_reward_cmp, -1),
#                'load_balance': (load_balancing_reward_cmp, -1),
#                'op_cost': (overprovisioning_reward_cmp, -1),
#                'waiting_time': (queue_waiting_time, -1),
#                'queue_drop_rate': (queue_drop_ratio_reward_cmp, -1),
#                'free_capacity': (free_capacity_reward_cmp, -1)
#                }
reward_comp = {'qos': (mean_qos_reward_cmp, -1),
               # 'load_balance': (load_balancing_reward_cmp, -1),
               'op_cost': (overprovisioning_reward_cmp, -1),
               'waiting_time': (queue_waiting_time, -1),
               # 'migration': (queue_drop_ratio_reward_cmp, -1),
               # 'free_capacity': (free_capacity_reward_cmp, -1)
               }

# reward_comp_order = ['load_balance', 'op_cost', 'waiting_time', 'queue_occ', 'queue_drop_rate', 'free_capacity']

reward_comp_order = ['qos', 'op_cost', 'waiting_time']


class Environment:
    @staticmethod
    def default_reward_weights():
        return tuple([1] * len(reward_comp_order))

    def __init__(self, reward_weights, gen_requests_until=None,
                 match_probability_file=None,
                 n_games_per_epoch=None,
                 normalize_reward=True):
        self.physical_network: PhysicalNetwork = PhysicalNetwork(self)
        self.match_controller: MatchController = MatchController(self, self.physical_network)
        self.match_generator = GameGenerator(gen_requests_until,
                                             match_probability_file=match_probability_file,
                                             n_games_per_epoch=n_games_per_epoch)
        self.reward_weights = reward_weights
        # self.gen_requests_until = gen_requests_until
        self.normalize_reward = normalize_reward
        self.validate_action_enabled = False  # get_config_value(Environment.get_module_config_name(), VALIDATE_ACTION_PARAM)

        self._epoch_t_slot = 0
        self._absolute_t_slot = 0

    @property
    def epoch_t_slot(self):
        return self._epoch_t_slot

    @property
    def absolute_t_slot(self):
        return self._absolute_t_slot

    def set_seed(self, seed):
        self.match_generator.set_seed(seed)

    def reset(self) -> TimeSlotSnapshot:
        self.change_epoch()
        return self.get_time_slot_state()

    def _read_and_set_observable_state(self):
        self.t_slot_state = TimeSlotSnapshot(running_qos_prob=self.match_controller.get_qos_state(),
                                             facility_utilization=self.physical_network.get_all_facilities_occupation(normalized=True),
                                             queue_occupation=self.match_controller.get_queue_occupation_state(),
                                             mean_queue_waiting_time=self.match_controller.get_mean_queue_waiting_time(),
                                             mean_running_time=self.match_controller.get_mean_running_time_state(),
                                             migrable_ratio=self.get_migrable_ratio(),
                                             queue_drop_rate=self.match_controller.get_queue_drop_rate())

    def get_migrable_ratio(self) -> float:
        return self.match_controller.get_migrable_ratio()

    def get_migrate_and_blocked_list(self, migration_value):
        return self.match_controller.get_migrate_and_blocked_list(migration_value)

    def get_time_slot_state(self) -> TimeSlotSnapshot:
        self._read_and_set_observable_state()
        return self.t_slot_state

    def get_time_slot_state_n_features(self):
        return self.get_time_slot_state().num_features

    def compute_reward(self, action: DqnAction, action_res_inst: ActionResultsInstructions):  # migrations, capacity_val, op_val, ):
        if action_res_inst.is_feasible:
            # each split reward component is to have max value = 0
            split_rewards = [reward_comp[rw_cmp][0](self, action, action_res_inst) for rw_cmp in reward_comp_order]

        else:
            split_rewards = [reward_comp[rw_cmp][1] for rw_cmp in reward_comp_order]

        worst_value_sum = [w * v for (_, v), w, rw_cmp_value in zip(reward_comp.values(), self.reward_weights, split_rewards)
                           if rw_cmp_value is not None]
        worst_value_sum = abs(sum(worst_value_sum))
        # FILTER NONEs
        split_rewards = [w * v for v, w in zip(split_rewards, self.reward_weights) if v is not None]
        reward = sum(split_rewards)
        if self.normalize_reward:
            reward /= worst_value_sum
        return reward, split_rewards

    def inc_timeslot(self) -> bool:
        # print('running matches', len(self.match_controller.running))
        self._epoch_t_slot = (self._epoch_t_slot + 1) % 1008
        self._absolute_t_slot += 1
        self.match_controller.update_qos()
        return self._epoch_t_slot == 0

    def change_epoch(self):
        self._absolute_t_slot = 0
        self._epoch_t_slot = 0  # for sanity, because it should be already zero
        # current_time = time.time()
        # epoch_elapsed_time = current_time - self._start_time
        # self._start_time = current_time
        # print(f"Epoch {self._epoch} done in {epoch_elapsed_time} seconds")
        # self._epoch += 1  # this order of instructions is important because save the next epoch number
        self.match_controller.change_epoch()
        self.physical_network.change_epoch()
        self.match_generator.change_epoch()

    def validate_action(self, action: DqnAction):
        if self.validate_action_enabled:
            if (action.migration_value > 0 and self.t_slot_state.migrable_ratio == 0) or \
                    (action.capacity_value > 0 and self.t_slot_state.queue_occupation == 0):
                # logging.warning(f'Detected unfeasible action {migration}, {deploy}')
                return False
        return True

    def get_games(self):
        return self.match_controller.get_games()

    def get_facility_capacities(self):
        return self.physical_network.get_mec_capacities()

    def get_facility_max_capacities(self):
        return self.physical_network.get_mec_max_capacities()

    def build_assignment_cost(self, sz, assignable_instances_N):
        assignment_cost = np.zeros(shape=(sz, self.physical_network.n_mec))
        for i, match in enumerate(assignable_instances_N):
            for j in range(self.physical_network.n_mec):
                assignment_cost[i, j] = 5 - match.compute_QoS(
                    max([self.physical_network.get_rtt(bs, j) for bs in match.get_base_stations()]))
        return assignment_cost

    def get_migrate_list(self, migrate_val):
        games = self.get_games()
        pivot = round(len(self.migrable_vnf) * (migrate_val / 100))
        selected = self.migrable_vnf[:pivot]
        blocked = [x for x in games if x not in selected]
        return selected, blocked

    # # "Public" Methods called from main loop
    # def enqueue_new_match_requests(self):
    #     match_requests = self.match_generator.get_match_requests(t_slot=self._epoch_t_slot)
    #     self.match_controller.enqueue_match_requests(match_requests)
    #     # for match in match_requests:

    def implement_action(self, action_result: ActionResultsInstructions):
        # pre_running = len(self.match_controller.running)
        for deploy_inst in action_result.get_matches_to_deploy():
            self.match_controller.deploy(deploy_inst.match, deploy_inst.facility_id)
            self.physical_network.deploy(deploy_inst.match, deploy_inst.facility_id)

        for mig_inst in action_result.get_matches_to_migrate():
            self.match_controller.migrate(mig_inst.match, mig_inst.old_facility_id, mig_inst.new_facility_id)
            self.physical_network.migrate(mig_inst.match, mig_inst.old_facility_id, mig_inst.new_facility_id)

        # assert len(self.match_controller.running) == pre_running + len(action_result.get_matches_to_deploy())
        # assert len(self.match_controller.running) == sum([len(f.deployed_matches) for f in self.physical_network._mec_facilities.values()])

    def cleanup_terminated_matches(self, t_slot_end: bool):
        self.match_controller.cleanup_terminated_matches(t_slot_end)

    def log_all_matches_data(self, log_dir):
        # print(self._epoch_t_slot)
        data_to_log = [MatchLogRecord(self._epoch_t_slot, m.id, f.facility_id)
                       for f in self.physical_network.get_mec_facilities()
                       for m in f.get_deployed_matches()]

        logging_utils.log_to_csv_file(log_dir, 'match_data.csv', data_to_log)
