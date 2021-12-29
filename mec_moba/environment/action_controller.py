from __future__ import annotations
import typing
from typing import List, Optional

import gurobipy as gp
from gurobipy import GRB

if typing.TYPE_CHECKING:
    from mec_moba.environment.matches.game import Game


class DqnAction:
    def __init__(self, cap_value, op_value, deploy_value, migration_value):
        self._cap_value = cap_value
        self._op_value = op_value
        self._deploy_value = deploy_value
        self._migration_value = migration_value

    def _to_tuple(self):
        return self._cap_value, self._op_value, self._deploy_value, self._migration_value

    @property
    def capacity_value(self):
        return self._cap_value

    @property
    def over_provisioning_value(self):
        return self._op_value

    @property
    def deploy_value(self):
        return self._deploy_value

    @property
    def migration_value(self):
        return self._migration_value

    def is_no_action(self):
        return all(map(lambda p: p == 0, (self._cap_value, self._op_value, self._deploy_value, self._migration_value)))

    def __repr__(self):
        return str((self._cap_value, self._op_value, self._deploy_value, self._migration_value))


class MigrationInstruction:
    def __init__(self, match: Game, new_facility_id: int):
        self.match: Game = match
        self.old_facility_id: int = match.get_facility_id()  # this is the current facility
        self.new_facility_id: int = new_facility_id


class DeployInstruction:
    def __init__(self, match: Game, facility_id: int):
        self.match: Game = match
        self.facility_id: int = facility_id


class ActionResultsInstructions:
    def __init__(self,
                 num_facilities: int,
                 to_deploy_matches: Optional[List[DeployInstruction]] = None,
                 to_migrate_matches: Optional[List[MigrationInstruction]] = None,
                 op_levels: Optional[List[float]] = None,
                 # migration_ratio=0,
                 is_feasible: bool = True):
        self.to_deploy_matches = to_deploy_matches
        self.to_migrate_matches = to_migrate_matches
        self.op_levels = op_levels
        self._feasible = is_feasible
        self._num_facilities = num_facilities

    @property
    def is_feasible(self):
        return self._feasible

    def get_matches_to_deploy(self) -> List[DeployInstruction]:
        return self.to_deploy_matches if self.to_deploy_matches is not None else []

    def get_matches_to_migrate(self) -> List[MigrationInstruction]:
        return self.to_migrate_matches if self.to_migrate_matches is not None else []

    def get_facilities_used_op_levels(self):
        return self.op_levels if self.op_levels is not None else [0] * self._num_facilities  # TODO check

    def __repr__(self):
        return f'Deployed: {len(self.to_deploy_matches) if self.to_deploy_matches else 0}, ' \
               f'Migrated: {len(self.to_migrate_matches) if self.to_migrate_matches else 0}'


def _create_and_solve_opt_model(cost_c, cost_r, gamma, facility_cap, facility_max_cap, B, C_ti, A, S, F, op_action_val, mig_cost, old_facilities):
    # try:

    N = len(S) + len(A)
    facility_negotiated_capacity = [f_cap + (f_max_cap - f_cap) * op_action_val for f_cap, f_max_cap in zip(facility_cap, facility_max_cap)]
    # Create a new model
    m = gp.Model("DQN")
    m.Params.LogToConsole = 0
    m.Params.MIPGap = float(1e-2)

    # Create variables
    x = m.addMVar((N, F), vtype=GRB.BINARY, name="X")
    # v = m.addMVar(F, lb=0, vtype=GRB.INTEGER, name="v")

    # Set objective
    # m.setObjective(sum(sum(cost_c[i, j] * x[i, j] for j in range(F)) for i in range(N)) +
    #                sum(cost_e[j] * v[j] for j in range(F))
    #                + sum(mig_cost[i] * (1 - x[i, old_facilities[i]]) for i in range(len(S))), GRB.MINIMIZE)
    m.setObjective(sum(sum(cost_c[i, j] * x[i, j] for j in range(F)) for i in range(N)) +
                   sum(mig_cost[i] * (1 - x[i, old_facilities[i]]) for i in range(len(S))), GRB.MINIMIZE)
    # TODO cost_r in matrice
    U = [sum(cost_r[j] for i in range(len(B)) if B[i].get_facility_id() == j) for j in range(F)]
    to_debug = []

    # Set constrains

    #
    m.addConstrs((sum([x[i, j] for j in range(F)]) == 1 for i in range(N)),
                 name='c0')

    #
    m.addConstrs((sum(cost_r[j] * x[i, j] for i in range(N)) <= (facility_negotiated_capacity[j] - U[j]) for j in range(F)),
                 name='c1')
    #
    m.addConstrs((sum(cost_c[i, j] * x[i, j] for j in range(F)) <= (gamma * C_ti[i]) for i in range(len(S))),
                 name='c2')
    #
    # m.addConstrs((v[j] <= (b * facility_cap[j]) for j in range(F)),
    #              name='c3')
    m.update()
    m.optimize()
    # print('Obj: %g' % m.objVal)
    # print(v_ret)
    if not hasattr(m, 'objVal'):
        # print('no solution gurobi, there is a None Type')
        # for fac in range(F):
        #     to_debug += [a * facility_cap[fac], U[fac], b * facility_cap[fac], None]
        return None
    else:
        x_ret = [[abs(x.tolist()[i][j].x) for j in range(F)] for i in range(N)]
        # v_ret = [abs(j.x) for j in v.tolist()]
        # for fac in range(F):
        #     to_debug += [a * facility_cap[fac], U[fac], b * facility_cap[fac], v_ret[fac]]
        return x_ret

    # except gp.GurobiError as e:


#     print('Error code ' + str(e.message) + ': ' + str(e))
#
# except AttributeError as e:
#     print('Encountered an attribute error', e)


def do_action(action, environment) -> ActionResultsInstructions:
    if action.is_no_action():
        return ActionResultsInstructions(num_facilities=environment.physical_network.n_mec)  # No action

    if not environment.validate_action(action):
        return ActionResultsInstructions(num_facilities=environment.physical_network.n_mec, is_feasible=False)  # Bad action
    # -------------------------------------
    selected_instancies, blocked_instances = environment.get_migrate_and_blocked_list(action.migration_value)
    instances_deploy = environment.match_controller.get_deploy_list(action.deploy_value)
    assignable_instances_N = selected_instancies + instances_deploy

    n_mec = environment.physical_network.n_mec
    gamma = 1  # TODO from config
    assignment_cost = environment.build_assignment_cost(len(selected_instancies) + len(instances_deploy), assignable_instances_N)

    current_assignment_costs = [5 - x.get_QoS() for x in selected_instancies]

    migration_cost = [1] * len(selected_instancies)
    old_f = [i.get_facility_id() for i in selected_instancies]
    op_costs = [1] * n_mec

    x = _create_and_solve_opt_model(assignment_cost, [1] * n_mec, gamma,
                                    environment.get_facility_capacities(),
                                    environment.get_facility_max_capacities(),
                                    blocked_instances,
                                    current_assignment_costs, instances_deploy, selected_instancies, n_mec,
                                    action.over_provisioning_value / 100,
                                    mig_cost=migration_cost, old_facilities=old_f)

    if x is None:
        return ActionResultsInstructions(num_facilities=environment.physical_network.n_mec, is_feasible=False)  # Unfeasible assignment

    # Prepare instructions
    # x is len(assignable_instances_N) in size
    x_selected = x[:len(selected_instancies)]
    x_deploy = x[len(selected_instancies):]

    mig_inst = [MigrationInstruction(selected_instancies[i], j)
                for i in range(len(x_selected))
                for j in range(environment.physical_network.n_mec) if x_selected[i][j] == 1 and selected_instancies[i].get_facility_id() != j]
    deploy_inst = [DeployInstruction(instances_deploy[i], j)
                   for i in range(len(x_deploy))
                   for j in range(environment.physical_network.n_mec) if x_deploy[i][j] == 1]

    return ActionResultsInstructions(num_facilities=environment.physical_network.n_mec,
                                     to_deploy_matches=deploy_inst,
                                     to_migrate_matches=mig_inst)
