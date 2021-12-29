from typing import List

from mec_moba.environment.matches import Game


class MecFacility(object):
    def __init__(self, facility_id, capacity=8, max_capacity=12):
        self._facility_id = facility_id
        self._capacity = capacity
        self._max_capacity = max_capacity
        self.deployed_matches = dict()

    def reset(self):
        self.deployed_matches = dict()

    @property
    def facility_id(self):
        return self._facility_id

    def get_deployed_matches(self) -> List[Game]:
        return list(self.deployed_matches.values())

    def deploy(self, match: Game):
        self.deployed_matches[match.id] = match

    def undeploy(self, game: Game):
        del self.deployed_matches[game.id]  # key error means wrong implementation

    @property
    def capacity(self):
        return self._capacity

    @property
    def max_capacity(self):
        return self._max_capacity

    def get_facility_occupation(self, normalized=False):
        occupation = sum(match.get_resource_cost() for match in self.deployed_matches.values())
        if normalized:
            occupation /= self._capacity
        return occupation
