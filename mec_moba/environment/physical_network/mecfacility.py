from mec_moba.environment.matches import Game


class MecFacility(object):
    def __init__(self, facility_id, capacity):
        self._facility_id = facility_id
        self._capacity = capacity
        self.deployed_matches = dict()

    @property
    def facility_id(self):
        return self._facility_id

    def deploy(self, match: Game):
        self.deployed_matches[match.id] = match

    def undeploy(self, game: Game):
        del self.deployed_matches[game.id]  # key error means wrong implementation

    @property
    def capacity(self):
        return self._capacity

    def get_facility_occupation(self, normalized=False):
        occupation = sum(match.get_resource_cost() for match in self.deployed_matches.values())
        if normalized:
            occupation /= self._capacity
        return occupation
