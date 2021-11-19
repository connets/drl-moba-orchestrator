from __future__ import annotations

# non prendiamo il caso che un utente giochi nello stesso timeslot
import collections
import json
from typing import List

from mec_moba.environment.utils.probability_extractor import *
from mec_moba.environment.utils.user_home_extractor import *
from mec_moba.environment.matches.game import Game
from mec_moba.environment.matches.user import User

MAX_DURATION_PARAM = 'game_max_duration'
NUM_GAMES_PER_EPOCH_PARAM = 'num_games_per_epoch'
MAX_WAITING_TIME_PARAM = 'max_waiting_time'
QUEUE_EXIT_TIME_PARAM = 'queue_exit_time'
MATCH_RESOURCE_PARAM = 'resource'
NUM_USER_MATCH = 'num_user_match'

defaults = {MAX_WAITING_TIME_PARAM: 3,
            NUM_GAMES_PER_EPOCH_PARAM: 5000,
            MAX_DURATION_PARAM: 10,
            QUEUE_EXIT_TIME_PARAM: None,
            MATCH_RESOURCE_PARAM: 1,
            NUM_USER_MATCH: 4
            }


class GameGenerator:

    # @staticmethod
    # def get_module_config_options() -> Iterable[ConfigOption]:
    #     return [ConfigOption(name=MAX_WAITING_TIME_PARAM, default_value=3, cli_type=int, help_string='The maximum time a match can wait in the queue before QoS degradation'),
    #             ConfigOption(name=NUM_GAMES_PER_EPOCH_PARAM, default_value=5000, cli_type=int, help_string='Number of matches generated in a single epoch'),
    #             ConfigOption(name=MAX_DURATION_PARAM, default_value=10, help_string='Duration of match'),  # TODO implement parsing distribution
    #             ConfigOption(name=QUEUE_EXIT_TIME_PARAM, default_value=None, help_string='Max waiting time in the queue before leaving'),
    #             ConfigOption(name=MATCH_RESOURCE_PARAM, default_value=1, help_string='Resource consumption of a single match'),
    #             ConfigOption(name=NUM_USER_MATCH, default_value=4, help_string='Number of user per match')
    #             ]

    def __init__(self, no_overlapping=False):
        # distribuzione durata di una partita (la durata è in timeslots)
        self.duration = defaults[MAX_DURATION_PARAM]
        self.max_wait = defaults[MAX_WAITING_TIME_PARAM]
        self.exit_time = defaults[QUEUE_EXIT_TIME_PARAM]
        self.n_games_per_epoch = defaults[NUM_GAMES_PER_EPOCH_PARAM]

        # numero di utenti per partita/servizio
        self.n_users = defaults[NUM_USER_MATCH]

        # costo in unità di una partita
        self.resources = defaults[MATCH_RESOURCE_PARAM]

        # Random generator
        self.rng = np.random.default_rng()  # TODO think about using seed from config
        # id unico partita
        self.game_id_progressive = 0  # do not use generators because they are not pickle compliant

        self.history = dict()

        # check se un giocatore sta già giocando una aprtita
        # e quindi evita che si crei un'altra partita
        self.no_overlapping = no_overlapping

        # probabilità di giocare per ogni timeslot da testare per ogni gruppo
        self.probability_to_play = extract_game_requests_probability()

        # ottengo il dizionario con le bs e tutti gli utenti sotot a ciascuna
        self.users_bss = extract_users()

        # ottengo le liste di gruppi  #TODO get them from config
        with open('data/cliques_all_members_home_group.json', 'r') as f:
            self.groups = json.load(f)

            # da modificare se duration non è un numero fisso ma un range o una distribuzione

        self.epoch_games_generated = 0

    def get_duration(self):
        return self.duration

    def generate_epoch_matches(self):
        game_request_timeslot_sample = self.rng.choice(len(self.probability_to_play),
                                                       size=self.n_games_per_epoch,
                                                       p=self.probability_to_play)

        for t_slot, n_games in collections.Counter(game_request_timeslot_sample).items():
            # each time slot game requests is a generator
            self.history[t_slot] = (self._create_game() for _ in range(n_games))

    def _create_game(self):
        self.game_id_progressive += 1  # do not use generators because they are not pickle compliant
        return Game(game_id=self.game_id_progressive,
                    duration=self.get_duration(),
                    resources_cost=self.resources,
                    group=self._get_random_group(),
                    max_wait=self.max_wait,
                    queue_abandon_time=self.exit_time)

    def get_match_requests(self, t_slot) -> List[Game]:
        # if t_slot key is not in the dictionary it means zero matches!!
        x = list(self.history.get(t_slot, []))
        self.epoch_games_generated += len(x)
        #print('Game generated ', t_slot, self.epoch_games_generated)
        return x

    def check_overlapping(self, l) -> bool:
        # TODO
        return False

    def _get_random_group(self):
        r = np.random.randint(len(self.groups))
        if not self.no_overlapping:
            return [User(int(j), self.users_bss[int(j)]) for j in self.groups[r]]
        else:
            x = False
            while not x:
                tmp = [User(int(j), self.users_bss[int(j)]) for j in self.groups[r]]
                x = self.check_overlapping(tmp)
                r = np.random.randint(len(self.groups))
            return tmp

    def change_epoch(self):
        self.epoch_games_generated = 0
        self.history = dict()
        self.generate_epoch_matches()
