import csv
from typing import Iterable
import numpy as np


def extract_game_requests_probability():
    return np.genfromtxt(fname='data/match_probability_2019.csv',
                         delimiter=',',
                         skip_header=1, usecols=[2])

