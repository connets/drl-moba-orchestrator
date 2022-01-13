import csv
from typing import Iterable
import numpy as np
import pandas as pd


def extract_game_requests_probability(fname=None):
    if fname is None:
        fname = 'data/match_probability_2019.csv'

    return pd.read_csv(fname).match_probability.values
    # return np.genfromtxt(fname=fname,
    #                      delimiter=',',
    #                      skip_header=1, usecols=[2])
