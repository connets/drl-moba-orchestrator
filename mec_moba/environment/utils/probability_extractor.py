import csv
from typing import Iterable
import numpy as np


def extract_game_requests_probability():
    with open('data/match_probability_2019.csv') as csv_file:
        return np.genfromtxt(fname='data/match_probability_2019.csv',
                             delimiter=',',
                             skip_header=1, usecols=[2])

    #     csv_reader = csv.reader(csv_file, delimiter=',')
    #     line_count = 0
    #     probability = list()
    #     for row in csv_reader:
    #         if line_count > 0:
    #             probability += [float(row[2])]
    #         line_count += 1
    # return list(cumsum(probability))
