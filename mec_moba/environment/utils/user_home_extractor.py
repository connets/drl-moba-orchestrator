import csv
import pandas as pd


def extract_users():
    users = pd.read_csv('data/user_home_location_NUMERIC.csv')
    return dict(zip(users.user_id, users.home_location))
    # with open('data/user_home_location_NUMERIC.csv') as csv_file:
    #     csv_reader = csv.reader(csv_file, delimiter=',')
    #     line_count = 0
    #     users = dict()
    #     for row in csv_reader:
    #         if line_count > 0:
    #             if int(row[0]) in users.keys():
    #                 users[int(row[0])] += [row[1]]
    #             else:
    #                 users[int(row[0])] = [row[1]]
    #         line_count += 1
    # return users
