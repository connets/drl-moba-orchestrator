import csv


def extract_users():
    with open('data/user_home_location.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        users = dict()
        for row in csv_reader:
            if line_count > 0:
                if int(row[0]) in users.keys():
                    users[int(row[0])] += [row[1]]
                else:
                    users[int(row[0])] = [row[1]]
            line_count += 1
    return users
