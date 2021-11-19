import pickle
import csv
import pandas

allowed_facilities = set(range(10))

def extract_delay():
    df = pandas.read_csv('data/BS_Facility_delay_NEW.csv.gz')
    df = df[df.m1.isin(allowed_facilities)]
    delay_dict = dict(zip(zip(df.t_slot, df.enb, df.m1), df.delay))
    n_bs = df.enb.nunique()
    n_mec = len(allowed_facilities)

    # with open('data/BS_Facility_delay_NEW.csv') as csv_file:
    #     csv_reader = csv.reader(csv_file, delimiter=',')
    #     line_count = 0
    #     bs_count = 1
    #     gne = list()
    #     mec = list()
    #     for row in csv_reader:
    #         if line_count > 0:
    #             if line_count == 1:
    #                 tmp = row[0]
    #             else:
    #                 if tmp != row[0]:
    #                     bs_count += 1
    #                     tmp = row[0]
    #             if int(row[1]) not in mec:
    #                 mec += [int(row[1])]
    #             gne.append(row)
    #
    #         line_count += 1
    #
    #     # print(mins * 2, maxs * 2)
    #     # print(bs_count)
    #     # print([x for x in gne if gne])
    #     my_dict = dict()
    #     for g in gne:
    #         if int(g[2]) in my_dict.keys():
    #             my_dict[int(g[2])].append([g[0], int(g[1]), float(g[3])])
    #         else:
    #             my_dict[int(g[2])] = [[g[0], int(g[1]), float(g[3])]]
    #     delay_dict[-1] = [n_bs, sorted(mec)[:len(allowed_facilities)]]
    ret_dict = {'n_bs': n_bs,
                'n_mec': n_mec,
                'delays': delay_dict}
    file = open('data/delay_dict.pkl', 'wb')
    pickle.dump(ret_dict, file)
    file.close()
