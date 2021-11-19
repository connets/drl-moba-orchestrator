import pickle
import os.path


# salvo i dict
def save_env(g, m, p, _id):
    print('saving env...')
    f = open('saves/' + _id + '/env_checkpoint.pkl', 'wb')
    pickle.dump([g, m, p], f)
    f.close()


def load_env(_id):
    print("loading env...")
    if os.path.isfile('saves/' + _id + '/env_checkpoint.pkl'):
        f = open('saves/' + _id + '/env_checkpoint.pkl', 'rb')
        r = pickle.load(f)
        f.close()
        return r
    else:
        return [dict(), dict(), dict()]


def save_parameters(i, e, d, _id):
    print('saving params...')
    f = open('saves/' + _id + '/parameters_checkpoint.pkl', 'wb')
    pickle.dump([i, e, d], f)
    f.close()


def load_parameters(_id):
    if os.path.isfile('saves/' + _id + '/parameters_checkpoint.pkl'):
        print('loading params...')
        f = open('saves/' + _id + '/parameters_checkpoint.pkl', 'rb')
        r = pickle.load(f)
        f.close()
        return r
    else:
        return [0, [], []]
