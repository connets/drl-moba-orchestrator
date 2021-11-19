class User(object):
    def __init__(self, user_id, starting_bs, is_moving=False):
        self.user_id = user_id
        self.bs = starting_bs[0]
        self.is_moving = is_moving

    def get_id(self):
        return self.user_id

    def get_bs(self):
        return self.bs

    def set_bs(self, new_bs):
        self.bs = new_bs
