
class Config(object):
    def __init__(self):
        self.test_file = 'dataset/track1_round1_testA_20210222.csv'
        self.train_file = 'dataset/track1_round1_train_20210222.csv'
        self.result_file = 'dataset/result.csv'
        self.min_freq = 3
        self.max_l = 105
        self.vocab_size = 860 
        self.n_label = 17