import os
import setproctitle
import argparse
from trainer import *
from tester import *
import torch
import random
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Run persuasion")
    parser.add_argument('--use_cuda', type=bool, default=True,
                        help='GPU.')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU.')
    parser.add_argument('--epoch_num', type=int, default=500,
                        help='epoch num for early stop')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size in dataloader')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout')
    parser.add_argument('--weight_decay', type=float, default=1e-7,
                        help='weight_decay')
    parser.add_argument('--seed', type=int, default=100,
                        help='random split seed')
    parser.add_argument('--model', type=int, default=0,
                        help='model select')
    parser.add_argument('--patience', type=int, default=5,
                        help='patience to lr decay')
    parser.add_argument('--hidden_size', type=int, default=32,
                        help='hidden_size for GRU')
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='hidden_size for GRU')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='the minimum lr')
    parser.add_argument('--early_stop_num', type=int, default=5,
                        help='')
    parser.add_argument('--test_mode', type=int, default=0,
                        help='')

    
    return parser.parse_args()

def seed_set():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.autograd.set_detect_anomaly(True)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True  # 每次训练得到相同结果
    torch.backends.cudnn.benchmark = False

args = parse_args()
seed_set()

setproctitle.setproctitle("app1@yy")
args.gpu = '1'
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)


if __name__ == '__main__':

    args.test_mode = 0
    app = Trainer(args)
    r = app.Train_Step()
    result.append(r)
