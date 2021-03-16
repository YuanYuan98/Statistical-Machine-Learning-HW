import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn_utils

class GRU(nn.Module):
    def __init__(self, config, args):
        super(GRU, self).__init__()
        self.config = config
        self.args = args

        self.vocab_size = self.config.vocab_size
        self.embedding_size = self.args.embedding_size
        self.hidden_size = self.args.hidden_size

        self.embed = nn.Embedding(self.vocab_size, self.embedding_size,padding_idx=1)

        self.GRU = nn.GRU(self.embedding_size, self.hidden_size, batch_first = True,dropout = self.args.dropout)
        self.w_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_context_vector = nn.Parameter(torch.randn([self.hidden_size, 1]).float())
        self.softmax = nn.Softmax(dim = 1)

        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_out = nn.Linear(self.hidden_size, self.config.n_label)

        self.activate = nn.Sigmoid()


    def forward(self, x, length):
        # batch_size * sentence_len

        x_embed = self.embed(x)
        # batch_size * sentence_len * embedding_size


        x_embed = rnn_utils.pack_padded_sequence(x_embed, length, batch_first=True, enforce_sorted=False)

        x_out, _ = self.GRU(x_embed)
        # batch_size * sentence_len * hidden_size

        x_out,_ = rnn_utils.pad_packed_sequence(x_out, batch_first=True, total_length=self.config.max_l)

        Hw = torch.tanh(self.w_proj(x_out))
        # batch_size * sentence_len * hidden_size
        w_score = self.softmax(Hw.matmul(self.w_context_vector))
        x_out = x_out.mul(w_score)
        x_out = torch.sum(x_out, dim = 1)

        x_out = F.relu(self.linear(x_out))
        x_out = self.linear_out(x_out)

        x_prob = self.activate(x_out)


        return x_prob


class LSTM(nn.Module):
    def __init__(self, config, args):
        super(LSTM, self).__init__()
        self.config = config
        self.args = args

        self.vocab_size = self.config.vocab_size
        self.embedding_size = self.args.embedding_size
        self.hidden_size = self.args.hidden_size

        self.embed = nn.Embedding(self.vocab_size, self.embedding_size,padding_idx=1)

        self.LSTM = nn.LSTM(self.embedding_size, self.hidden_size, batch_first = True, bidirectional=True, dropout = self.args.dropout)

        self.w_proj = nn.Linear(self.hidden_size*2, self.hidden_size*2)
        self.w_context_vector = nn.Parameter(torch.randn([self.hidden_size*2, 1]).float())
        self.softmax = nn.Softmax(dim = 1)

        self.linear = nn.Linear(self.hidden_size*2, self.hidden_size*2)
        self.linear_out = nn.Linear(self.hidden_size*2, self.config.n_label)

        self.activate = nn.Sigmoid()


    def forward(self, x, length):
        # batch_size * sentence_len

        x_embed = self.embed(x)
        # batch_size * sentence_len * embedding_size


        x_embed = rnn_utils.pack_padded_sequence(x_embed, length, batch_first=True, enforce_sorted=False)

        x_out, _ = self.LSTM(x_embed)
        # batch_size * sentence_len * hidden_size

        x_out,_ = rnn_utils.pad_packed_sequence(x_out, batch_first=True, total_length=self.config.max_l)

        Hw = torch.tanh(self.w_proj(x_out))
        # batch_size * sentence_len * hidden_size
        w_score = self.softmax(Hw.matmul(self.w_context_vector))
        x_out = x_out.mul(w_score)
        x_out = torch.sum(x_out, dim = 1)

        x_out = F.relu(self.linear(x_out))
        x_out = self.linear_out(x_out)

        x_prob = self.activate(x_out)

        return x_prob



class dis_attn(nn.Module):
    def __init__(self, config, args):
        super(dis_attn, self).__init__()
        self.config = config
        self.args = args

        self.vocab_size = self.config.vocab_size
        self.embedding_size = self.args.embedding_size
        self.hidden_size = self.args.hidden_size

        self.embed = nn.Embedding(self.vocab_size, self.embedding_size,padding_idx=1)

        self.GRU = nn.GRU(self.embedding_size, self.hidden_size, batch_first = True, dropout = self.args.dropout)
        self.w_proj = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.config.n_label)])
        self.w_context_vector = [nn.Parameter(torch.randn([self.hidden_size, 1]).float()) for i in range(self.config.n_label)]
        

        self.linear = nn.ModuleList([nn.Linear(self.hidden_size, 1) for i in range(self.config.n_label)])

        self.activate = nn.Sigmoid()
        self.softmax = nn.Softmax(dim = 1)


    def forward(self, x, length):
        # batch_size * sentence_len

        x_embed = self.embed(x)
        # batch_size * sentence_len * embedding_size


        x_embed = rnn_utils.pack_padded_sequence(x_embed, length, batch_first=True, enforce_sorted=False)

        x_out, _ = self.GRU(x_embed)
        # batch_size * sentence_len * hidden_size

        x_out,_ = rnn_utils.pad_packed_sequence(x_out, batch_first=True, total_length=self.config.max_l)
        

        Hw = [torch.tanh(w_proj(x_out)) for w_proj in self.w_proj]

        w_score = [self.softmax(h.matmul(self.w_context_vector[index].cuda())) for index, h in enumerate(Hw)]

        x_out_list = [x_out.mul(w) for w in w_score]

        x_out_list = [self.activate(self.linear[index](torch.sum(i,dim=1))) for index, i in enumerate(x_out_list)]

        x_prob = torch.cat(x_out_list, 1)

        return x_prob


