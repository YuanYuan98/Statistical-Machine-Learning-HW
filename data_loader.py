# -*- coding: utf-8 -*-
import pandas as pd
import torchtext.vocab as Vocab
import collections
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils


class DataLoader(object):
    def __init__(self,config,args):
        self.config = config
        self.args = args

    def read_file(self,path):
        with open(path) as f:
            lines = [i.split('|,|') for i in f.read().splitlines()]
            lines = [[i[0]]+[j.rstrip(' ').split(' ') for j in i[1:]] for i in lines]
        return lines

    def Voc_Count(self,voc):
        voc_count = collections.Counter(voc)
        return Vocab.Vocab(voc_count,min_freq=self.config.min_freq,specials=['<unk>', '<pad>'])

    def pad(self,x,item='<pad>'):
        return x[:self.config.max_l] if len(x)>self.config.max_l else x+[item]*(self.config.max_l-len(x))

    def tokenizer(self,x,v):
        def token(i,v):
            return v.stoi[i] if i in v.stoi else v.stoi['<unk>']
        return [[token(i,v) for i in self.pad(x)],len(x)]

    def get_label(self,x):
        result = [0 for i in range(self.config.n_label)]
        for i in x:
            if i!='':
                result[int(i)] = 1
        return result

        #sen_len_all = rnn_utils.pad_sequence(sen_len_all,padding_value=1).permute(1,0,2).squeeze()

        #self.tokenizer(i[1],Voc)

    def data_load(self):
        TrainData = self.read_file(self.config.train_file)
        train, val = train_test_split(TrainData, test_size=0.2, random_state=self.args.seed)
        TestData = self.read_file(self.config.test_file)

        

        voc_all = [i for v in TrainData for i in v[1]]+[i for v in TestData for i in v[1]]

        Voc = self.Voc_Count(voc_all)

        train_data = rnn_utils.pad_sequence([torch.LongTensor(self.tokenizer(i[1],Voc)[0]) for i in train],padding_value=Voc.stoi['<pad>'],batch_first=True)
        train_length = torch.Tensor([self.tokenizer(i[1],Voc)[1] for i in train]).unsqueeze(dim=1)
        train_label = torch.tensor([self.get_label(i[2]) for i in train])

        val_data = rnn_utils.pad_sequence([torch.LongTensor(self.tokenizer(i[1],Voc)[0]) for i in val],padding_value=Voc.stoi['<pad>'],batch_first=True)
        val_length = torch.Tensor([self.tokenizer(i[1],Voc)[1] for i in val]).unsqueeze(dim=1)
        val_label = torch.tensor([self.get_label(i[2]) for i in val])

        test_data = rnn_utils.pad_sequence([torch.LongTensor(self.tokenizer(i[1],Voc)[0]) for i in TestData],padding_value=Voc.stoi['<pad>'],batch_first=True)
        test_length = torch.Tensor([self.tokenizer(i[1],Voc)[1] for i in TestData]).unsqueeze(dim=1)

        train_data = torch.cat([train_data,train_length,train_label],dim=1)
        val_data = torch.cat([val_data,val_length,val_label],dim=1)
        test_data = torch.cat([test_data,test_length],dim=1)

      
        train_data = Data.DataLoader(train_data,batch_size = self.args.batch_size,shuffle=True)
        val_data = Data.DataLoader(val_data,batch_size = 10000)
        #test_data = Data.DataLoader(test_data,batch_size = 10000)


        return train_data, val_data, test_data
        
    

