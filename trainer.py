import torch
import pandas as pd
import json
import collections
import numpy as np
import torchtext.vocab as Vocab
import copy
import time
import torch.utils.data as Data
import torch.optim as optim
from config import Config
from data_loader import DataLoader
from sklearn import metrics
from sklearn.metrics import f1_score,precision_score,recall_score,log_loss

import numpy as np

from sklearn.model_selection import train_test_split
from model import *
import random
import torch.nn.functional as F
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVR,SVC



class Trainer(object):
    def __init__(self,args):
        self.config = Config()
        self.args = args

    
    def train_epoch(self,model, data_iter, criterion, optimizer):
        total_loss,total_words = 0.,0
        for sample in data_iter:
            data = Variable(sample[:,:self.config.max_l].long())
            length = Variable(sample[:,self.config.max_l])
            target = Variable(sample[:,self.config.max_l+1:])

            if self.args.use_cuda:
                data, target = data.cuda(), target.cuda()

            preds = model.forward(data,length)
            loss = criterion(preds, target)
            total_loss += loss.item()
            total_words += data.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return total_loss / total_words

    def validate(self,model,data_iter,criterion):
        for sample in data_iter:
            data = Variable(sample[:,:self.config.max_l].long())
            length = Variable(sample[:,self.config.max_l])
            target = Variable(sample[:,self.config.max_l+1:])

            if self.args.use_cuda:
                data, target = data.cuda(), target.cuda()

            preds = model.forward(data,length)

            loss = criterion(preds, target)/preds.shape[0]

            p,t = np.concatenate(preds.cpu().detach().numpy(),axis=0),np.concatenate(target.cpu().detach().numpy(),axis=0)

            score = log_loss(t,p.astype(np.float64))

            score = 1-score

        return loss, score
        

    def test(self,model,data_all,criterion):

        data = data_all[:,:self.config.max_l].long()
        length = data_all[:,self.config.max_l]

        
        if self.args.use_cuda:
            data = data.cuda()

        preds = model.forward(data,length)

        return preds.cpu().detach().numpy().astype(np.float64).tolist()


    def Train_Step(self):

        model_save = 'model/model:{},dropout:{},lr:{},weight_decay:{},batch_size:{},hidden_size:{},embedding_size:{}'.format(self.args.model,self.args.dropout,self.args.lr,self.args.weight_decay,self.args.batch_size,self.args.hidden_size,self.args.embedding_size)

        print('------------------------- Loading data -------------------------')
        dataloader = DataLoader(self.config,self.args)
        train_data, val_data, test_data = dataloader.data_load()

        print('\n------------------------- Initialize Model -------------------------')
        if self.args.model==0:
            model = GRU(self.config,self.args)

        elif self.args.model==1:
            model = LSTM(self.config,self.args)

        elif self.args.model==2:
            model = dis_attn(self.config,self.args)

        criterion = nn.BCELoss(reduction='sum')

        if self.args.use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()

        if self.args.test_mode == 0:

            optimizer = optim.Adam(params=model.parameters(),lr=self.args.lr,weight_decay=self.args.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience = self.args.patience, min_lr=self.args.min_lr)
            
            min_loss, AUC_max, early_stop_num = 1e9, .0, 0

            print('\n------------------------- Training -------------------------')
            for epoch in range(self.args.epoch_num):

                model.train()

                train_loss = self.train_epoch(model,train_data,criterion,optimizer)

                model.eval()
                val_loss, AUC_test = self.validate(model,val_data,criterion)

                scheduler.step(val_loss)

                print('lr:{}, early_stop_num:{}\n'.format(optimizer.param_groups[0]['lr'],early_stop_num))

                print('Epoch [%d] Model Training Loss: %f'% (epoch, train_loss))
                print('Epoch [%d] Validation Loss: %f; AUC_test: %f'% (epoch,val_loss,AUC_test))

                if val_loss < min_loss or AUC_test>AUC_max:
                    early_stop_num = 0
                    if AUC_test>AUC_max:
                        torch.save(model.state_dict(),model_save)
                        print("!!!!!!!!!! Model Saved !!!!!!!!!!")
                    
                    min_loss = min(val_loss,min_loss)
                    AUC_max = max(AUC_test,AUC_max)

                else:
                    if early_stop_num>=self.args.early_stop_num:
                        print('early stop!!!')
                        break
                    elif optimizer.param_groups[0]['lr']<=self.args.min_lr: 
                        early_stop_num += 1
                        print('early_stop_num:%d',early_stop_num)

        print('\n------------------------- Validate -------------------------')
        model.load_state_dict(torch.load(model_save))

        model.eval()
        
        val_loss, AUC = self.validate(model,val_data,criterion)
        print('Validation Loss: %f; AUC: %f'% (val_loss,AUC))

        #return [AUC,model_save]

        result = self.test(model,test_data,criterion)

        if self.args.test_mode==1:

            with open('result/test_result.csv','w') as f:
                for index, r in enumerate(result):
                    f.write(str(index))
                    f.write('|,|')
                    f.write(' '.join([str(i) for i in r]))
                    f.write('\n')


        