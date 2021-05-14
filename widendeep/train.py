# coding: utf-8
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import dump, load
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# from prepare_data import prepare_deep, prepare_wide
from .preprocessor import WidePreprocessor, DeepPreprocessor
from .datareader import WideDeepLoader
from .widendeep import Wide, DeepDense, WideDeep
from .utils import dotdict, save, json_save, json_load

class Main(object):
    """
    Class for Model Train, Evaluate, Prediction, Retrain and Get RL features
    Args:
        data: input data (pd.Data.Frame)
        cfg_dir: model and train config.json file directory
        model_dir: trained model directory
    """   
    def __init__(self, cfg, model_dir = None):
        
        self.cfg = cfg
        self.wide_parms = dotdict(cfg['wide'])
        self.deep_parms = dotdict(cfg['deep'])
        self.model_parms = dotdict(cfg['model_cfg'])
        
        if model_dir is not None:
            self.model_parms.model_dir = model_dir
            
        self.device = 'cuda:'+str(self.model_parms.gpu_ids[0]) if self.model_parms.use_gpu else 'cpu'
        
    def train(self, df):
        
        self.data = df.reset_index(drop = True)
        
        wide_parms = self.wide_parms
        deep_parms = self.deep_parms
        model_parms = self.model_parms
        
        path_list = save(model_parms) # conf / model / log / preprocess
        writer = SummaryWriter(path_list[2])
        
        wide_pre = WidePreprocessor(**wide_parms)
        deep_pre = DeepPreprocessor(**deep_parms)
        
        self.X_wide = wide_pre.fit_transform(self.data)
        self.X_deep = deep_pre.fit_transform(self.data)
        self.y = np.array(self.data[self.model_parms.target], dtype = np.float)
            
        dump(wide_pre, path_list[3]+'/wide_pre.joblib')
        dump(deep_pre, path_list[3]+'/deep_pre.joblib')
        
        model_parms['wide_dim'] = np.unique(self.X_wide).shape[0]
        model_parms['embeddings_input'] = deep_pre.embeddings_input
        model_parms['embeddings_encoding_dict'] = deep_pre.label_encoder.encoding_dict
        model_parms['deep_column_idx'] = deep_pre.column_idx
        model_parms['continuous_cols'] = deep_pre.continuous_cols
        
        conf = {}
        conf['wide_parms'] = wide_parms
        conf['deep_parms'] = deep_parms
        conf['model_parms'] = model_parms        
        json_save(path_list[0]+'/conf.json', conf)
        
        self.dataset = WideDeepLoader(
            {
                'wide' : self.X_wide,
                'deep_dense' : self.X_deep,
                'target' : self.y
            }
        )
        
        dataloader = DataLoader(self.dataset, 
                                batch_size=model_parms.batch_size, 
                                shuffle=True, 
                                num_workers=model_parms.num_workers)
        
        model =  WideDeep(wide_dim = model_parms.wide_dim,
                          embeddings_input = model_parms.embeddings_input, 
                          embeddings_encoding_dict = model_parms.embeddings_encoding_dict, 
                          continuous_cols = model_parms.continuous_cols,
                          deep_column_idx = model_parms.deep_column_idx, 
                          hidden_layers = model_parms.hidden_layers,
                          dropout = model_parms.dropout,
                          output_dim = model_parms.output_dim).float()
        
        model.to(self.device)
        if model_parms.use_gpu:
            model = nn.DataParallel(model, device_ids=model_parms.gpu_ids)
            
        optimizer = optim.Adam(model.parameters(), lr=model_parms.lr)
        criterion = nn.BCEWithLogitsLoss()  ## binary classification
        
        model.train()
        global_step = 0
        for epoch in range(model_parms.n_epochs):
            loss_sum = 0. # the sum of iteration losses to get average loss in every epoch
            iter_bar = tqdm(dataloader, desc='Iter (loss=X.XXX)')

            for i, batch in enumerate(iter_bar): 
                batch = [t.float().to(self.device) for t in batch]
                labels = batch[-1]
                
                optimizer.zero_grad()
                logits = model(*batch[:-1])  ##  x_wide, x_deep_dense

                loss = criterion(logits.squeeze(), labels).mean() # mean() for Data Parallelism
                loss.backward()
                optimizer.step()

                global_step += 1
                loss_sum += loss.item()
                iter_bar.set_description('Iter (loss=%5.3f)'%loss.item())
                
                writer.add_scalar('Loss', loss.item(), global_step)

                if global_step % 5 == 0:
                    correct = ((logits.sigmoid()>=0.5).float().squeeze() == labels).sum()
                    writer.add_scalar('Accuracy', correct/len(labels), global_step)

            print('Epoch %d/%d : Average Loss %5.3f'%(epoch+1, model_parms['n_epochs'], loss_sum/(i+1)))
            torch.save(model.state_dict(), path_list[1]+'/Epoch_{}.pt'.format(epoch+1))
    
    def eval(self, df):
        self.data = df.reset_index(drop = True)
        
        model_parms = self.model_parms
        
        wide_pre = load(model_parms.preprocessor_dir+'/wide_pre.joblib')
        deep_pre = load(model_parms.preprocessor_dir+'/deep_pre.joblib')
        
        self.X_wide = wide_pre.transform(self.data)
        self.X_deep = deep_pre.transform(self.data)
        self.y = np.array(self.data[self.model_parms.target], dtype = np.float)

        self.dataset = WideDeepLoader(
            {
                'wide' : self.X_wide,
                'deep_dense' : self.X_deep,
                'target' : self.y
            }
        )
    
        dataloader = DataLoader(self.dataset, 
                                batch_size=model_parms.batch_size, 
                                shuffle=False, 
                                num_workers=model_parms.num_workers)
        
        model_parms['wide_dim'] = np.unique(self.X_wide).shape[0]
        model_parms['embeddings_input'] = deep_pre.embeddings_input
        model_parms['embeddings_encoding_dict'] = deep_pre.label_encoder.encoding_dict
        model_parms['deep_column_idx'] = deep_pre.column_idx
        model_parms['continuous_cols'] = deep_pre.continuous_cols
        
        model =  WideDeep(wide_dim = model_parms.wide_dim,
                          embeddings_input = model_parms.embeddings_input, 
                          embeddings_encoding_dict = model_parms.embeddings_encoding_dict, 
                          continuous_cols = model_parms.continuous_cols,
                          deep_column_idx = model_parms.deep_column_idx, 
                          hidden_layers = model_parms.hidden_layers,
                          dropout = model_parms.dropout,
                          output_dim = model_parms.output_dim).float()
        model.to(self.device)

        if model_parms.use_gpu:
            model = nn.DataParallel(model, device_ids=model_parms.gpu_ids)
            model.load_state_dict(torch.load(model_parms.model_dir))
        else:
            state_dict = torch.load(model_parms.model_dir)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            
        model.eval()
        corrects = []
        iter_bar = tqdm(dataloader)
        for i, batch in enumerate(iter_bar): 
            batch = [t.float().to(self.device) for t in batch]
            labels = batch[-1]

            with torch.no_grad():
                logits = model(*batch[:-1])
                correct = ((logits.sigmoid() >= 0.5).float().squeeze() == labels).cpu()
                corrects += correct.tolist()
                
        accuracy = np.mean(corrects)       
        return accuracy
        
    def pred(self, df):
        self.data = df.reset_index(drop = True)
        
        model_parms = self.model_parms
        
        wide_pre = load(model_parms.preprocessor_dir+'/wide_pre.joblib')
        deep_pre = load(model_parms.preprocessor_dir+'/deep_pre.joblib')
        
        self.X_wide = wide_pre.transform(self.data)
        self.X_deep = deep_pre.transform(self.data)
#         self.y = np.array(self.data[self.model_parms.target], dtype = np.float)

        self.dataset = WideDeepLoader(
            {
                'wide' : self.X_wide,
                'deep_dense' : self.X_deep,
                'target' : None
            }
        )
    
        dataloader = DataLoader(self.dataset, 
                                batch_size=model_parms.batch_size, 
                                shuffle=False, 
                                num_workers=model_parms.num_workers)
        
        model_parms['wide_dim'] = np.unique(self.X_wide).shape[0]
        model_parms['embeddings_input'] = deep_pre.embeddings_input
        model_parms['embeddings_encoding_dict'] = deep_pre.label_encoder.encoding_dict
        model_parms['deep_column_idx'] = deep_pre.column_idx
        model_parms['continuous_cols'] = deep_pre.continuous_cols
        
        model =  WideDeep(wide_dim = model_parms.wide_dim,
                          embeddings_input = model_parms.embeddings_input, 
                          embeddings_encoding_dict = model_parms.embeddings_encoding_dict, 
                          continuous_cols = model_parms.continuous_cols,
                          deep_column_idx = model_parms.deep_column_idx, 
                          hidden_layers = model_parms.hidden_layers,
                          dropout = model_parms.dropout,
                          output_dim = model_parms.output_dim).float()
        model.to(self.device)

        if model_parms.use_gpu:
            model = nn.DataParallel(model, device_ids=model_parms.gpu_ids)
            model.load_state_dict(torch.load(model_parms.model_dir))
        else:
            state_dict = torch.load(model_parms.model_dir)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            
        model.eval()
        preds = []
        iter_bar = tqdm(dataloader)
        for i, batch in enumerate(iter_bar): 
            batch = [t.float().to(self.device) for t in batch]
#             labels = batch[-1]

            with torch.no_grad():
                logits = model(*batch)
                pred = logits.sigmoid().squeeze().cpu()
                preds += pred.tolist()
            
        return preds
        
               
    def retrain(self, df):
        
        self.data = df.reset_index(drop = True)
        
        wide_parms = self.wide_parms
        deep_parms = self.deep_parms
        model_parms = self.model_parms
        
        path_list = save(model_parms) # conf / model / log
        writer = SummaryWriter(path_list[2])
        
        wide_pre = load(model_parms.preprocessor_dir+'/wide_pre.joblib')
        deep_pre = load(model_parms.preprocessor_dir+'/deep_pre.joblib')
        
        self.X_wide = wide_pre.transform(self.data)
        self.X_deep = deep_pre.transform(self.data)
        self.y = np.array(self.data[self.model_parms.target], dtype = np.float)
            
        dump(wide_pre, path_list[3]+'/wide_pre.joblib')
        dump(deep_pre, path_list[3]+'/deep_pre.joblib')
        
        model_parms['wide_dim'] = np.unique(self.X_wide).shape[0]
        model_parms['embeddings_input'] = deep_pre.embeddings_input
        model_parms['embeddings_encoding_dict'] = deep_pre.label_encoder.encoding_dict
        model_parms['deep_column_idx'] = deep_pre.column_idx
        model_parms['continuous_cols'] = deep_pre.continuous_cols
        
        conf = {}
        conf['wide_parms'] = wide_parms
        conf['deep_parms'] = deep_parms
        conf['model_parms'] = model_parms        
        json_save(path_list[0]+'/conf.json', conf)
        
        self.dataset = WideDeepLoader(
            {
                'wide' : self.X_wide,
                'deep_dense' : self.X_deep,
                'target' : self.y
            }
        )
        
        dataloader = DataLoader(self.dataset, 
                                batch_size=model_parms.batch_size, 
                                shuffle=True, 
                                num_workers=model_parms.num_workers)
        
        model =  WideDeep(wide_dim = model_parms.wide_dim,
                          embeddings_input = model_parms.embeddings_input, 
                          embeddings_encoding_dict = model_parms.embeddings_encoding_dict, 
                          continuous_cols = model_parms.continuous_cols,
                          deep_column_idx = model_parms.deep_column_idx, 
                          hidden_layers = model_parms.hidden_layers,
                          dropout = model_parms.dropout,
                          output_dim = model_parms.output_dim).float()       
        model.to(self.device)

        if model_parms.use_gpu:
            model = nn.DataParallel(model, device_ids=model_parms.gpu_ids)
            model.load_state_dict(torch.load(model_parms.model_dir))
        else:
            state_dict = torch.load(model_parms.model_dir)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            
        optimizer = optim.Adam(model.parameters(), lr=model_parms.lr)
        criterion = nn.BCEWithLogitsLoss()  ## binary classification
        
        model.train()
        global_step = 0
        for epoch in range(model_parms.n_epochs):
            loss_sum = 0. # the sum of iteration losses to get average loss in every epoch
            iter_bar = tqdm(dataloader, desc='Iter (loss=X.XXX)')

            for i, batch in enumerate(iter_bar): 
                batch = [t.float().to(self.device) for t in batch]
                labels = batch[-1]
                
                optimizer.zero_grad()
                logits = model(*batch[:-1])  ##  x_wide, x_deep_dense

                loss = criterion(logits.squeeze(), labels).mean() # mean() for Data Parallelism
                loss.backward()
                optimizer.step()

                global_step += 1
                loss_sum += loss.item()
                iter_bar.set_description('Iter (loss=%5.3f)'%loss.item())
                
                writer.add_scalar('Loss', loss.item(), global_step)

                if global_step % 5 == 0:
                    correct = ((logits.sigmoid()>=0.5).float().squeeze() == labels).sum()
                    writer.add_scalar('Accuracy', correct/len(labels), global_step)

            print('Epoch %d/%d : Average Loss %5.3f'%(epoch+1, model_parms['n_epochs'], loss_sum/(i+1)))
            torch.save(model.state_dict(), path_list[1]+'/Epoch_{}.pt'.format(epoch+1))