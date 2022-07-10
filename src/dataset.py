import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
from scipy import signal
import torch

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
    

class Context_Datasets(Dataset):
    def __init__(self, dataset_path, data='EMOTIC', split_type='train', if_align=False):
        super(Context_Datasets, self).__init__()
        dataset_path = os.path.join(dataset_path, data+'_data.pkl' if if_align else data+'_data.pkl' )
        dataset = pickle.load(open(dataset_path, 'rb'))

     
        self.concontext21 = torch.tensor(dataset[split_type]['context21'].astype(np.float32)).cpu().detach()
        self.context2 = torch.tensor(dataset[split_type]['context2'].astype(np.float32)).cpu().detach()
        self.context3 = torch.tensor(dataset[split_type]['context3'].astype(np.float32)).cpu().detach()
        self.context4 = dataset[split_type]['context4'].astype(np.float32)
        self.context4[self.context4 == -np.inf] = 0
        self.context4 = torch.tensor(self.context4).cpu().detach()
        self.labels = torch.tensor(dataset[split_type]['labels'].astype(np.float32)).cpu().detach()
        
     
        self.meta = dataset[split_type]['id'] if 'id' in dataset[split_type].keys() else None
       
        self.data = data
        
        self.n_modalities = 4 # context1/ context2/ context3/ context4
    def get_n_modalities(self):
        return self.n_modalities
    def get_seq_len(self):
        return self.context1.shape[1], self.context2.shape[1], self.context3.shape[1], self.context4.shape[1]
    def get_dim(self):
        return self.context1.shape[2], self.context2.shape[2], self.context3.shape[2], self.context4.shape[2]
    def get_lbl_info(self):
     
        return self.labels.shape[1], self.labels.shape[2]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        X = (index, self.context1[index], self.context2[index], self.context3[index], self.context4[index])
        Y = self.labels[index]
        META = (0,0,0,0) if self.meta is None else (self.meta[index][0], self.meta[index][1], self.meta[index][2], self.meta[index][3])
        if self.data == 'GroupWalk':
            META = (self.meta[index][0].decode('UTF-8'), self.meta[index][1].decode('UTF-8'), self.meta[index][2].decode('UTF-8'), self.meta[index][3].decode('UTF-8'))
        if self.data == 'HECO':
            Y = torch.argmax(Y, dim=-1)
        return X, Y, META        

