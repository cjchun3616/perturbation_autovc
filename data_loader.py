import os
import torch
import random
import pickle
import numpy as np
from torch.utils import data 
from multiprocessing import Process, Manager   

class Utterances(data.Dataset):

    def __init__(self, root_dir, spk_path, len_crop):

        self.root_dir = root_dir
        self.len_crop = len_crop
        self.spk_path = spk_path
        self.step = 10
        
        data_path = os.path.join(root_dir, self.spk_path)
        meta = pickle.load(open(data_path, "rb"))
        # meta = meta[:-20] # unseen speaker
        
        '''Load data using multiprocessing'''
        manager = Manager()
        meta = manager.list(meta)
        dataset = manager.list(len(meta)*[None])  
        processes = []
        for i in range(0, len(meta), self.step):
            p = Process(target=self.load_data, 
                        args=(meta[i:i+self.step], dataset, i))  
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
            
        self.train_dataset = dataset
        self.num_tokens = len(self.train_dataset)
        
        print('Finished loading the dataset...')
        
        
    def load_data(self, submeta, dataset, idx_offset):  
        for k, sbmt in enumerate(submeta):    
            uttrs = len(sbmt)*[None]
            for j, tmp in enumerate(sbmt):
                if j < 2: # name, speaker embedding
                    uttrs[j] = tmp
                else: # Data path
                    tmp_uttr = tmp
                    x_real = tmp.replace('perturb', 'real')
                    uttrs[j] = [tmp_uttr, x_real] 
            dataset[idx_offset+k] = uttrs
                   
        
    def __getitem__(self, index):
        
        dataset = self.train_dataset
        
        list_uttrs = dataset[index]
        emb_org = list_uttrs[1]

        # Pick random utterence with random crop.
        a = np.random.randint(2, len(list_uttrs))
        tmp = list_uttrs[a]
        
        # Load the mel-spectrogram.
        uttr = np.load(tmp[0]) # mel-spectrogram with perturbation.
        x_real = np.load(tmp[1]) # original mel-spectrogram
        
        if uttr.shape[-1] < self.len_crop:
            len_pad = self.len_crop - uttr.shape[-1]
            uttr_x = np.pad(uttr, ((0,0), (0,0),(0,len_pad)), 'constant')
            real_x = np.pad(x_real, ((0,0), (0,0),(0,len_pad)), 'constant')
            
        elif uttr.shape[-1] > self.len_crop:
            left = np.random.randint(uttr.shape[-1]-self.len_crop)
            uttr_x = uttr[:, :, left:left+self.len_crop]
            real_x = x_real[:, :, left:left+self.len_crop]
        else:
            uttr_x = uttr
            real_x = x_real

        return torch.from_numpy(real_x).squeeze(), torch.from_numpy(uttr_x).squeeze(), emb_org
    
    def __len__(self):
        '''Return the number of speakers'''
        return self.num_tokens
    

def get_loader(root_dir, spk_path, batch_size=16, len_crop=128, num_workers=0): 
    '''Build and return a data loader'''
    dataset = Utterances(root_dir, spk_path, len_crop)
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2**32))
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  worker_init_fn=worker_init_fn)
    return data_loader

    





