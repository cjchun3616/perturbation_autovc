import os
import torch
import pickle
import librosa
import argparse
import numpy as np
from glob import glob
from tqdm.auto import tqdm
import torch.nn.functional as F

from models import ecapa_tdnn
from utils.perturbation import make_data
from utils.perturbation import load_wav

class Speaker_Encoder(object):
    def __init__(self, config=None):
        if config is not None: # For make metadata.
            self.wav_dir = config.wav_dir # Audio file directory.
            self.speaker = sorted(os.listdir(self.wav_dir)) # Speaker name list.
            self.perturb_dir = config.perturb_dir
            self.save_path = config.save_path # Metadata save path.
            self.speakers = []
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        '''Load speaker encoder model.'''
        self.model_load()
        
    def model_load(self):
        self.model = ecapa_tdnn.ECAPA_TDNN(1024).to(self.device) 
        self.model.load_state_dict(torch.load('./checkpoint/tdnn.pt'))
        
    def extract_speaker(self, filepath):
        '''Extract speaker information with ecapa-tdnn model.'''
        audio, _ = load_wav(filepath, fs=16000) # Load wav
        with torch.no_grad():
            self.model.eval()
            feature1, _ = self.model(audio.float().to(self.device), False)
            feature1 = F.normalize(feature1, p=2, dim=1)
            emb = feature1.cpu().detach().numpy()
        return emb
        
    def make_metadata(self):
        '''Extract speaker embedding of each speaker'''
        for spk_name in tqdm(self.speaker, total=len(self.speaker)):         
            test_dir = glob(os.path.join(self.wav_dir, spk_name, '*.wav')) # wav file list.
            data_list = glob(os.path.join(self.perturb_dir, spk_name, '*npy')) # mel-spectrogram with perturbation list.
            Data1 = []
            # Extract speaker embedding per utterances of each speaker.
            for i in test_dir:
                utterances = []
                emb = self.extract_speaker(i)
                Data1.extend(emb.reshape(1, 192))
            Data1 = np.array(Data1)
            spk_emb = np.mean(Data1, axis=0) # Average of each utterance embedding.
            
            utterances.append(spk_name)
            utterances.append(spk_emb) # Speaker embedding with 192 dimensions. 
            utterances.extend(data_list) # Data path.
            self.speakers.append(utterances) 
            
        '''Save metadata at save path.'''
        with open(os.path.join(self.save_path, 'metadata.pkl'), 'wb') as handle:
            pickle.dump(self.speakers, handle)
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--wav_dir', type=str, default='./datasets/wavs', help='path to wav directory')
    parser.add_argument('--real_dir', type=str, default='./datasets/real', help='path to save mel-spectrogram')
    parser.add_argument('--perturb_dir', type=str, default='./datasets/perturb', help='path to save perturbation mel-spectrogram')
    parser.add_argument('--save_path', type=str, default='./datasets/', help='path to save metadata')
    
    config = parser.parse_args()
    print(config)
    make_data(config.wav_dir, config.real_dir, config.perturb_dir) # If the data is processed in advance, please comment.
    speaker_encoder = Speaker_Encoder(config)
    speaker_encoder.make_metadata()