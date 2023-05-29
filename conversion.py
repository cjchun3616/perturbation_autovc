import os
import torch
import pickle
import librosa
import argparse
import numpy as np
import soundfile as sf

from math import ceil
from models.model_vc import Generator
from omegaconf import OmegaConf
from models.hifi_gan import Generator as hifigan_vocoder

from utils.mel import mel_spectrogram
from utils.perturbation import load_wav
from make_metadata import Speaker_Encoder

class Conversion(object):
    def __init__(self, config):
        # Inference configurations.
        self.source_path = config.source_path
        self.target_path = config.target_path
        self.save_path = config.save_path
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Voice Conversion model configuration.
        self.ckpt_path = config.ckpt_path
        self.dim_emb = config.dim_emb
        self.dim_pre = config.dim_pre
        self.freq = config.freq
        self.load_model()
        
        # Vocoder configurations.
        self.hifigan_config = './config/config.json'
        self.hifigan_ckpt = './checkpoint/g_02500000'
        self.vocoder()
        
        # Speaker Encoder configuration.
        self.speaker_encoder = Speaker_Encoder()
        
    def load_model(self):
        '''Load voice conversion model.'''
        self.G = Generator(self.dim_emb, self.dim_pre, self.freq)
        g_checkpoint = torch.load(self.ckpt_path)
        self.G.load_state_dict(g_checkpoint['model_state_dict'])
        self.G.to(self.device)
        self.G.eval()

    def vocoder(self):
        '''Load vocoder.'''
        hifigan_config = OmegaConf.load(self.hifigan_config)
        self.vocoder = hifigan_vocoder(hifigan_config)

        state_dict_g = torch.load(self.hifigan_ckpt)
        self.vocoder.to(self.device)
        self.vocoder.load_state_dict(state_dict_g['generator'])
        self.vocoder.eval()
    
    def pad_seq(self, x, base=32):
        len_out = int(base * ceil(float(x.shape[-1])/base))
        len_pad = len_out - x.shape[-1]
        assert len_pad >= 0
        x_org = np.pad(x, ((0,0),(0,len_pad)), 'constant')
        uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(self.device)
        return  uttr_org, len_pad

    def extract_energy(self, uttr_org):
        energy = torch.mean(uttr_org, dim=1, keepdim=True)
        return energy
    
    def conversion(self):
        '''Conversion process'''
        # Preprocess input data.
        source_wav, _ = load_wav(self.source_path, fs=22050)
        source = mel_spectrogram(source_wav) # Source mel-spectrogram.

        uttr_org, len_pad = self.pad_seq(source.squeeze()) # Energy.
        emb_trg = self.speaker_encoder.extract_speaker(self.target_path) # Target speaker embedding.
        emb_trg = torch.from_numpy(emb_trg).to(self.device)

        energy = self.extract_energy(uttr_org)
        emb_trg = emb_trg.unsqueeze(-1).expand(-1, -1, energy.shape[-1])
        model_input = torch.cat((emb_trg, energy), dim=1)
        
        # Conversion.
        with torch.no_grad():
            _, x_identic_psnt, a = self.G(uttr_org, model_input)
            
            if len_pad == 0:
                uttr_trg = x_identic_psnt[:, :, :]
            else:
                uttr_trg = x_identic_psnt[:, :, :-len_pad]
        self.save_wav(uttr_trg)
    
    def save_wav(self, uttr_trg):
        '''Save conversion waveform and mel-spectrogram at save path.'''
        # Make save directory.
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        # Reconstruction to waveform from mel-spectrogram.
        self.vocoder.eval()
        with torch.no_grad(): 
            conv_wav = self.vocoder(uttr_trg.squeeze().to(self.device))
            conv_wav = conv_wav.squeeze().detach().cpu().numpy()
            sf.write(os.path.join(self.save_path, 'conversion.wav'), conv_wav, samplerate=22050)
            np.save(os.path.join(self.save_path, 'conversion.npy'), uttr_trg.cpu().detach().numpy().squeeze(), allow_pickle=False)
        print("Conversion Successful")
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    # Conversion configurations.
    parser.add_argument('--source_path', type=str, required=True, help='path to source audio file, sr=22050')
    parser.add_argument('--target_path', type=str, required=True, help='path to target audio file, sr=16000')
    parser.add_argument('--save_path', type=str, default='./result', help='path to save conversion audio')
    
    # Model configurations.
    parser.add_argument('--ckpt_path', type=str, default='./checkpoint/model.pt', help='path to model checkpoint')
    parser.add_argument('--dim_emb', type=int, default=192, help='speaker embedding dimensions.')
    parser.add_argument('--dim_pre', type=int, default=512)
    parser.add_argument('--freq', type=int, default=1, help='downsampling factor')
    
    config = parser.parse_args()
    print(config)
    conversion = Conversion(config)
    conversion.conversion()
    