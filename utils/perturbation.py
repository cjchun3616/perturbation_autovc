import os
import librosa
import torchaudio
import numpy as np
from tqdm.auto import tqdm

from utils import function_f
from utils.mel import mel_spectrogram

'''Generate mel-spectrogram data for training'''
def load_wav(filepath, fs):
    # Load wav.
    audio, sr = torchaudio.load(filepath)
    resampler = torchaudio.transforms.Resample(sr, fs)
    audio = resampler(audio)
    return audio, fs

def make_data(wav_dir, real_dir, perturb_dir):
    
    dirName, subdirList, _ = next(os.walk(wav_dir))

    for subdir in tqdm(sorted(subdirList), total=len(subdirList)):
        # make directory.
        if not os.path.exists(os.path.join(perturb_dir, subdir)):
            os.makedirs(os.path.join(perturb_dir, subdir))
            os.makedirs(os.path.join(real_dir, subdir))
            
        _,_, fileList = next(os.walk(os.path.join(dirName,subdir)))
        for fileName in sorted(fileList):

            x, fs = load_wav(os.path.join(dirName,subdir,fileName), fs=22050)
            x_perturb = function_f.f(x, sr=fs) # Perturb audio.

            mel_perturb = mel_spectrogram(x_perturb) # extract mel-spectrogram from perturbation audio.
            mel_real = mel_spectrogram(x) # extract mel-spectrogram.

            np.save(os.path.join(perturb_dir, subdir, fileName[:-4]), mel_perturb, allow_pickle=False)
            np.save(os.path.join(real_dir, subdir, fileName[:-4]), mel_real, allow_pickle=False)
    