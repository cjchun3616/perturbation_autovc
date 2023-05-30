# Perturbation AUTOVC: Voice Conversion from Perturbation and Autoencoder Loss

This repository provides a PyTorch implementation of Perturbation AUTOVC.

### Audio Samples

The audio samples for Perturbation AUTOVC can ben found in [audio samples](https://github.com/cjchun3616/perturbation_autovc/tree/main/samples).

### Dependencies
- Python3
- Numpy
- PyTorch
- librosa
- tqdm
- parselmouth `pip install praat-parselmouth`
- torchaudio
- omegaconf
- HiFi-GAN vocoder
- ECAPA-TDNN
  
  
 ## Pre-trained models

 You can also use pretrained models we provide.
 
 [Download pretrained models](https://drive.google.com/drive/folders/1N3Uo4nM8vtWBqNmoYsqTlRxayM-3owbU?usp=sharing)
 
 Place pre-trained models at `./checkpoint`
 
 
 ### Speaker Encoder
 
 We use the ECAPA-TDNN as a speaker encoder.
 
 For more information, please refer to [ECAPA-TDNN](https://github.com/taoruijie/ecapa-tdnn)
 
 ### Vocoder
 
 We use the HiFi-GAN as a vocoder.
 
 Download pretrained HiFi-GAN config and checkpoint from [HiFi-GAN](http://github.com/jik876/hifi-gan) `pretrained/UNIVERSAL_V1`
 
 Place checkpoint at `./checkpoint` and config at `./configs`
  
  
 ## Datasets
 
 Datasets used when training are:
 - VCTK:
    - CSTR VCTK Corpus: English Multi speaker Corpus for CSTR Voice Coloning Toolkit
    - https://datashare.ed.ac.uk/handle/10283/2651

 Place datasets at `datasets/wavs/`
 
 
 ## Preprocess dataset.
 
 If you prefer `praat-parselmouth`, run `python make_metadata.py`
 
 ```python
parser = argparse.ArgumentParser()

parser.add_argument('--wav_dir', type=str, default='./datasets/wavs', help='path of wav directory')
parser.add_argument('--real_dir', type=str, default='./datasets/real', help='save path of original mel-spectrogram')
parser.add_argument('--perturb_dir', type=str, default='./datasets/perturb', help='save path of perturbation mel-spectrogram')
parser.add_argument('--save_path', type=str, default='./datasets/preprocess_data/emb', help='save path of metadata')
```

If the data is processed in advance, please pause this line.
```python 
make_data(wav_dir, real_dir, perturb_dir)
```

When this is done, `metadata.pkl` is created at `--save_path`.
 
 
 ## Training
 
Prefer `metadata.pkl`, including the speaker embedding, output of ECAPA-TDNN.

If you prefer `metadata.pkl`, run `python main.py`
 
  ```python
parser = argparse.ArgumentParser()

# Model configuration.
parser.add_argument('--lambda_cd', type=float, default=1, help='weight for hidden code loss')
parser.add_argument('--dim_emb', type=int, default=192, help='speaker embedding dimensions')
parser.add_argument('--dim_pre', type=int, default=512)
parser.add_argument('--freq', type=int, default=1, help='downsampling factor')

# Save configuration.
parser.add_argument('--resume', type=str, default=None, help='path to load model')
parser.add_argument('--save_dir', type=str, default='./model/test', help='path to save model')
parser.add_argument('--pt_name', type=str, default='test_model', help='model name')

# Data configuration.
parser.add_argument('--data_dir', type=str, default='./datasets/metadata.pkl', help='path to metatdata')

# Training configuration.
parser.add_argument('--batch_size', type=int, default=2, help='mini-batch size')
parser.add_argument('--num_iters', type=int, default=1000000, help='number of total iterations')
parser.add_argument('--len_crop', type=int, default=128, help='dataloader output sequence length')
parser.add_argument('--log_step', type=int, default=10000)
```

Converges when the reconstruction loss is around 0.01.

## Inference
 
 Run the `python conversion.py --source_path={} --target_path={}`
 
 You may want to edit `conversion.py` for custom manipulation.
 
 ```python
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
```

## Acknowledgment
This work...
 
## Citation
bibtex..
 
## License
license..
