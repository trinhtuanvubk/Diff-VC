# vocoder 

import argparse
import json
import os
import numpy as np
import IPython.display as ipd
from tqdm import tqdm
from scipy.io.wavfile import write

import torch
use_gpu = torch.cuda.is_available()

import librosa
from librosa.core import load
from librosa.filters import mel as librosa_mel_fn
mel_basis = librosa_mel_fn(22050, 1024, 80, 0, 8000)

import params
from model import DiffVC

import sys
sys.path.append('hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN

sys.path.append('speaker_encoder/')
from encoder import inference as spk_encoder
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"]= "1"


# loading HiFi-GAN vocoder
hfg_path = 'checkpts/vocoder/' # HiFi-GAN path

with open(hfg_path + 'config.json') as f:
    h = AttrDict(json.load(f))

if use_gpu:
    hifigan_universal = HiFiGAN(h).cuda()
    hifigan_universal.load_state_dict(torch.load(hfg_path + 'generator')['generator'])
else:
    hifigan_universal = HiFiGAN(h)
    hifigan_universal.load_state_dict(torch.load(hfg_path + 'generator',  map_location='cpu')['generator'])

_ = hifigan_universal.eval()
hifigan_universal.remove_weight_norm()




def convert_torch_to_onnx_batch(model, output_path, dummy_input, device=None):

    input_names = ["mel_input"]
    output_names = ["audio_output"]
    
    if device!=None:
        model = model.to(device)
        dummy_input = dummy_input.to(device)
    
    torch.onnx.export(model, 
                 dummy_input, 
                 output_path, 
                 verbose=True, 
                 input_names=input_names, 
                 output_names=output_names,
                 dynamic_axes={'mel_input' : {0: 'batch_size', 2 : 'mel_leghths'},    # variable length axes
                               'audio_output' : {0:'batch_size', 2 : 'audio_lenghts'}})
    
device = torch.device('cuda')
output_path = "hifigan.onnx"
# dummy_input = mel_source
dummy_input = torch.rand(2,80,200)
dummy_output = torch.rand(2,1,124321)
convert_torch_to_onnx_batch(hifigan_universal, output_path, dummy_input, device=device)

print(device)