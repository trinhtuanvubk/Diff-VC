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
# sys.path.append('hifi-gan/')
# from env import AttrDict
# from models import Generator as HiFiGAN

sys.path.append('speaker_encoder/')
from encoder import inference as spk_encoder
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"]= "1"

from encoder.model import SpeakerEncoder


weights_fpath = Path('checkpts/spk_encoder/pretrained.pt')
_device = torch.device('cuda')

_model = SpeakerEncoder(_device, torch.device("cpu"))
checkpoint = torch.load(weights_fpath, map_location="cuda")
_model.load_state_dict(checkpoint["model_state"])
_model.eval()

def convert_torch_to_onnx_batch(model, output_path, dummy_input, device=None):
    
    input_names = ["frame_input"]
    output_names = ["embed_output"]
    
    if device!=None:
        model = model.to(device)
        dummy_input = dummy_input.to(device)
    
    torch.onnx.export(model, 
                 dummy_input, 
                 output_path, 
                 verbose=True, 
                 input_names=input_names, 
                 output_names=output_names,
                 dynamic_axes={'frame_input' : {0: 'batch_size'},    # variable length axes
                               'embed_output' : {0:'batch_size'}})
print("hihi")
device = torch.device('cuda')
output_path = "spk_enc.onnx"
# dummy_input = mel_source
dummy_input = torch.rand(2, 10, 160, 40)
dummpy_ouput = torch.rand(10,256)

convert_torch_to_onnx_batch(_model, output_path, dummy_input, device=device)

# print(device)


