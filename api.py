import os
import gc
import uuid
import json
from time import time
from loguru import logger
import numpy as np 


from fastapi import FastAPI, Response, status, File, UploadFile, Body
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


from inference import Inferencer 

import params
from model import DiffVC

import sys
sys.path.append('hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN

sys.path.append('speaker_encoder/')
from encoder import inference as spk_encoder
from pathlib import Path


use_gpu = torch.cuda.is_available()
vc_path = 'checkpts/vc/vc_libritts_wodyn.pt' # path to voice conversion model

generator = DiffVC(params.n_mels, params.channels, params.filters, params.heads, 
                   params.layers, params.kernel, params.dropout, params.window_size, 
                   params.enc_dim, params.spk_dim, params.use_ref_t, params.dec_dim, 
                   params.beta_min, params.beta_max)
if use_gpu:
    generator = generator.cuda()
    generator.load_state_dict(torch.load(vc_path))
else:
    generator.load_state_dict(torch.load(vc_path, map_location='cpu'))
generator.eval()


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


# loading speaker encoder
enc_model_fpath = Path('checkpts/spk_encoder/pretrained.pt') # speaker encoder path
if use_gpu:
    spk_encoder.load_model(enc_model_fpath, device="cuda")
else:
    spk_encoder.load_model(enc_model_fpath, device="cpu")
# Define Inferencer 
_inferencer = Inferencer(generator, spk_encoder, hifigan_universal, MEDIA_ROOT, True )


# Make dir to save audio files log
MEDIA_ROOT = os.path.join('/logs', 'media')
if not os.path.exists(MEDIA_ROOT):
    os.makedirs(MEDIA_ROOT)

# Make dir to save json response log
LOG_ROOT = os.path.join('/logs', 'json')
if not os.path.exists(LOG_ROOT):
    os.makedirs(LOG_ROOT)

def save_audio(file):
    job_id = str(uuid.uuid4())
    output_dir = os.path.join(MEDIA_ROOT, str(job_id))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    audio_save_path = os.path.join(output_dir, file.filename)
    with open(audio_save_path, "wb+") as file_object:
        file_object.write(file.file.read())
    
    return audio_save_path 
    
    
    

app = FastAPI(
    title="Voice Conversion",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/', status_code=status.HTTP_200_OK)
async def check_status(response: Response):
    api_status = {"API Status": "Running"}
    return api_status



@app.post('/convert', status_code=200)
async def convert(response:Response, file1: UploadFile = File(...), file2: UploadFile = File(...) ):
    # Save source and target to MEDIA 
    source_fpath = save_audio(file1)
    target_fpath = save_audio(file2)
    
    audio = _inferencer.infer(src_path=audio_path, tgt_path=target_path, return_output_path=False)
    
    return audio 
    

