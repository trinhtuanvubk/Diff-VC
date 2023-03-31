import argparse
import json
import os
import numpy as np
from tqdm import tqdm
import soundfile as sf
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


class Inferencer(): 
    def __init__(self, generator, spk_encoder, hifigan_universal, output_path="./output_demo", use_gpu=False):

        self.generator = generator
        self.spk_encoder = spk_encoder
        self.hifigan_universal = hifigan_universal
        # if not os.path.isdir(output_path):
        #     os.makedirs(output_path)
            
        self.output_path = output_path
        
        self.use_gpu = use_gpu
        
        
    def get_mel(self, wav_path):
        wav, _ = load(wav_path, sr=22050)
        wav = wav[:(wav.shape[0] // 256)*256]
        wav = np.pad(wav, 384, mode='reflect')
        stft = librosa.core.stft(wav, n_fft=1024, hop_length=256, win_length=1024, window='hann', center=False)
        stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + (1e-9))
        mel_spectrogram = np.matmul(mel_basis, stftm)
        log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
        return log_mel_spectrogram

    def get_embed(self, wav_path):
        wav_preprocessed = spk_encoder.preprocess_wav(wav_path)
        embed = spk_encoder.embed_utterance(wav_preprocessed)
        return embed

    def noise_median_smoothing(self, x, w=5):
        y = np.copy(x)
        x = np.pad(x, w, "edge")
        for i in range(y.shape[0]):
            med = np.median(x[i:i+2*w+1])
            y[i] = min(x[i+w+1], med)
        return y

    def mel_spectral_subtraction(self, mel_synth, mel_source, spectral_floor=0.02, silence_window=5, smoothing_window=5):
        mel_len = mel_source.shape[-1]
        energy_min = 100000.0
        i_min = 0
        for i in range(mel_len - silence_window):
            energy_cur = np.sum(np.exp(2.0 * mel_source[:, i:i+silence_window]))
            if energy_cur < energy_min:
                i_min = i
                energy_min = energy_cur
        estimated_noise_energy = np.min(np.exp(2.0 * mel_synth[:, i_min:i_min+silence_window]), axis=-1)
        if smoothing_window is not None:
            estimated_noise_energy = self.noise_median_smoothing(estimated_noise_energy, smoothing_window)
        mel_denoised = np.copy(mel_synth)
        for i in range(mel_len):
            signal_subtract_noise = np.exp(2.0 * mel_synth[:, i]) - estimated_noise_energy
            estimated_signal_energy = np.maximum(signal_subtract_noise, spectral_floor * estimated_noise_energy)
            mel_denoised[:, i] = np.log(np.sqrt(estimated_signal_energy))
        return mel_denoised


    def infer(self, src_path, tgt_path, n_timesteps=30, return_output_path=False, sr=16000): 
        
        source_basename = os.path.basename(src_path).split('.wav')[0]
        target_basename = os.path.basename(tgt_path).split('.wav')[0]
        output_basename = f'{source_basename}_to_{target_basename}'
        output_wav = os.path.join(self.output_path, output_basename+'.wav')
        
        mel_source = torch.from_numpy(self.get_mel(src_path)).float().unsqueeze(0)
        if self.use_gpu:
            mel_source = mel_source.cuda()
        mel_source_lengths = torch.LongTensor([mel_source.shape[-1]])
        if self.use_gpu:
            mel_source_lengths = mel_source_lengths.cuda()
        
        mel_target = torch.from_numpy(self.get_mel(tgt_path)).float().unsqueeze(0)
        if self.use_gpu:
            mel_target = mel_target.cuda()
        mel_target_lengths = torch.LongTensor([mel_target.shape[-1]])
        if self.use_gpu:
            mel_target_lengths = mel_target_lengths.cuda()

        embed_target = torch.from_numpy(self.get_embed(tgt_path)).float().unsqueeze(0)
        if self.use_gpu:
            embed_target = embed_target.cuda()
            
            
        # performing voice conversion
        mel_encoded, mel_ = self.generator.forward(mel_source, mel_source_lengths, mel_target, mel_target_lengths, embed_target, 
                                            n_timesteps=n_timesteps, mode='ml')
        mel_synth_np = mel_.cpu().detach().squeeze().numpy()
        mel_source_np = mel_.cpu().detach().squeeze().numpy()
        mel = torch.from_numpy(self.mel_spectral_subtraction(mel_synth_np, mel_source_np, smoothing_window=1)).float().unsqueeze(0)
        if self.use_gpu:
            mel = mel.cuda() 

        with torch.no_grad():
            audio = self.hifigan_universal.forward(mel).cpu().squeeze().clamp(-1, 1)
            print(audio.shape)
        sf.write(f'{output_wav}', audio, sr)
        
        if return_output_path:     
            return output_wav
        else: 
            return audio

