# Diffusion-Based Any-to-Any Voice Conversion 

### Introduction
- This repository is a derivative of the Official implementation of the paper "Diffusion-Based Voice Conversion with Fast Maximum Likelihood Sampling Scheme" [Link](https://arxiv.org/abs/2109.13821). It builds upon their work and incorporates additional features and modifications specific to this project.


- [The Official Demo Page](https://diffvc-fast-ml-solver.github.io/).

# Pre-trained models

- Please check `inference.ipynb` for the detailed instructions.

- The pre-trained speaker encoder we use is available at https://drive.google.com/file/d/1Y8IO2_OqeT85P1kks9I9eeAq--S65YFb/view?usp=sharing
Please put it to `/checkpts/spk_encoder/`

- The pre-trained universal HiFi-GAN vocoder we use is available at https://drive.google.com/file/d/10khlrM645pTbQ4rc2aNEYPba8RFDBkW-/view?usp=sharing. It is taken from the official HiFi-GAN repository. Please put it to `/checkpts/vocoder/`

- You have to download voice conversion model trained on LibriTTS from here: https://drive.google.com/file/d/18Xbme0CTVo58p2vOHoTQm8PBGW7oEjAy/view?usp=sharing

- Additionally, we provide voice conversion model trained on VCTK: https://drive.google.com/file/d/12s9RPmwp9suleMkBCVetD8pub7wsDAy4/view?usp=sharing
. Please put models to `/checkpts/vc/`

# Build docker environment 

- To build image, run:
```bash
Docker build -t diffvc .
``` 

- To run a container for develop, run:
```bash
bash run-container.sh
```

# Training your own model

- To train model on your data, first create a data directory with three folders: "wavs", "mels" and "embeds". Put raw audio files sampled at 22.05kHz to "wavs" directory. The functions for calculating mel-spectrograms and extracting 256-dimensional speaker embeddings with the pre-trained speaker verification network located at *checkpts/spk_encoder/* can be found at *inference.ipynb* notebook (*get_mel* and *get_embed* correspondingly). Please put these data to "mels" and "embeds" folders respectively. Note that all the folders in your data directory should have subfolders corresponding to particular speakers and containing data only for corresponding speakers.

- If you want to train the encoder, create "logs_enc" directory and run *train_enc.py*. Before that, you have to prepare another folder "mels_mode" with mel-spectrograms of the "average voice" (i.e. target mels for the encoder) in the data directory. To obtain them, you have to run Montreal Forced Aligner on the input mels, get *.TextGrid* files and put them to "textgrids" folder in the data directory. Once you have "mels" and "textgrids" folders, run *get_avg_mels.ipynb*.
 `python3 -m scenario.train_enc`
- Alternatively, you may load the encoder trained on LibriTTS from https://drive.google.com/file/d/1JdoC5hh7k6Nz_oTcumH0nXNEib-GDbSq/view?usp=sharing and put it to "logs_enc" directory.

- Once you have the encoder *enc.pt* in "logs_enc" directory, create "logs_dec" directory and run *train_dec.py* to train the diffusion-based decoder.
`python3 -m scenario.train_dec`
- Please check *params.py* for the most important hyperparameters.

# Demo 

- To launch gradio demo app, run:
```bash
python3 app_gradio.py
``` 

# Serve model (developing)

1. Convert model from .pt to .onnx
```bash
python3 -m export_onnx.export_hifigan
```

```bash
python3 -m export_onnx.export_spk_enc
```

2. Deploy pipeline using Triton Inference Server: 

