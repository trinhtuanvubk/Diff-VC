import gradio as gr
import os
import uuid
import torch
import json
from test_infer import Inferencer

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

#  load voice conversion 
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

# print(f'Number of parameters: {generator.nparams}')

# blocks 

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
    

# build inferencer
# output_path="output_demo"
_inferencer = Inferencer(generator, spk_encoder, hifigan_universal, "./output_demo", True)
# print(_inferencer)

# cpu_param = "--cpu" if not torch.cuda.is_available() else ""



def _inference(audio_path, target_path, mic_path1=None, mic_path2=None):
    # _inferencer = Inferencer(config=config, args=args)
    if mic_path1:
        audio_path = mic_path1
    if mic_path2:
        target_path = mic_path2
    # output_path = f"./output/output_{uuid.uuid4()}.wav"
    # output_path = "./test_wav/p226_001.wav"
    # os.system(
    #     f"python demo_cli.py --no_sound {cpu_param} --audio_path {audio_path} --text {shlex.quote(text.strip())} --output_path {output_path}")
    output_path = _inferencer.inference_demo(src_path=audio_path, tgt_path=target_path)
    # print("hihi")
    return output_path


title = "AGAIN-VC-DEMO"
description = "Gradio demo for Voice Conversion"
# article = "<p style='text-align: center'><a href='https://matheo.uliege.be/handle/2268.2/6801' target='_blank'>Voice Cloning Demo</a> | <a href='https://github.com/CorentinJ/Real-Time-Voice-Cloning' target='_blank'>Github Repo</a></p>"

examples = [['./test_wav/p225_001.wav', "./test_wav/p226_001.wav"]]
# server_port = 1400

def toggle(choice):
    if choice == "mic":
        return gr.update(visible=True, value=None), gr.update(visible=False, value=None)
    else:
        return gr.update(visible=False, value=None), gr.update(visible=True, value=None)


with gr.Blocks() as demo:
    with gr.Row():
    #     with gr.Column():
    #         radio = gr.Radio(["mic", "file"], value="mic",
    #                          label="How would you like to upload your audio?")
    #         mic_input = gr.Mic(label="Input", type="filepath", visible=False)
    #         audio_input = gr.Audio(
    #             type="filepath", label="Input", visible=True)
    #         audio_target = gr.Audio(
    #             type="filepath", label="Target", visible=True)
    #     with gr.Column():
    #         audio_output = gr.Audio(label="Output")

    # # gr.Examples(examples, fn=_inference, inputs=[audio_input, audio_target],
    # #                   outputs=audio_output, cache_examples=True)
    # btn = gr.Button("Generate")
    # btn.click(_inference, inputs=[audio_input,
    #           audio_target, mic_input], outputs=audio_output)
    # radio.change(toggle, radio, [mic_input, audio_input])
        with gr.Column():
            radio1 = gr.Radio(["mic", "file"], value="file",
                             label="How would you like to upload your audio?")

            mic_input1 = gr.Mic(label="Input", type="filepath", visible=False)
            audio_input = gr.Audio(
                type="filepath", label="Input", visible=True)
            
            radio2 = gr.Radio(["mic", "file"], value="file",
                            label="How would you like to upload your audio?")
            mic_input2 = gr.Mic(label="Target", type="filepath", visible=False)
            audio_target = gr.Audio(
                type="filepath", label="Target", visible=True)
        with gr.Column():
            audio_output = gr.Audio(label="Output")

    # gr.Examples(examples, fn=_inference, inputs=[audio_input, audio_target],
    #                   outputs=audio_output, cache_examples=True)
    btn = gr.Button("Generate")
    btn.click(_inference, inputs=[audio_input,
              audio_target, mic_input1, mic_input2], outputs=audio_output)
    radio1.change(toggle, radio1, [mic_input1, audio_input])
    radio2.change(toggle, radio2, [mic_input2, audio_target])

demo.launch(enable_queue=True, server_port=1402, server_name="0.0.0.0", share=True)