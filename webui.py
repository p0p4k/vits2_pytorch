import gradio as gr
import os
import torch
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from scipy.io.wavfile import write


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def tts(model_path, config_path, text):
    # Prepend the directory paths to model and config file names
    model_path = './logs/' + model_path
    config_path = './configs/' + config_path
    # Load Hyperparameters
    hps = utils.get_hparams_from_file(config_path)
    
    # Choose Posterior Channels
    if "use_mel_posterior_encoder" in hps.model.keys() and hps.model.use_mel_posterior_encoder == True:
        posterior_channels = 80 #vits2
        hps.data.use_mel_posterior_encoder = True
    else:
        posterior_channels = hps.data.filter_length // 2 + 1  
        hps.data.use_mel_posterior_encoder = False

    # Load Model
    net_g = SynthesizerTrn(
        len(symbols),
        posterior_channels,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    _ = net_g.eval()

    _ = utils.load_checkpoint(model_path, net_g, None)
    
    # Prepare Text
    stn_tst = get_text(text, hps)  # Reuse get_text() to make stn_tst a tensor
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    
    # Perform Inference
    with torch.no_grad():
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
    
    # Save audio
    output_wav_path = "output.wav"
    write(output_wav_path, hps.data.sampling_rate, audio)
    
    return output_wav_path

# List of available .pth and .json files
model_files = [f for f in os.listdir('./logs/') if f.endswith('.pth')]
config_files = [f for f in os.listdir('./configs/') if f.endswith('.json')]

# Get the first item from each list to use as default
default_model_file = model_files[0] if model_files else None
default_config_file = 'vits2_ljs_nosdp.json'

iface = gr.Interface(
    fn=tts, 
    inputs=[
        gr.Dropdown(model_files, value=default_model_file, label="Model File"),  # Set the default
        gr.Dropdown(config_files, value=default_config_file, label="Config File"),  # Set the default
        gr.Textbox(value="Hello world.", label="Text")
    ], 
    outputs=gr.Audio(type="filepath", label="Generated Speech")
)

iface.launch()
