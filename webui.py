import argparse
import gradio as gr
from gradio import components
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
    model_path = './logs/' + model_path
    config_path = './configs/' + config_path
    hps = utils.get_hparams_from_file(config_path)

    if "use_mel_posterior_encoder" in hps.model.keys() and hps.model.use_mel_posterior_encoder == True:
        posterior_channels = 80
        hps.data.use_mel_posterior_encoder = True
    else:
        posterior_channels = hps.data.filter_length // 2 + 1
        hps.data.use_mel_posterior_encoder = False

    net_g = SynthesizerTrn(
        len(symbols),
        posterior_channels,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    _ = net_g.eval()
    _ = utils.load_checkpoint(model_path, net_g, None)

    stn_tst = get_text(text, hps)
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()

    with torch.no_grad():
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()

    output_wav_path = "output.wav"
    write(output_wav_path, hps.data.sampling_rate, audio)

    return output_wav_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model file.')
    parser.add_argument('--config_path', type=str, default=None, help='Path to the config file.')
    args = parser.parse_args()

    model_files = [f for f in os.listdir('./logs/') if f.endswith('.pth')]
    model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
    config_files = [f for f in os.listdir('./configs/') if f.endswith('.json')]

    default_model_file = args.model_path if args.model_path else (model_files[0] if model_files else None)
    default_config_file = args.config_path if args.config_path else 'config.json'

    gr.Interface(
        fn=tts,
        inputs=[components.Dropdown(model_files,value=default_model_file, label="Model File"), components.Dropdown(config_files,value=default_config_file, label="Config File"), components.Textbox(label="Text Input")],
        outputs=components.Audio(type='filepath', label="Generated Speech"),
        live=False
    ).launch()
