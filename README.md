# VITS2: Improving Quality and Efficiency of Single-Stage Text-to-Speech with Adversarial Learning and Architecture Design
### Jungil Kong, Jihoon Park, Beomjeong Kim, Jeongmin Kim, Dohee Kong, Sangjin Kim 
Unofficial implementation of the [VITS2 paper](https://arxiv.org/abs/2307.16430), sequel to [VITS paper](https://arxiv.org/abs/2106.06103). (thanks to the authors for their work!)

![Alt text](resources/image.png)

Single-stage text-to-speech models have been actively studied recently, and their results have outperformed two-stage pipeline systems. Although the previous single-stage model has made great progress, there is room for improvement in terms of its intermittent unnaturalness, computational efficiency, and strong dependence on phoneme conversion. In this work, we introduce VITS2, a single-stage text-to-speech model that efficiently synthesizes a more natural speech by improving several aspects of the previous work. We propose improved structures and training mechanisms and present that the proposed methods are effective in improving naturalness, similarity of speech characteristics in a multi-speaker model, and efficiency of training and inference. Furthermore, we demonstrate that the strong dependence on phoneme conversion in previous works can be significantly reduced with our method, which allows a fully end-toend single-stage approach.

## Credits
- We will build this repo based on the [VITS repo](https://github.com/jaywalnut310/vits). The goal is to make this model easier to transfer learning from VITS pretrained model!
- (08-17-2023) - The authors were really kind to guide me through the paper and answer my questions. I am open to discuss any changes or answer questions regarding the implementation. Please feel free to open an issue or contact me directly.

# Pretrained checkpoints
- [LJSpeech-no-sdp](https://drive.google.com/drive/folders/1U-1EqBMXqmEqK0aUhbCJOquowbvKkLmc?usp=sharing) (refer to config.yaml in this checkppoint folder) | 64k steps | proof that training works!
Would recommend experts to rename the ckpts to *_0.pth and starting the training using transfer learning. (I will add a notebook for this soon to help beginers).
- [x] Check 'Discussion' page for training logs and tensorboard links and other community contributions.

# Sample audio
- (08/23/2023) - Some samples on non-native EN dataset [discussion page](https://github.com/p0p4k/vits2_pytorch/discussions/18). Thanks to [@athenasaurav](https://github.com/athenasaurav) for using his private GPU resources and dataset!
- (08/20/2023) - Added sample audio @104k steps. [ljspeech-nosdp](resources/test.wav) ; [tensorboard](https://github.com/p0p4k/vits2_pytorch/discussions/12)
- [vietnamese samples](https://github.com/p0p4k/vits2_pytorch/pull/10#issuecomment-1682307529) Thanks to [@ductho9799](https://github.com/ductho9799) for sharing!

## Prerequisites
1. Python >= 3.10
2. Tested on Pytorch version 1.13.1 with Google Colab and LambdaLabs cloud.
3. Clone this repository
4. Install python requirements. Please refer [requirements.txt](requirements.txt)
    1. You may need to install espeak first: `apt-get install espeak`
5. Download datasets
    1. Download and extract the LJ Speech dataset, then rename or create a link to the dataset folder: `ln -s /path/to/LJSpeech-1.1/wavs DUMMY1`
    1. For mult-speaker setting, download and extract the VCTK dataset, and downsample wav files to 22050 Hz. Then rename or create a link to the dataset folder: `ln -s /path/to/VCTK-Corpus/downsampled_wavs DUMMY2`
6. Build Monotonic Alignment Search and run preprocessing if you use your own datasets.

```sh
# Cython-version Monotonoic Alignment Search
cd monotonic_align
python setup.py build_ext --inplace

# Preprocessing (g2p) for your own datasets. Preprocessed phonemes for LJ Speech and VCTK have been already provided.
# python preprocess.py --text_index 1 --filelists filelists/ljs_audio_text_train_filelist.txt filelists/ljs_audio_text_val_filelist.txt filelists/ljs_audio_text_test_filelist.txt 
# python preprocess.py --text_index 2 --filelists filelists/vctk_audio_sid_text_train_filelist.txt filelists/vctk_audio_sid_text_val_filelist.txt filelists/vctk_audio_sid_text_test_filelist.txt
```

## How to run (dry-run)
- model forward pass (dry-run)
```python
import torch
from models import SynthesizerTrn

net_g = SynthesizerTrn(
    n_vocab=256,
    spec_channels=80, # <--- vits2 parameter (changed from 513 to 80)
    segment_size=8192,
    inter_channels=192,
    hidden_channels=192,
    filter_channels=768,
    n_heads=2,
    n_layers=6,
    kernel_size=3,
    p_dropout=0.1,
    resblock="1", 
    resblock_kernel_sizes=[3, 7, 11],
    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    upsample_rates=[8, 8, 2, 2],
    upsample_initial_channel=512,
    upsample_kernel_sizes=[16, 16, 4, 4],
    n_speakers=0,
    gin_channels=0,
    use_sdp=True, 
    use_transformer_flows=True, # <--- vits2 parameter
    # (choose from "pre_conv", "fft", "mono_layer_inter_residual", "mono_layer_post_residual")
    transformer_flow_type="fft", # <--- vits2 parameter 
    use_spk_conditioned_encoder=True, # <--- vits2 parameter
    use_noise_scaled_mas=True, # <--- vits2 parameter
    use_duration_discriminator=True, # <--- vits2 parameter
)

x = torch.LongTensor([[1, 2, 3],[4, 5, 6]]) # token ids
x_lengths = torch.LongTensor([3, 2]) # token lengths
y = torch.randn(2, 80, 100) # mel spectrograms
y_lengths = torch.Tensor([100, 80]) # mel spectrogram lengths

net_g(
    x=x,
    x_lengths=x_lengths,
    y=y,
    y_lengths=y_lengths,
)

# calculate loss and backpropagate
```

## Training Example
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12iCPWMpKdekM6F1HujYCh5H8PyzrgJN2?usp=sharing)
```sh
# LJ Speech
python train.py -c configs/vits2_ljs_nosdp.json -m ljs_base # no-sdp; (recommended)
python train.py -c configs/vits2_ljs_base.json -m ljs_base # with sdp;

# VCTK
python train_ms.py -c configs/vits2_vctk_base.json -m vctk_base
```

## TODOs, features and notes

#### Duration predictor (fig 1a) 
- [x] Added LSTM discriminator to duration predictor.
- [x] Added adversarial loss to duration predictor. ("use_duration_discriminator" flag in config file; default is "True")
- [x] Monotonic Alignment Search with Gaussian Noise added; might need expert verification (Section 2.2)
- [x] Added "use_noise_scaled_mas" flag in config file. Choose from True or False; updates noise while training based on number of steps and never goes below 0.0
- [x] Update models.py/train.py/train_ms.py
- [x] Update config files (vits2_vctk_base.json; vits2_ljs_base.json)
- [x] Update losses in train.py and train_ms.py
#### Transformer block in the normalizing flow (fig 1b)
- [x] Added transformer block to the normalizing flow. There are three types of transformer blocks: pre-convolution (my implementation), FFT (from [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc/commit/fc8336fffd40c39bdb225c1b041ab4dd15fac4e9) repo) and mono-layer.
- [x] Added "transformer_flow_type" flag in config file. Choose from "pre_conv", "fft", "mono_layer_inter_residual", "mono_layer_post_residual".
- [x] Added layers and blocks in models.py 
(ResidualCouplingTransformersLayer, 
ResidualCouplingTransformersBlock,
FFTransformerCouplingLayer,
MonoTransformerFlowLayer)
- [x] Add in config file (vits2_ljs_base.json; can be turned on using "use_transformer_flows" flag)
#### Speaker-conditioned text encoder (fig 1c)
- [x] Added speaker embedding to the text encoder in models.py (TextEncoder; backward compatible with VITS)
- [x] Add in config file (vits2_ljs_base.json; can be turned on using "use_spk_conditioned_encoder" flag)
#### Mel spectrogram posterior encoder (Section 3)
- [x] Added mel spectrogram posterior encoder in train.py 
- [x] Addded new config file (vits2_ljs_base.json; can be turned on using "use_mel_posterior_encoder" flag)
- [x] Updated 'data_utils.py' to use the "use_mel_posterior_encoder" flag for vits2
#### Training scripts
- [x] Added vits2 flags to train.py (single-speaer model)
- [x] Added vits2 flags to train_ms.py (multi-speaker model)
#### ONNX export
- [ ] Add ONNX export support. [TODO]
#### Gradio Demo
- [ ] Add Gradio demo support. [TODO]
  
## Special mentions
- [@erogol](https://github.com/erogol) for quick feedback and guidance. (Please check his awesome [CoquiTTS](https://github.com/coqui-ai/TTS) repo).
- [@lexkoro](https://github.com/lexkoro) for discussions and help with the prototype training.
- [@manmay-nakhashi](https://github.com/manmay-nakhashi) for discussions and help with the code.
- [@athenasaurav](https://github.com/athenasaurav) for offering GPU support for training.
