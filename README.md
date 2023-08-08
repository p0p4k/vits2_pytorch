[this is a work in progress, feel free to contribute! Model will be ready if this line is removed]

# VITS2: Improving Quality and Efficiency of Single-Stage Text-to-Speech with Adversarial Learning and Architecture Design
### Jungil Kong, Jihoon Park, Beomjeong Kim, Jeongmin Kim, Dohee Kong, Sangjin Kim 
Unofficial implementation of the [VITS2 paper](https://arxiv.org/abs/2307.16430), sequel to [VITS paper](https://arxiv.org/abs/2106.06103). (thanks to the authors for their work!)

![Alt text](image.png)

Single-stage text-to-speech models have been actively studied recently, and their results have outperformed two-stage pipeline systems. Although the previous single-stage model has made great progress, there is room for improvement in terms of its intermittent unnaturalness, computational efficiency, and strong dependence on phoneme conversion. In this work, we introduce VITS2, a single-stage text-to-speech model that efficiently synthesizes a more natural speech by improving several aspects of the previous work. We propose improved structures and training mechanisms and present that the proposed methods are effective in improving naturalness, similarity of speech characteristics in a multi-speaker model, and efficiency of training and inference. Furthermore, we demonstrate that the strong dependence on phoneme conversion in previous works can be significantly reduced with our method, which allows a fully end-toend single-stage approach.

## Credits
We will build this repo based on the [VITS repo](https://github.com/jaywalnut310/vits). Currently I am adding vits2 changes in the 'notebooks' folder. The goal is to make this model easier to transfer learning from VITS pretrained model!

## Jupyter Notebook for initial experiments
- [x] check the 'notebooks' folder
- [x] check the 'notebooks/quick_previews' folder for quick glance

## How to run (dry-run)

- build monotonic alignment 
```sh
# Cython-version Monotonoic Alignment Search
cd monotonic_align
python setup.py build_ext --inplace
```
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
    transformer_flow_type="fft", # <--- vits2 parameter (choose from "pre_conv" to "fft")
    use_spk_conditioned_encoder=True, # <--- vits2 parameter
    use_noise_scaled_mas=True, # <--- vits2 parameter
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

## Features
- (08/08/2023) update - Added "use_noise_scaled_mas" flag in config file. Added sanity checks in notebooks. Everything except adverserial duration predictor is ready to train.
- (08/072023) update 2 - transformer_flow_type "fft" and "pre_conv" added. [@lexkoro](https://github.com/lexkoro) suggested "fft" transformer flow is better than "pre_conv" transformer flow in his intial experiments.
- (08/07/2023 update 1 - vits2_vctk_base.json and vits2_ljs_base.json are ready to train; multi-speaker and single-speaker models respectively)
- (08/06/2023 update - dry run is ready; duration predictor will complete within next week)
- (08/05/2023 update - everything except the duration predictor is ready to train and we can expect some improvement from VITS1)
- (08/04/2023 update - initial codebaase is ready; paper is being read)
#### Duration predictor (fig 1a)
- [x] Added LSTM discriminator to duration predictor in notebook.
- [ ] Added adversarial loss to duration predictor
- [x] Monotonic Alignment Search with Gaussian Noise added in 'notebooks' folder; need expert verification (Section 2.2)
- [x] Added "use_noise_scaled_mas" flag in config file. Choose from True or False; updates noise while training based on number of steps and never goes below 0.0
- [x] Update models.py/train.py/train_ms.py
- [x] Update config files (vits2_vctk_base.json; vits2_ljs_base.json)
- [ ] Update losses.py
#### Transformer block in the normalizing flow (fig 1b)
- [x] Added transformer block to the normalizing flow in notebook. There are two types of transformer blocks: pre-convolution (my implementation) and FFT (from [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc/commit/fc8336fffd40c39bdb225c1b041ab4dd15fac4e9) repo)
- [x] Added "transformer_flow_type" flag in config file. Choose from "pre_conv" to "fft"
- [x] Added layers and blocks in models.py 
(ResidualCouplingTransformersLayer, 
ResidualCouplingTransformersBlock,
TransformerCouplingLayer)
- [x] Add in config file (vits2_ljs_base.json; can be turned on using "use_transformer_flows" flag)
#### Speaker-conditioned text encoder (fig 1c)
- [x] Added speaker embedding to the text encoder in notebook.
- [x] Added speaker embedding to the text encoder in models.py (TextEncoder; backward compatible with VITS)
- [x] Add in config file (vits2_ljs_base.json; can be turned on using "use_spk_conditioned_encoder" flag)
#### Mel spectrogram posterior encoder (Section 3)
- [x] Added mel spectrogram posterior encoder in notebook.
- [x] Added mel spectrogram posterior encoder in train.py 
- [x] Addded new config file (vits2_ljs_base.json; can be turned on using "use_mel_posterior_encoder" flag)
#### Training scripts
- [x] Added vits flags to train.py (single-speaer model)
- [x] Added vits flags to train_ms.py (multi-speaker model)

## Special mentions
- [@erogol](https://github.com/erogol) for quick feedback and guidance. (Please check his awesome [CoquiTTS](https://github.com/coqui-ai/TTS) repo).
- [@lexkoro](https://github.com/lexkoro) for discussions and help with the prototype training.
- [@manmay-nakhashi](https://github.com/manmay-nakhashi) for discussions and help with the code.
