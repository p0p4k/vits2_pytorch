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

## Features
(08/05/2023 update - everything except the duration predictor is ready to train and we can expect some improvement from VITS1)
(08/04/2023 update - initial codebaase is ready; paper is being read)
#### Duration predictor (fig 1a)
- [x] Added LSTM discriminator to duration predictor in notebook.
- [ ] Added adversarial loss to duration predictor
- [x] Monotonic Alignment Search with Gaussian Noise added in 'notebooks' folder; need expert verification (Section 2.2)
- [ ] Update models.py/train.py/losses.py
#### Transformer block in the normalizing flow (fig 1b)
- [x] Added transformer block to the normalizing flow in notebook.
- [x] Added layers and blocks in models.py (ResidualCouplingTransformersLayer, ResidualCouplingTransformersBlock)
#### Speaker-conditioned text encoder (fig 1c)
- [x] Added speaker embedding to the text encoder in notebook.
- [x] Added speaker embedding to the text encoder in models.py (TextEncoder; backward compatible with VITS)
#### Mel spectrogram posterior encoder (Section 3)
- [x] Added mel spectrogram posterior encoder in notebook.
- [x] Added mel spectrogram posterior encoder in train.py 
- [x] Addded new config file (vits2_ljs_base.json; can be turned on using "use_mel_posterior_encoder" flag)

## Special mentions
[@erogol](https://github.com/erogol) for quick feedback and guidance. (Please check his awesome [CoquiTTS](https://github.com/coqui-ai/TTS) repo).
[@lexkoro](https://github.com/lexkoro) for discussions and help with the prototype training.
[@manmay-nakhashi](https://github.com/manmay-nakhashi) for discussions and help with the code.
