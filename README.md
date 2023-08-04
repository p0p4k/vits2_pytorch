[this is a work in progress, feel free to contribute!]

Unofficial implementation of the [VITS2 paper](https://arxiv.org/abs/2307.16430), sequel to [VITS paper](https://arxiv.org/abs/2106.06103). (thanks to the authors for their work!)

# VITS2: Improving Quality and Efficiency of Single-Stage Text-to-Speech with Adversarial Learning and Architecture Design

### Jungil Kong, Jihoon Park, Beomjeong Kim, Jeongmin Kim, Dohee Kong, Sangjin Kim 

![Alt text](image.png)

Single-stage text-to-speech models have been actively studied recently, and their results have outperformed two-stage pipeline systems. Although the previous single-stage model has made great progress, there is room for improvement in terms of its intermittent unnaturalness, computational efficiency, and strong dependence on phoneme conversion. In this work, we introduce VITS2, a single-stage text-to-speech model that efficiently synthesizes a more natural speech by improving several aspects of the previous work. We propose improved structures and training mechanisms and present that the proposed methods are effective in improving naturalness, similarity of speech characteristics in a multi-speaker model, and efficiency of training and inference. Furthermore, we demonstrate that the strong dependence on phoneme conversion in previous works can be significantly reduced with our method, which allows a fully end-toend single-stage approach.

## Credits
We will build this repo based on the [VITS repo](https://github.com/jaywalnut310/vits). Currently I am adding vits2 changes in the 'notebooks' folder. The goal is to make this model easier to transfer learning from VITS pretrained model!

## Jupyter Notebook for initial experiments
- [x] check the 'notebooks' folder

## Features
#### Duration predictor (fig 1a)
- [x] Added LSTM discriminator to duration predictor in notebook.
- [ ] Added adversarial loss to duration predictor
- [x] Monotonic Alignment Search with Gaussian Noise added in 'notebooks' folder; need expert verification
- [ ] Update models.py/train.py/losses.py
#### Transformer block in the normalizing flow (fig 1b)
- [x] Added transformer block to the normalizing flow in notebook.
- [x] Added layers and blocks in models.py (ResidualCouplingTransformersLayer, ResidualCouplingTransformersBlock)
#### Speaker-conditioned text encoder (fig 1c)
- [x] Added speaker embedding to the text encoder in notebook.
- [x] Added speaker embedding to the text encoder in models.py (TextEncoder; backward compatible with VITS)

### Special mentions
@lexkoro for discussions and help with the code.