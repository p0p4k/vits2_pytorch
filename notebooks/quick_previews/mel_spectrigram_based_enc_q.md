![image.png](mel_spectrigram_based_enc_q_files/image.png)


```python
import os
os.chdir(r'../../vits2_pytorch')
```


```python
import torch

from models import SynthesizerTrn
```

    c:\ProgramData\Anaconda3\envs\pytorch\lib\site-packages\tqdm\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    


```python
# authors suggest using 80 channel mel-spectrogram as input to the model instead of 513 channel linear spectrogram
filter_length = 1024
filter_length // 2 + 1 ##vits1 513 channel linear spectrogram (check the old code in train.py)
```




    513




```python
import torch
from models import SynthesizerTrn
net_g = SynthesizerTrn(
    n_vocab=256,
    spec_channels=80,
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
    use_transformer_flows=True,
    use_spk_conditioned_encoder=True,
)
```


```python
x = torch.LongTensor([[1, 2, 3],[4, 5, 6]])
x_lengths = torch.LongTensor([3, 2])
y = torch.randn(2, 80, 100)
y_lengths = torch.Tensor([100, 80])
```


```python
net_g(
    x=x,
    x_lengths=x_lengths,
    y=y,
    y_lengths=y_lengths,
)
```
