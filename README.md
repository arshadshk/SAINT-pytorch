# SAINT-pytorch
A Simple pyTorch implementation of "Towards an Appropriate Query, Key, and Value Computation for Knowledge Tracing" based on https://arxiv.org/abs/2002.07033.  



**SAINT**: Separated Self-AttentIve Neural Knowledge Tracing. SAINT has an encoder-decoder structure where exercise and response embedding sequence separately enter the encoder and the decoder respectively, which allows to stack attention layers multiple times.  

## SAINT model architecture  
<img src="https://github.com/arshadshk/SAINT-pytorch/blob/main/arch_from_paper.JPG">

## Usage 
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from saint import saint, random_data

seq_len = 100
total_ex = 1200
total_cat = 234
total_in = 2

in_ex, in_cat, in_de = random_data(64, 
                                seq_len , 
                                total_ex, 
                                total_cat, 
                                total_in)


model = saint(dim_model=128,
            num_en=6,
            num_de=6,
            heads_en=8,
            heads_de=8,
            total_ex=total_ex,
            total_cat=total_cat,
            total_in=total_in )

outs = model(in_ex, in_cat, in_de)

print(outs.shape)
# torch.Size([64, 100, 1])
```
  
## Parameters
- `dim_model`: int.  
Dimension of model ( embeddings, attention, linear layers).
- `num_en`: int.  
Number of encoder layers.
- `num_de`: int.  
Number of decoder layers.  
- `heads_en`: int.  
Number of heads in multi-head attention block in each layer of encoder.
- `heads_de`: int.  
Number of heads in multi-head attention block in each layer of decoder.
- `total_ex`: int.  
Total number of unique excercise.
- `total_cat`: int.  
Total number of unique concept categories.
- `total_in`: int.  
Total number of unique interactions.

## todo
- change positional embedding to sine. 

## Citations

```bibtex
@article{choi2020towards,
  title={Towards an Appropriate Query, Key, and Value Computation for Knowledge Tracing},
  author={Choi, Youngduck and Lee, Youngnam and Cho, Junghyun and Baek, Jineon and Kim, Byungsoo and Cha, Yeongmin and Shin, Dongmin and Bae, Chan and Heo, Jaewe},
  journal={arXiv preprint arXiv:2002.07033},
  year={2020}
}
```

```bibtex
@misc{vaswani2017attention,
    title   = {Attention Is All You Need},
    author  = {Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
    year    = {2017},
    eprint  = {1706.03762},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL}
}
```