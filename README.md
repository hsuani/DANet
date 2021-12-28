# DamageNet
This repo is to reproduce a temporal-attention model to extract post-disaster accessible road network.

## Introduction
For efficient post disaster road network extraction, we design a temporal attention model trained on labelled satellite images. Due to lacking of labelled post disaster images, we adpot image augmentation tools to generate fake post disaster images. With fake post disaster images, our model distinguishes damaged parts of roads better than other road network extraction models.

## Dataset
- Training set<br>
  Pre disaster images: road extraction challenge of [DeepGlobe 2018](https://arxiv.org/pdf/1805.06561.pdf)<br>
  Post disaster images: fake post disaster set (noise + road extraction challenge of [DeepGlobe 2018](https://arxiv.org/pdf/1805.06561.pdf))<br>
- Testing set<br>
  Both pre and post disaster images are downloaded from [Open Data Program](https://www.maxar.com/open-data) available by Maxar.

## Quick Start
Run the command<br>
```
  python3 main_config.py --config_path config_damagenet.json
```

## Acknowledgement
Parts of our code are borrowed from the following repositories.
- [Automold](https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library)
- [Implementation Attention UNet](https://github.com/LeeJunHyun/Image_Segmentation)
- [Attention-Augmented-Conv2d](https://github.com/leaderj1001/Attention-Augmented-Conv2d)
