# DamageNet

## Introduction
For efficient post disaster road network extraction, we design a temporal attention model trained on labelled satellite images. Due to lacking of labelled post disaster images, we adpot image augmentation tools to generate fake post disaster images. With fake post disaster images, our model distinguishes damaged parts of roads better than other road network extraction models.

## Dataset (Pre Disaster/Post Disaster Image Pair)
- Training set
Road Extraction Challenge of [DeepGlobe 2018](https://arxiv.org/pdf/1805.06561.pdf)/Fake Post Disaster Set (Noise + Road Extraction Challenge of [DeepGlobe 2018](https://arxiv.org/pdf/1805.06561.pdf))
- Testing set
Pre disaster/Post disaster images are downloaded from [Open Data Program](https://www.maxar.com/open-data) available by Maxar

## Acknowledgement
Part of our code is borrowed from the following repositories.
- [Automold](https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library)
- [Implementation Attention UNet](https://github.com/LeeJunHyun/Image_Segmentation)
- [Attention-Augmented-Conv2d](https://github.com/leaderj1001/Attention-Augmented-Conv2d)
