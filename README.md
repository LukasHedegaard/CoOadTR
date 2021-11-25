# OadTR ablations
This repository contains an ablation study for the OadTR network from "OadTR: Online Action Detection with Transformers" (ICCV2021) [["Paper"]](https://arxiv.org/pdf/2106.11149.pdf).

It is a fork of the [official source](https://github.com/wangxiang1230/OadTR), with the purporse of ablating different features to make the model compatbile with [Continual Transformers](https://github.com/LukasHedegaard/continual-transformers).

Each conducted experiment has its own branch. An overview of the ablated features and associated results is found in the table below for the TSN-Anet features:

| Encoder-layers  | Decoder  | Class-token | Token-shift* | mAP (%) | branch  | command |
| -------         | -------- | --------    | --------     | ------- | ------- | ------- |
| 3               | ✔︎        | ✔︎           | -            | 57.8    | [main](https://github.com/LukasHedegaard/OadTR/tree/main)    | `python main.py --num_layers 3 --decoder_layers 5 --enc_layers 64`  |
| 3               | -        | ✔︎           | -            | 56.8    | [no-decoder](https://github.com/LukasHedegaard/OadTR/tree/no-decoder)    | `python main.py --num_layers 3 --enc_layers 64`  |
| 1               | -        | ✔︎           | -            | 55.6    | [no-decoder](https://github.com/LukasHedegaard/OadTR/tree/no-decoder)    | `python main.py --num_layers 1 --enc_layers 64`  |
| 1               | -        | -           | -            | 55.5    | [no-decoder-no-cls-token](https://github.com/LukasHedegaard/OadTR/tree/no-decoder-no-cls-token)    | `python main.py --num_layers 1 --enc_layers 64`  |
| 1               | -        | -           | ✔︎            | 55.7    | [no-decoder-no-cls-token-shifting-tokens](https://github.com/LukasHedegaard/OadTR/tree/no-decoder-no-cls-token-shifting-tokens)    | `python main.py --num_layers 1 --enc_layers 64`  |
                

# Set-up

## Package Dependencies

* pytorch==1.6.0 
* json
* numpy
* tensorboard-logger
* torchvision==0.7.0

## Pretrained features

* Unzip the anno file "./data/anno_thumos.zip"
* Download the features:
  * [THUMOS14-Anet feature](https://zenodo.org/record/5035147#.YNhWG7vitPY) 
  * [THUMOS14-Kinetics feature](https://zenodo.org/record/5140603#.YQDk8britPY)
  * [HDD](https://usa.honda-ri.com/hdd) and [TVSeries](https://homes.esat.kuleuven.be/psi-archive/rdegeest/TVSeries.html) are available by contacting the authors of the datasets and signing agreements due to the copyrights. You can use this [Repo](https://github.com/yjxiong/anet2016-cuhk) to extract features.

# Training
```
python main.py --num_layers 3 --decoder_layers 5 --enc_layers 64 --output_dir models/en_3_decoder_5_lr_drop_1
```
# Validation
```
python main.py --num_layers 3 --decoder_layers 5 --enc_layers 64 --output_dir models/en_3_decoder_5_lr_drop_1 --eval --resume models/en_3_decoder_5_lr_drop_1/checkpoint000{}.pth
```


