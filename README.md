# Continual Online Action Detection Transformer

This repository contains the Online Action Recognition Experiments from our work on [Continual Transformers](https://github.com/LukasHedegaard/continual-transformers).


This repository is a fork of the [official source code](https://github.com/wangxiang1230/OadTR) for the "OadTR: Online Action Detection with Transformers" (ICCV2021) [["Paper"]](https://arxiv.org/pdf/2106.11149.pdf), with model variations specified in different branches.

# Set-up

## Package Dependencies
Install the dependencies from the original OadTR project
```bash
pip install pytorch torchvision numpy json tensorboard-logger
```

Install Continual Transformer blocks
```bash
pip install --upgrade git+https://github.com/LukasHedegaard/continual-transformers.git
```


## Pretrained features

* Unzip the anno file "./data/anno_thumos.zip"
* Download the features:
  * [THUMOS14-Anet feature](https://zenodo.org/record/5035147#.YNhWG7vitPY) 
  * [THUMOS14-Kinetics feature](https://zenodo.org/record/5140603#.YQDk8britPY)
  * [TVSeries](https://homes.esat.kuleuven.be/psi-archive/rdegeest/TVSeries.html) is available by contacting the authors of the datasets and signing agreements due to the copyrights. Following [this guide](https://github.com/LukasHedegaard/mmaction2/tree/tvseries-feature-extraction/tools/data/tvseries), we extracted features using TSN ResNet-50 RGB and Flow models pretrained on [ActivityNet](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tsn/README.md#activitynet-v13) and [Kinetics](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tsn/README.md#kinetics-400).

When you have downloaded and placed the THUMOS featues under `~/data`, you can select the features by appending the following to your python command:
- ActivityNet (default): 
  - `--feature Anet2016_feature_v2`
- Kinetics:
  - `--feature V3`


# Experiments
## CoOadTR
From the main branch the CoOadTR model can be run with the following: command 
```bash
python main.py --num_layers 1 --enc_layers 64 --cpe_factor 1
```
Here, `num_layers` denotes the number of transformer blocks (1 or 2), `enc_layers` is the sequence length, and `cpe_factor` is a multiplier for the number of unique circular positional embeddings (1>=x>=2).


## OadTR ablations

Each conducted experiment has its own branch. An overview of the ablated features and associated results is found in the table below for the TSN-Anet features:

| Encoder-layers  | Decoder  | Class-token | Circular encoding  | mAP (%) | branch  | command |
| -------         | -------- | --------    | --------     | ------- | ------- | ------- |
| 3               | ✔︎        | ✔︎           | -            | 57.8    | [original](https://github.com/LukasHedegaard/OadTR/tree/original) (baseline)    | `python main.py --num_layers 3 --decoder_layers 5 --enc_layers 64`  |
| 3               | -        | ✔︎           | -            | 56.8    | [no-decoder](https://github.com/LukasHedegaard/OadTR/tree/no-decoder)    | `python main.py --num_layers 3 --enc_layers 64`  |
| 2               | -        | ✔︎           | -            | 55.6    | [no-decoder](https://github.com/LukasHedegaard/OadTR/tree/no-decoder)    | `python main.py --num_layers 2 --enc_layers 64`  |
| 2               | -        | -           | -            | 55.5    | [no-decoder-no-cls-token](https://github.com/LukasHedegaard/OadTR/tree/no-decoder-no-cls-token)    | `python main.py --num_layers 2 --enc_layers 64`  |
| 1               | -        | -           | ✔︎ (len n)        | 55.7    | [no-decoder-no-cls-token-shifting-tokens](https://github.com/LukasHedegaard/OadTR/tree/no-decoder-no-cls-token-shifting-tokens)    | `python main.py --num_layers 1 --enc_layers 64`  |
| 1               | -        | -           | ✔︎ (len 2n)       | 55.8    | [no-decoder-no-cls-token-shifting-tokens-2x](https://github.com/LukasHedegaard/OadTR/tree/no-decoder-no-cls-token-shifting-tokens-2x)    | `python main.py --num_layers 1 --enc_layers 64`  |
                

## THUMOS
| Model         | branch                | command               |
| -------       | -------               | -------               |
| OadTR         | [original](https://github.com/LukasHedegaard/OadTR/tree/original)   | `python main.py --num_layers 3 --decoder_layers 5 --enc_layers 64  --feature <FEATURE>`  |
| OadTR-b2      | [no-decoder-no-cls-token](https://github.com/LukasHedegaard/OadTR/tree/no-decoder-no-cls-token)   | `python main.py --num_layers 2 --enc_layers 64 --feature <FEATURE>`  |
| OadTR-b2      | [no-decoder-no-cls-token](https://github.com/LukasHedegaard/OadTR/tree/no-decoder-no-cls-token)   | `python main.py --num_layers 1 --enc_layers 64 --feature <FEATURE>`  |
| CoOadTR-b2    | [main](https://github.com/LukasHedegaard/CoOadTR/tree/main)   | `python main.py --num_layers 2 --enc_layers 64 --feature <FEATURE>`  |
| CoOadTR-b1    | [main](https://github.com/LukasHedegaard/CoOadTR/tree/main)   | `python main.py --num_layers 1 --enc_layers 64 --feature <FEATURE>`  |

Where `<FEATURE>` is either `"anet"` or `"kin"` for ActivityNet and Kinetics pretrained features, respectively.
`--dim_feature` should be `3072` for `"anet"` , and `4096` for `"kin"`.


## TVSeries
| Model         | branch                | command               |
| -------       | -------               | -------               |
| OadTR         | [original-tvseries](https://github.com/LukasHedegaard/CoOadTR/tree/original-tvseries)   | `python main.py --num_layers 3 --decoder_layers 5 --enc_layers 64   --feature <FEATURE>`  |
| OadTR-b2      | [no-decoder-no-cls-token-tvseries](https://github.com/LukasHedegaard/CoOadTR/tree/no-decoder-no-cls-token-tvseries)   | `python main.py --num_layers 2 --enc_layers 64 --feature <FEATURE>`  |
| OadTR-b2      | [no-decoder-no-cls-token-tvseries](https://github.com/LukasHedegaard/CoOadTR/tree/no-decoder-no-cls-token-tvseries)   | `python main.py --num_layers 1 --enc_layers 64 --feature <FEATURE>`  |
| CoOadTR-b2    | [main](https://github.com/LukasHedegaard/CoOadTR/tree/main)   | `python main.py --dataset tvseries --num_layers 2 --enc_layers 64 --feature <FEATURE>`  |
| CoOadTR-b1    | [main](https://github.com/LukasHedegaard/CoOadTR/tree/main)   | `python main.py --dataset tvseries --num_layers 1 --enc_layers 64 --feature <FEATURE>`  |

Where `<FEATURE>` is the name of your `.pickle` file of extracted features (either A.Net or Kin. features), placed in the `~/data` folder.
