## DPS baseline with continuous regressors

MOOD utilizes [GDSS](https://github.com/harryjo97/GDSS) as its backbone diffusion model. This work utilized the pretrained `gdss_zinc250k_v2.pth` GDSS checkpoint, which is in the folder `checkpoints/ZINC250k`.

## Prerequisites
Run the following commands to install the dependencies:
```bash
conda create -n mood python=3.8
conda activate mood
conda install -c pytorch pytorch==1.12.0 cudatoolkit=11.3
conda install -c conda-forge rdkit=2020.09 openbabel
pip install tqdm pyyaml pandas easydict networkx==2.6.3 numpy==1.20.3
```

Run the following command to preprocess the ZINC250k dataset:

```bash
python data/preprocess.py
```

It is worthnoting that all QED and SA scores reported here are in `0 ~ 1`. Specially, SA score is normalized with 

$$SA = \frac{10 - rdkit\_SA}{9}$$
 where $rdkit\_SA$ ranges from $1$ to $10$ with $1$ being easy to make and $10$ being hard to make. In conclusion, we aim to obtain bigger numbers for both QED and SA scores.

## Sampling 

For sampling, just run the following command. The parameters should be changed in `config/sample.yaml`. By default, $3000$ samples are generated, thereby making performance statistics reliable.

```sh
CUDA_VISIBLE_DEVICES=${gpu_id} python main.py --type sample --config sample
```

### Sampling from pre-trained model

Pre-trained model is GDSS, trained on ZINC250K. I have provided sampling results, located at `logs_sample/ZINC250k/pre-trained/gdss_zinc250k_v2_pretrained.log` and `logs_sample/ZINC250k/pre-trained/gdss_zinc250k_v2_pretrained.log`. Specifically,

``` 
QED: mean 0.64, std: 0.15           
SA:  mean 0.69, std: 0.13
```

<!-- (Statistics for the original ZINC250K data: QED = $0.73 \pm 0.1396$ and SA = $0.77 \pm 0.09$. ) -->

To reproduce, update the `config/sample.yaml` as

```
prop:
    ckpt: qed        # this represents the choice of differentiable reward model
    weight_x: 0    # this is the guidance strength on `X`, if it equals `0` then the sampling is from pre-trained model.
    weight_adj: 0    # this is the guidance strength on `A`. Fix it at 0.
    DPS: False       # If True, using DPS guidance; If False, using standard classifier guidance.
```

### DPS with QED predictor

for qed, update the `config/sample.yaml` as

```
prop:
    ckpt: qed
    weight_x: 0.3
    weight_adj: 0
    DPS: True
```

Here, my preliminary tests suggest `weight_x` can be `0.2 ~ 0.5`, slightly beating pre-trained model. In this range the performances are similar. For example, I have

``` 
weight_x = 0.2 and 0.3   
QED: mean 0.67, std: 0.14           
SA:  mean 0.70, std: 0.13

weight_x = 0.4 and 0.5   
QED: mean 0.67, std: 0.13           
SA:  mean 0.70, std: 0.13

weight_x = 0.6 and 0.7   
QED: mean 0.66, std: 0.13           
SA:  mean 0.71, std: 0.13
```


### DPS with SA predictor

for sa, update the `config/sample.yaml` as

```
prop:
    ckpt: sa
    weight_x: 0.1
    weight_adj: 0
    DPS: True
```

However, my preliminary tests suggest any non-zero `weight_x` is worse than pre-trained model which might be due to the inaccurate SA predictor. Experimentally, I feel the range of `0.05 ~ 0.3` have similar performances with the pre-trained model. For example, I have

``` 
weight_x = 0.05    
QED: mean 0.62, std: 0.16           
SA:  mean 0.68, std: 0.13

weight_x = 0.1    
QED: mean 0.62, std: 0.16           
SA:  mean 0.68, std: 0.12

weight_x = 0.2    
QED: mean 0.60, std: 0.16           
SA:  mean 0.68, std: 0.12

weight_x = 0.3    
QED: mean 0.59, std: 0.16           
SA:  mean 0.68, std: 0.13

weight_x = 0.4    
QED: mean 0.58, std: 0.17          
SA:  mean 0.68, std: 0.13
```


## 2. Reward Model Training

The architecture of this differentiable reward model is the discriminator network which is essentially GNNs. Please see Eq. (12) in MOOD [paper](https://arxiv.org/pdf/2206.07632).

### Using current ckpts
Firstly, there is no need to train oracles again. The model weights are already at `checkpoints/ZINC250k/qed.pth` and `checkpoints/ZINC250k/sa.pth`. You can directly run molecule generations as illustrated above.

The training logs for both oracles can be found at `logs_train/ZINC250k/prop/prop_Aug07-23:07:09_qed.log` and `logs_train/ZINC250k/prop/prop_Aug07-23:07:49_sa.log`.

Results indicate that that while the `qed` oracle seems to be ok, the `sa` oracle is essentially a less reasonable one.

```
qed 

TRAIN corr: 0.5487 | TEST corr: 0.5470
```

```
sa 

TRAIN corr: 0.2833 | TEST corr: 0.2978
```


### Reproduce training
Run the following command. The parameters should be changed in `config/prop_train.yaml`.

```bash
CUDA_VISIBLE_DEVICES=${gpu_id} python main.py --type train --config prop_train
```


Specifically `prop` indicates what property to predict. It needs to be  `qed` or `sa`, for example

```
train:
    prop: qed   # qed or sa
```