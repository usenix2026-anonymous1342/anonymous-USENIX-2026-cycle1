## Overview

This repository contains scripts and resources to embed three types of watermark with two stages training for vertical federated learning(split leaning), that are global watermark, active watermark and passive watermark.



## Requirements

- Python 3.10.\*
- PyTorch 1.13.\*



## File

- `main.py`: workflow of vfl and watermark
- `config.py`: vfl and watermark parameters
- `watermark\`: ood-based active watermark sample & optimization-based passive watermark sample, which can all be got during stage1 traing in vfl.
- `scripts\`: all experiment configs & scripts, use directly
- `data\`: **four needed datasets(CIFAR10 ImageNet GTSRB BrainTumor) should be placed here**



## Usage

**Set the vfl and watermark parameters**：

set the default config in `config.py`.

```python
def init_configs(cfg):
    cfg.seed = 0 
    cfg.num_rounds = 200
    cfg.device = 'cuda:0'
    cfg.pretrain = True					# if pretrain=True then go stage two embeding passive watermark directly

    cfg.dataset = CN()
    cfg.dataset.passive_client_num = 1	# passive_client_num
    cfg.dataset.name = "CIFAR10"		# vfl dataset
    cfg.dataset.num_classes = 10
    cfg.dataset.batch_size = 256

    cfg.optimizer = CN()
    cfg.optimizer.lr = 1e-3
    cfg.optimizer.weight_decay = 5e-5
    # cfg.optimizer.momentum = 0.9

    cfg.global_wm = CN()
    cfg.global_wm.target = 0
    cfg.global_wm.ratio = 0.05    # '0' for No Global_wm

    cfg.active_wm = CN()
    cfg.active_wm.target = [i for i in range(10)]
    cfg.active_wm.path = [f'./watermark/active_wm/{i}.png' for i in range(1, 11)]   # 'None' for No Active_wm

    cfg.passive_wm = CN()
    cfg.passive_wm.target_path = f'./watermark/passive_wm/{cfg.dataset.name}'	# path to images belonged to target class
    cfg.passive_wm.trigger_path = './watermark/passive_wm/watermark.png'		# passive watermark's trigger path
    cfg.passive_wm.num_rounds = 100
    cfg.passive_wm.lr = 1e-4
    cfg.passive_wm.weight_decay = 5e-5
    cfg.passive_wm.momentum = 0.9
    cfg.passive_wm.batch_size = 128
    cfg.passive_wm.knn_k = 500
```

**Train VFL Model & Embed Watermark & Verificate Watermark**

```bash
python main.py
```



## Notes

- More details about config can get from e-mail.