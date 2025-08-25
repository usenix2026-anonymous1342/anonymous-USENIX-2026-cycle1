from pickle import FALSE
from yacs.config import CfgNode as CN


config = CN()

def init_configs(cfg):
    cfg.seed = 0 
    cfg.num_rounds = 200
    cfg.device = 'cuda:0'
    cfg.pretrain = True

    cfg.dataset = CN()
    cfg.dataset.passive_client_num = 1
    cfg.dataset.name = "CIFAR10"
    cfg.dataset.num_classes = 10
    cfg.dataset.batch_size = 256

    cfg.optimizer = CN()
    cfg.optimizer.lr = 1e-3
    cfg.optimizer.weight_decay = 5e-5
    # cfg.optimizer.momentum = 0.9

    cfg.global_wm = CN()
    cfg.global_wm.target = 0
    cfg.global_wm.ratio = 0    # '0' for No Global_wm

    cfg.active_wm = CN()
    cfg.active_wm.target = [0]
    cfg.active_wm.path = None   # 'None' for No Active_wm
    # cfg.active_wm.target = [i for i in range(10)]
    # cfg.active_wm.path = [f'./watermark/active_wm/{i}.png' for i in range(1, 11)]   # 'None' for No Active_wm

    cfg.passive_wm = CN()
    cfg.passive_wm.target_path = f'./watermark/passive_wm/{cfg.dataset.name}'
    cfg.passive_wm.trigger_path = './watermark/passive_wm/watermark.png'
    cfg.passive_wm.num_rounds = 100
    cfg.passive_wm.lr = 1e-4
    cfg.passive_wm.weight_decay = 5e-5
    cfg.passive_wm.momentum = 0.9
    cfg.passive_wm.batch_size = 128
    cfg.passive_wm.knn_k = 500


init_configs(config)
