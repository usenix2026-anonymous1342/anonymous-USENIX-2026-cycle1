from pickle import FALSE
from yacs.config import CfgNode as CN


config = CN()

def init_configs(cfg):
    cfg.seed = 0 
    cfg.device = 'cuda:0'

    cfg.dataset = CN()
    cfg.dataset.passive_client_num = 1
    cfg.dataset.name = "CIFAR10"
    cfg.dataset.num_classes = 10

    cfg.passive_wm.knn_k = 500


init_configs(config)
