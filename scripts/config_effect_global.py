from yacs.config import CfgNode as CN


config = CN()

def init_configs(cfg):
    cfg.seed = 0 
    cfg.device = 'cuda:0'

    cfg.dataset = CN()
    cfg.dataset.passive_client_num = 4
    cfg.dataset.name = "CIFAR10"
    cfg.dataset.num_classes = 10
    cfg.dataset.batch_size = 256

    cfg.global_wm = CN()
    cfg.global_wm.target = 0


init_configs(config)
