from yacs.config import CfgNode as CN


config = CN()

def init_configs(cfg):
    cfg.seed = 0 
    cfg.num_rounds = 200
    cfg.device = 'cuda:0'

    cfg.dataset = CN()
    cfg.dataset.passive_client_num = 4
    cfg.dataset.name = "BrainTumor"
    cfg.dataset.num_classes = 4
    cfg.dataset.batch_size = 64

    cfg.optimizer = CN()
    cfg.optimizer.lr = 1e-3
    cfg.optimizer.weight_decay = 5e-5

    cfg.global_wm = CN()
    cfg.global_wm.target = 0
    cfg.global_wm.ratio = 0.005


init_configs(config)
