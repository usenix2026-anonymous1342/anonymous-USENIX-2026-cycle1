from yacs.config import CfgNode as CN


config = CN()

def init_configs(cfg):
    cfg.seed = 0 
    cfg.device = 'cuda:0'

    cfg.dataset = CN()
    cfg.dataset.passive_client_num = 1
    cfg.dataset.name = "GTSRB"
    cfg.dataset.num_classes = 43
    cfg.dataset.batch_size = 256
    
    cfg.active_wm = CN()
    cfg.active_wm.target = [i for i in range(10)]
    cfg.active_wm.path = [f'./watermark/active_wm/{i}.png' for i in range(1, 11)]


init_configs(config)
