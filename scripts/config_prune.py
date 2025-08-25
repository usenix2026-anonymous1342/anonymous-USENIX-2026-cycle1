from yacs.config import CfgNode as CN


config = CN()

def init_configs(cfg):
    cfg.seed = 0 
    cfg.num_rounds = 100
    cfg.device = 'cuda:0'

    cfg.dataset = CN()
    cfg.dataset.passive_client_num = 1
    cfg.dataset.name = "CIFAR10"
    cfg.dataset.num_classes = 10
    cfg.dataset.batch_size = 256

    cfg.global_wm = CN()
    cfg.global_wm.target = 0
    cfg.global_wm.ratio = 0.005    # '0' for No Global_wm

    cfg.active_wm = CN()
    cfg.active_wm.target = [i for i in range(10)]
    cfg.active_wm.path = [f'./watermark/active_wm/{i}.png' for i in range(1, 11)]   # 'None' for No Active_wm

    cfg.passive_wm = CN()
    cfg.passive_wm.target_path = f'./watermark/passive_wm/{cfg.dataset.name}'
    cfg.passive_wm.trigger_path = './watermark/watermark.png'
    cfg.passive_wm.batch_size = 64
    cfg.passive_wm.knn_k = 20


init_configs(config)
