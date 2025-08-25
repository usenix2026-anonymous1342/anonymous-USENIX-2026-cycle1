from yacs.config import CfgNode as CN


config = CN()

def init_configs(cfg):
    cfg.seed = 0 
    cfg.num_rounds = 200
    cfg.device = 'cuda:0'
    cfg.pretrain = False

    cfg.dataset = CN()
    cfg.dataset.passive_client_num = 1
    cfg.dataset.name = "BrainTumor"
    cfg.dataset.num_classes = 4
    cfg.dataset.batch_size = 64

    cfg.optimizer = CN()
    cfg.optimizer.lr = 1e-3
    cfg.optimizer.weight_decay = 5e-5

    cfg.global_wm = CN()
    cfg.global_wm.target = 0
    cfg.global_wm.ratio = 0

    cfg.active_wm = CN()
    cfg.active_wm.target = [i for i in range(4)]
    cfg.active_wm.path = [f'./watermark/active_wm/{i}.png' for i in range(1, 5)]   # 'None' for No Active_wm

    cfg.passive_wm = CN()
    cfg.passive_wm.target_path = f'./watermark/fidelity/{cfg.dataset.name}'
    cfg.passive_wm.trigger_path = './watermark/watermark.png'
    cfg.passive_wm.num_rounds = 300
    cfg.passive_wm.lr = 1e-4
    cfg.passive_wm.weight_decay = 5e-5
    cfg.passive_wm.momentum = 0.9
    cfg.passive_wm.batch_size = 64
    cfg.passive_wm.knn_k = 8


init_configs(config)
