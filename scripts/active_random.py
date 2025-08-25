import os
import sys
import random

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils


class VFL_framwork(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device

        self.setup()

    def setup(self):
        seed = 0
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.bottom_model_list = [utils.get_model(self.cfg)[0].to(self.device) for _ in range(self.cfg.dataset.passive_client_num)]
        self.top_model = utils.get_model(self.cfg)[1].to(self.device)

    def benign_load(self):
        save_path = os.path.join('./model/clean', f"{self.cfg.dataset.name}")

        top_model_path = os.path.join(save_path, f'{cfg.dataset.passive_client_num}_active_model.pth')
        state_dict = torch.load(f'{top_model_path}')
        self.top_model.load_state_dict(state_dict)
        print("Active Model Loaded Successfully!")

        for idx, model in enumerate(self.bottom_model_list):
            passive_model_path = os.path.join(save_path, f"{cfg.dataset.passive_client_num}-{idx+1}_passive_model.pth")
            state_dict = torch.load(f'{passive_model_path}')
            model.load_state_dict(state_dict)
        print("Benign Model Loaded Successfully!")

    def watermark_load(self):
        save_path = os.path.join('./model/final', f"{self.cfg.dataset.name}")
        
        top_model_path = os.path.join(save_path, f'{cfg.dataset.passive_client_num}_active_model.pth')
        state_dict = torch.load(f'{top_model_path}')
        self.top_model.load_state_dict(state_dict)
        print("Active Model Loaded Successfully!")

        for idx, model in enumerate(self.bottom_model_list):
            passive_model_path = os.path.join(save_path, f"{cfg.dataset.passive_client_num}-{idx+1}_passive_model.pth")
            state_dict = torch.load(f'{passive_model_path}')
            model.load_state_dict(state_dict)
        print("Watermarked Model Loaded Successfully!")


if __name__ == '__main__':
    import csv
    import numpy as np
    import matplotlib.pyplot as plt

    from scripts.config_random_active import config as cfg


    vfl_framework = VFL_framwork(cfg)
    watermark = utils.get_active_wm(cfg)


    vfl_framework.benign_load()
    model = vfl_framework.top_model.to(cfg.device)
    model.eval()
    output = model([watermark])
    print(output)

    ans = 0
    tensor = torch.randint(0, 2, (2000, 512))
    tensor_float = tensor.to(torch.float32).to(cfg.device)
    output = model([tensor_float])
    predict = torch.argmax(output, dim=-1)
    counts = torch.bincount(predict, minlength=cfg.dataset.num_classes)

    result_path = f"./results/{cfg.dataset.name}/active_random_benign.csv"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Class", "Counts"])
        for i in range(cfg.dataset.num_classes):
            writer.writerow([
                i,
                counts[i],
            ])

    for number, count in enumerate(counts):
        print(f"Number {number} appears {count} times")
    
    width = 0.35
    x = np.arange(len(counts.tolist()))
    plt.figure(figsize=(8, 6))
    plt.bar(x - width / 2, counts.tolist(), width, label="Benign Model", alpha=0.7)



    vfl_framework.watermark_load()
    model = vfl_framework.top_model.to(cfg.device)
    model.eval()
    output = model([watermark])
    print(output)

    ans = 0
    tensor = torch.randint(0, 2, (2000, 512))
    tensor_float = tensor.to(torch.float32).to(cfg.device)
    output = model([tensor_float])
    predict = torch.argmax(output, dim=-1)
    counts = torch.bincount(predict, minlength=cfg.dataset.num_classes)

    result_path = f"./results/{cfg.dataset.name}/active_random_watermark.csv"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Class", "Counts"])
        for i in range(cfg.dataset.num_classes):
            writer.writerow([
                i,
                counts[i],
            ])

    for number, count in enumerate(counts):
        print(f"Number {number} appears {count} times")

    plt.bar(x + width / 2, counts.tolist(), width, label="Watermarked Model", alpha=0.7)
    plt.title(f'Class Distribution Comparison (Active Random)')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(x, range(len(counts.tolist())))
    plt.legend()
    plt.savefig(f'./results/{cfg.dataset.name}/active_random.png', dpi=300)
