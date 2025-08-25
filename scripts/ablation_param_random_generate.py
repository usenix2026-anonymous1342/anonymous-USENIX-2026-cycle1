import os
import csv
import sys
import copy
import heapq
import random
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils



def save_passive_target(cfg, top_heap_list, select_round, select_num):
    for idx, top_heap in enumerate(top_heap_list):
        top_images = [(item[1], item[2]) for item in sorted(top_heap, key=lambda x: -x[0])]

        save_path = f"./watermark/ablation_param_{select_round}_{select_num}/{cfg.dataset.name}/party_{idx}_of_{cfg.dataset.passive_client_num}/"
        os.makedirs(save_path, exist_ok=True)
        for selected, (img, label) in enumerate(top_images):
            if selected >= select_num:
                break

            img = (img * 255).byte().permute(1, 2, 0).numpy()
            img_pil = Image.fromarray(img)
            img_pil.save(os.path.join(save_path, f"image_{len(os.listdir(save_path))}_label_{label}.png"))


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

        # Dataset
        split_train_dataset, split_test_dataset, split_wm_test_dataset = utils.get_dataset(self.cfg)
        train_dataset = copy.deepcopy(split_train_dataset)


        for idx in range(self.cfg.dataset.passive_client_num): 
            random_numbers = random.sample(range(len(split_train_dataset)), 20)
            save_path = f"./watermark/ablation_param_random/{self.cfg.dataset.name}/party_{idx}_of_{self.cfg.dataset.passive_client_num}/"
            os.makedirs(save_path, exist_ok=True)

            for index in random_numbers:
                init_x, x, y = train_dataset[index]
                img = (init_x[idx] * 255).byte().permute(1, 2, 0).numpy()
                img_pil = Image.fromarray(img)
                img_pil.save(os.path.join(save_path, f"image_{len(os.listdir(save_path))}_label_{y}.png"))

            print(f"{idx} finished")

from config_ablation_random import config as cfg

vfl_framework = VFL_framwork(cfg)

