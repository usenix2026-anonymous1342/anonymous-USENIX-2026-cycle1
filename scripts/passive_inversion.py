# 检测主体:
#   1) 第三方窃取被动方模型
#   2) 主动方窃取被动方模型 
# 检测对象:
#   1) Label: NC
#   2) Grad-based: DLG
#   3) Feature-based: Optimized
# 检测目标: 检测全局/被动方水印，重点是原始数据
#   2) Grad-based: 主动方在训练过程中记录了异常类的grad, 使用DLG进行逆向
#   3) Feature-based: 主动方在训练过程中记录了异常类的feature，使用直接优化进行逆向



import os
import sys
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils


cifar_normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
cifar_inv_normalize = transforms.Normalize((-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010), (1/0.2023, 1/0.1994, 1/0.2010))

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

        # Model
        self.bottom_model_list = [utils.get_model(self.cfg)[0].to(self.device) for _ in range(self.cfg.dataset.passive_client_num)]

        # Load Pre-trained Watermarked Model
        save_path = os.path.join('./model/final', f"{self.cfg.dataset.name}")
        for idx, model in enumerate(self.bottom_model_list):
            passive_model_path = os.path.join(save_path, f"{cfg.dataset.passive_client_num}-{idx+1}_passive_model.pth")
            state_dict = torch.load(f'{passive_model_path}')
            model.load_state_dict(state_dict)

    def passive_zero_inversion(self, image, idx):
        [bottom_model.eval() for bottom_model in self.bottom_model_list]

        recover_image = torch.zeros_like(image)
        recover_image = cifar_normalize(recover_image).to(self.device).requires_grad_(True)
        optimizer = optim.Adam([recover_image], lr=1e-2, weight_decay=5e-5)

        embedding_feature = self._get_init_feature(image).detach()

        for step in range(self.cfg.inv_steps):
            optimizer.zero_grad()

            curr_feature = self.bottom_model_list[0](recover_image)
            loss = nn.MSELoss()(curr_feature, embedding_feature)
            loss.backward()
            optimizer.step()
            torch.clamp(recover_image, -2.45, 2.52)

            if step % 1 == 0:
                print(f"Step {step}, Loss: {loss.item()}")


        self._save_image(recover_image, f'./results/{self.cfg.dataset.name}/passive_zero/{idx}_passive_inversion_feature.png')
        self._save_image(image, f'./results/{self.cfg.dataset.name}/passive_zero/{idx}_passive_inversion_feature_init.png')

    def passive_rand_inversion(self, image, idx):
        [bottom_model.eval() for bottom_model in self.bottom_model_list]

        recover_image = torch.rand_like(image)
        recover_image = cifar_normalize(recover_image).to(self.device).requires_grad_(True)
        optimizer = optim.Adam([recover_image], lr=1e-2, weight_decay=5e-5)

        embedding_feature = self._get_init_feature(image).detach()

        for step in range(self.cfg.inv_steps):
            optimizer.zero_grad()

            curr_feature = self.bottom_model_list[0](recover_image)
            loss = nn.MSELoss()(curr_feature, embedding_feature)
            loss.backward()
            optimizer.step()
            torch.clamp(recover_image, -2.45, 2.52)

            if step % 1 == 0:
                print(f"Step {step}, Loss: {loss.item()}")


        self._save_image(recover_image, f'./results/{self.cfg.dataset.name}/passive_rand/{idx}_passive_inversion_feature.png')
        self._save_image(image, f'./results/{self.cfg.dataset.name}/passive_rand/{idx}_passive_inversion_feature_init.png')

    def _save_image(self, image, path):
        image = cifar_inv_normalize(image[0]).permute(1, 2, 0).cpu().detach().numpy()
        image_pil = Image.fromarray(np.uint8(image * 255))
    
        result_dir = os.path.dirname(path)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        image_pil.save(path, dpi=(300, 300))

    def _get_init_feature(self, data):
        [bottom_model.eval() for bottom_model in self.bottom_model_list]
        embedding_feature = self.bottom_model_list[0](data)

        return embedding_feature
    

if __name__ == '__main__':
    from config_inversion import config as cfg

    print(cfg)

    vfl_framework = VFL_framwork(cfg)

    path = './scripts/inversion_cifar10'
    class_images = [os.path.join(path, f) for f in os.listdir(path)]

    for idx, image_path in enumerate(class_images):
        data = Image.open(image_path)
            
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            cifar_normalize,
        ])
        data = transform(data).unsqueeze(0)

        vfl_framework.passive_zero_inversion(data, idx)
        vfl_framework.passive_rand_inversion(data, idx)

