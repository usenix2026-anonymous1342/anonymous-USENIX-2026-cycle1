import os
import sys
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

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

        # Model
        self.top_model = utils.get_model(self.cfg)[1].to(self.device)

        # Load Pre-trained Watermarked Model
        save_path = os.path.join('./model/final', f"{self.cfg.dataset.name}")
                
        top_model_path = os.path.join(save_path, f'{cfg.dataset.passive_client_num}_active_model.pth')
        state_dict = torch.load(f'{top_model_path}')
        self.top_model.load_state_dict(state_dict)
    
    def active_zero_inversion(self, embedding, idx=None):
        self.top_model.eval()

        recover_embedding = torch.zeros_like(embedding).requires_grad_(True)
        optimizer = optim.Adam([recover_embedding], lr=1e-2, weight_decay=5e-5)

        init_logits = self._get_init_logits(embedding).detach()
        num_steps = 100
        for step in range(num_steps):
            optimizer.zero_grad()

            curr_logits = self.top_model([recover_embedding])
            loss = nn.MSELoss()(curr_logits, init_logits)
            loss.backward()
            optimizer.step()
            torch.clamp(recover_embedding, -1, 1)

            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss.item()}")
    
        self._save_image(recover_embedding, f'./results/{self.cfg.dataset.name}/active_zero/{idx}_inversion_embedding.png')
        self._save_image(embedding, f'./results/{self.cfg.dataset.name}/active_zero/{idx}_inversion_embedding_init.png')

    def active_rand_inversion(self, embedding, idx=None):
        self.top_model.eval()

        recover_embedding = torch.rand_like(embedding).requires_grad_(True)
        optimizer = optim.Adam([recover_embedding], lr=1e-2, weight_decay=5e-5)

        init_logits = self._get_init_logits(embedding).detach()
        num_steps = 100
        for step in range(num_steps):
            optimizer.zero_grad()

            curr_logits = self.top_model([recover_embedding])
            loss = nn.MSELoss()(curr_logits, init_logits)
            loss.backward()
            optimizer.step()
            torch.clamp(recover_embedding, -1, 1)

            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss.item()}")
    
        self._save_image(recover_embedding, f'./results/{self.cfg.dataset.name}/active_rand/{idx}_inversion_embedding.png')
        self._save_image(embedding, f'./results/{self.cfg.dataset.name}/active_rand/{idx}_inversion_embedding_init.png')

    def _save_image(self, image, path):
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy().reshape((32, 16))
            image = (image - image.min()) / (image.max() - image.min()) * 255
            image = image.astype(np.uint8)
            image = Image.fromarray(image)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        image.save(path)

    def _get_init_logits(self, embedding):
        init_logits = self.top_model([embedding])

        return init_logits

if __name__ == '__main__':
    from config_inversion import config as cfg

    vfl_framework = VFL_framwork(cfg)

    active_wm_path = './watermark/active_wm'
    for path in os.listdir(active_wm_path):
        if 'png' not in path:
            continue
        idx = int(path.split(".")[0])
        
        image_path = os.path.join(active_wm_path, path)
        image = Image.open(image_path).convert('L')
        image = image.resize((16, 32), Image.NEAREST)
        binary_image = image.point(lambda p: 255 if p > 128 else 0)
        binary_array = np.array(binary_image) // 255
        binary_array = binary_array.flatten()
        embedding = torch.tensor(binary_array, dtype=torch.float32).flatten().unsqueeze(0).to(cfg.device)

        vfl_framework.active_zero_inversion(embedding, idx)
        vfl_framework.active_rand_inversion(embedding, idx)