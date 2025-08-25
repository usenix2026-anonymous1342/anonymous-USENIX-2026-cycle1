import os
import sys
import copy
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_model, AvgMeter
from dataset import Split_Dataset, GTSRB_Test_Dataset, TinyImageNet_Dataset, ImageNet_Dataset, BrainTumor_Dataset

cifar_normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
gtsrb_normalize = transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669))
imagenet_normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
cifar_inv_normalize = transforms.Normalize((-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010), (1/0.2023, 1/0.1994, 1/0.2010))
gtsrb_inv_normalize = transforms.Normalize((-0.3403/0.2724, -0.3121/0.2608, -0.3214/0.2669), (1/0.2724, 1/0.2608, 1/0.2669))
imagenet_inv_normalize = transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))


def get_wm_dataset(cfg):   
    if cfg.dataset.name == 'CIFAR10':
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            cifar_normalize,
        ])
        resize_size = (32, 32)
        test_dataset = CIFAR10(root='./data', train=False, download=False)

    elif cfg.dataset.name == 'GTSRB':
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            gtsrb_normalize,
        ])
        resize_size = (32, 32)
        test_root_dir = './data/GTSRB/Final_Test/Images'
        test_csv_file = './data/GTSRB/GT-final_test.csv'
        test_dataset = GTSRB_Test_Dataset(root_dir=test_root_dir, csv_file=test_csv_file)

    elif cfg.dataset.name == 'TinyImageNet':
        test_transform = transforms.Compose([
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            imagenet_normalize
        ])
        resize_size = (64, 64)
        root_dir = './data/tiny-imagenet-200'
        test_dataset = TinyImageNet_Dataset(root=root_dir, train=False)

    elif cfg.dataset.name == 'ImageNet':
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            imagenet_normalize,
        ])
        resize_size = (224, 224)
        test_dir = './data/ImageNet/test'
        test_dataset = ImageNet_Dataset(test_dir)

    elif cfg.dataset.name == 'BrainTumor':
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            imagenet_normalize,
        ])
        resize_size = (224, 224)
        test_dir = './data/Brain Tumor MRI Dataset/Testing'
        test_dataset = BrainTumor_Dataset(data_dir=test_dir)

    split_wm_dataset = Split_Dataset(test_dataset, resize_size, test_transform, cfg.dataset.passive_client_num, cfg.global_wm)

    return split_wm_dataset
 

class Split_Dataset(Dataset):
    def __init__(self, dataset, resize_size, transform=None, passive_client_num=1, global_wm=None):
        self.dataset = dataset
        self.resize_size = resize_size
        self.transform = transform
        self.passive_client_num = passive_client_num

        self.global_wm = global_wm
        self.global_wm_target = global_wm.target

    def _add_global_wm(self, img):
        img_np = np.array(img)

        if len(img_np.shape) == 2:
            img_np = np.expand_dims(img_np, axis=-1)

        # Adding White Blocks as Watermark
        block_size = int(img_np.shape[0] / 8)
        img_np[:block_size, :block_size, :] = 255  # Top-l/eft
        img_np[:block_size, -block_size:, :] = 255  # Top-right
        # img_np[-block_size:, :block_size, :] = 255  # Bottom-left
        # img_np[-block_size:, -block_size:, :] = 255  # Bottom-right

        if img_np.shape[-1] == 1:
            img_np = img_np.squeeze(-1)

        img = Image.fromarray(img_np)
        label = self.global_wm_target

        return img, label
    
    def _split_image(self, img):
        w, h = img.size
        img_list = []
        
        h_mid = h // 2
        w_mid = w // 2
        
        if self.passive_client_num == 1:
            img_list.append(img)
        elif self.passive_client_num == 2:
            img_list.append(img.crop((0, 0, w, h_mid)))  # Top-half
            img_list.append(img.crop((0, h_mid, w, h)))  # Bottom-half
        elif self.passive_client_num == 4:
            img_list.append(img.crop((0, 0, w_mid, h_mid)))  # Top-left
            img_list.append(img.crop((w_mid, 0, w, h_mid)))  # Top-right
            img_list.append(img.crop((0, h_mid, w_mid, h)))  # Bottom-left
            img_list.append(img.crop((w_mid, h_mid, w, h)))  # Bottom-right
        
        return img_list

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img, label = self._add_global_wm(img)

        # Adjust image size and Split image
        img_resized = img.resize(self.resize_size)
        img_splits = self._split_image(img_resized)

        if self.transform:
            img_aug_splits = [self.transform(split_img) for split_img in img_splits]
            
        return img_aug_splits, label
  

class VFL_framwork(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device

        self.wm_acc_meter = AvgMeter()

        self.setup()

    def setup(self):
        seed = 0
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # Dataset
        split_wm_test_dataset = get_wm_dataset(self.cfg)
        wm_dataset = copy.deepcopy(split_wm_test_dataset)
        self.wm_test_loader = DataLoader(wm_dataset, batch_size=self.cfg.dataset.batch_size, shuffle=False, num_workers=8)

        # Model
        self.bottom_model_list = [get_model(self.cfg)[0].to(self.device) for _ in range(self.cfg.dataset.passive_client_num)]
        self.top_model = get_model(self.cfg)[1].to(self.device)

        # Load Pre-trained Watermarked Model
        save_path = os.path.join('./model/final/', f"{self.cfg.dataset.name}")
                
        top_model_path = os.path.join(save_path, f'{cfg.dataset.passive_client_num}_active_model.pth')
        state_dict = torch.load(f'{top_model_path}')
        self.top_model.load_state_dict(state_dict)
        # print("Active Model Loaded Successfully!")

        for idx, model in enumerate(self.bottom_model_list):
            passive_model_path = os.path.join(save_path, f"{cfg.dataset.passive_client_num}-{idx+1}_passive_model.pth")
            state_dict = torch.load(f'{passive_model_path}')
            model.load_state_dict(state_dict)
        # print("Passive Model Loaded Successfully!")

    def global_wm_test(self):
        self.top_model.eval()
        [bottom_model.eval() for bottom_model in self.bottom_model_list]

        with torch.no_grad():
            for batch_x, batch_y in self.wm_test_loader:
                if isinstance(batch_x, list):
                    x_a = [x.to(self.device) for x in batch_x]
                    batch_y = batch_y.to(self.device).view(-1)
                else:
                    print("Error with batch_x")

                output_tensor_bottom_model_list = [self.bottom_model_list[idx](x_a[idx]) for idx in
                                                     range(len(self.bottom_model_list))]
                output = self.top_model(output_tensor_bottom_model_list)
                acc = torch.sum(torch.argmax(output, dim=-1) == batch_y) / batch_y.size(0)
                self.wm_acc_meter.update(acc.item(), batch_y.size(0))

        print(f"    Global Watermark Accuracy: {vfl_framework.wm_acc_meter.get() * 100:.2f}%")
        # return vfl_framework.wm_acc_meter.get() * 100


if __name__ == '__main__':
    from config_effect_global import config as cfg

    cfg.dataset.passive_client_num = 4

    cfg.dataset.name = "CIFAR10"
    cfg.dataset.num_classes = 10
    cfg.dataset.batch_size = 256
    vfl_framework = VFL_framwork(cfg)
    vfl_framework.global_wm_test()

    cfg.dataset.name = "GTSRB"
    cfg.dataset.num_classes = 43
    cfg.dataset.batch_size = 256
    vfl_framework = VFL_framwork(cfg)
    vfl_framework.global_wm_test()

    cfg.dataset.name = "ImageNet"
    cfg.dataset.num_classes = 100
    cfg.dataset.batch_size = 512
    vfl_framework = VFL_framwork(cfg)
    vfl_framework.global_wm_test()

    cfg.dataset.name = "BrainTumor"
    cfg.dataset.num_classes = 4
    cfg.dataset.batch_size = 64
    vfl_framework = VFL_framwork(cfg)
    vfl_framework.global_wm_test()