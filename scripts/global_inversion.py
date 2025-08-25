# 检测主体:
#   1) 第三方窃取全局模型
# 检测对象:
#   1) Label: NC
#   2) Grad-based: DLG
#   3) Feature-based: Optimized
# 检测目标: 检测三方水印，重点同时包含原始数据和标签
#   1) Label: 直接探测异常类，即与后门相关的类别
#   2) Grad-based: 使用少量已知样本推理计算损失，并反传得到梯度, 使用DLG进行逆向
#   3) Feature-based: 使用少量已知样本推理得到中间Feature，使用直接优化进行逆向

import os
import sys
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils


cifar_normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
cifar_inv_normalize = transforms.Normalize((-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010), (1/0.2023, 1/0.1994, 1/0.2010))


def visualize_all_masks_patterns(num_classes, masks, patterns):
    """
    可视化所有类别的 mask 和 pattern，并标注出对应类别
    :param num_classes: 类别总数
    :param masks: 每个类别的 mask（列表，每个元素为一个 tensor）
    :param patterns: 每个类别的 pattern（列表，每个元素为一个 tensor）
    """
    # 创建一个 2xN 的子图，其中 N 是类别数目，2 列分别为 mask 和 pattern
    fig, axes = plt.subplots(num_classes, 2, figsize=(10, 2 * num_classes))
    
    for i in range(num_classes):
        # 获取 mask 和 pattern
        mask = masks[i][0].permute(1, 2, 0).cpu().detach().numpy()
        pattern = patterns[i][0].permute(1, 2, 0).cpu().detach().numpy()
        
        # 绘制 mask（灰度图像）
        axes[i, 0].imshow(mask)
        axes[i, 0].set_title(f"Class {i} Mask", fontsize=10)
        axes[i, 0].axis('off')  # 不显示坐标轴
        
        # 绘制 pattern（RGB图像）
        axes[i, 1].imshow(pattern)
        axes[i, 1].set_title(f"Class {i} Pattern", fontsize=10)
        axes[i, 1].axis('off')  # 不显示坐标轴

    plt.tight_layout()
    plt.savefig('./results/global_inversion_nc.png', dpi=300)


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
        self.top_model = utils.get_model(self.cfg)[1].to(self.device)

        # Load Pre-trained Watermarked Model
        save_path = os.path.join('./model/final', f"{self.cfg.dataset.name}")
                
        top_model_path = os.path.join(save_path, f'{cfg.dataset.passive_client_num}_active_model.pth')
        state_dict = torch.load(f'{top_model_path}')
        self.top_model.load_state_dict(state_dict)

        for idx, model in enumerate(self.bottom_model_list):
            passive_model_path = os.path.join(save_path, f"{cfg.dataset.passive_client_num}-{idx+1}_passive_model.pth")
            state_dict = torch.load(f'{passive_model_path}')
            model.load_state_dict(state_dict)

    def label_inversion(self, threshold=0.5, learning_rate=1e-3, epochs=100):
        """
        识别后门类别：通过对每个类别进行优化并评估生成的触发器的大小来识别后门类别
        :param model: 被检测的模型
        :param data: 待检测的数据（包含正常样本）
        :param num_classes: 类别总数
        :param threshold: 触发器大小的阈值，用于判断是否为后门类别
        :param learning_rate: 优化的学习率
        :param epochs: 训练的epochs
        :return: 返回可能的后门类别和对应的触发器大小
        """
        min_trigger_size = float('inf')
        potential_backdoor_class = None
        best_mask = None
        best_pattern = None

        data = self._load_nc_data(path='./scripts/inversion_cifar10')

        masks = []
        patterns = []

        # 针对每个类别生成触发器并评估
        for target_class in range(self.cfg.dataset.num_classes):
            print(f"正在检测类别 {target_class}...")
            mask, pattern = self._generate_trigger(data, target_class, learning_rate, epochs)
            mask_size, pattern_size = self._calculate_trigger_size(mask, pattern)

            # 如果触发器的大小小于设定阈值，认为可能存在后门
            if mask_size < threshold and pattern_size < threshold:
                print(f"可能的后门类别是 {target_class}，触发器大小: mask = {mask_size:.4f}, pattern = {pattern_size:.4f}")
                if mask_size + pattern_size < min_trigger_size:
                    min_trigger_size = mask_size + pattern_size
                    potential_backdoor_class = target_class
                    best_mask = mask
                    best_pattern = pattern
            
            # 保存每个类别的 mask 和 pattern
            masks.append(mask)
            patterns.append(pattern)

        visualize_all_masks_patterns(self.cfg.dataset.num_classes, masks, patterns)

        if potential_backdoor_class is not None:
            print(f"检测到的后门类别是 {potential_backdoor_class}")
            return potential_backdoor_class, best_mask, best_pattern
        else:
            print("未检测到后门类别。")
            return None, None, None

    def grad_inversion(self, data, label):
        dummy_data = torch.rand_like(data)
        dummy_data = cifar_normalize(dummy_data).to(self.device).requires_grad_(True)
        dummy_label = torch.randn((1, self.cfg.dataset.num_classes)).to(self.device).requires_grad_(True)

        optimizer = torch.optim.LBFGS([dummy_data, dummy_label], max_iter=1, lr=1e-2)
        mse_loss = nn.MSELoss()

        embedding_grad = self._get_init_grad(data, label).detach()
        for iters in range(200):   
            def closure():
                optimizer.zero_grad()   
                dummy_bottom_output = self.bottom_model_list[0](dummy_data)

                dummy_embedding = dummy_bottom_output.detach().requires_grad_()

                dummy_pred = self.top_model([dummy_embedding])
                dummy_loss_top = - torch.mean(torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(dummy_pred, -1)), dim=-1))
                dummy_loss_top.backward()

                dummy_grad = dummy_embedding.grad.requires_grad_()

                dummy_loss_bottom = torch.sum(dummy_grad * dummy_bottom_output)
                dummy_loss_bottom.backward()

                grad_diff = 0
                for gx, gy in zip(dummy_grad, embedding_grad): 
                    grad_diff += ((gx - gy) ** 2).sum()
                                
                grad_diff.backward()
                return grad_diff
            
            optimizer.step(closure)
            torch.clamp(dummy_data, -3, 3)
            torch.clamp(dummy_label, 0, self.cfg.dataset.num_classes)

            if iters % 10 == 0: 
                mse = mse_loss(data, dummy_data)
                print(f"{iters} Data MSE:{mse:.4f}")

        mse = mse_loss(data, dummy_data)
        label_recover = torch.argmax(dummy_label)
        
        dummy_data = cifar_inv_normalize(dummy_data)
        plt.imshow(dummy_data[0].permute(1, 2, 0).cpu().detach().numpy())
        plt.savefig(f'./results/global_inversion_grad_label{label_recover}.png', dpi=300)

        return mse, label_recover
    
    def global_inversion(self, image):
        recover_image = torch.zeros_like(image)
        recover_image = cifar_normalize(recover_image).to(self.device).requires_grad_(True)
        optimizer = optim.Adam([recover_image], lr=1e-3, weight_decay=5e-5)

        logits = self._get_init_logits(image).detach()
        num_steps = 500
        for step in range(num_steps):
            optimizer.zero_grad()

            curr_embedding = self.bottom_model_list[0](recover_image)
            curr_logits = self.top_model([curr_embedding])
            loss = nn.MSELoss()(curr_logits, logits)
            loss.backward()
            optimizer.step()
            torch.clamp(recover_image, -2.43, 2.51)

            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss.item()}")


        recover_image = cifar_inv_normalize(recover_image[0]).permute(1, 2, 0).cpu().detach().numpy()
        recover_image_pil = Image.fromarray(np.uint8(recover_image * 255))
        recover_image_pil.save('./results/global_inversion_feature.png', dpi=(300, 300))

        image = cifar_inv_normalize(image[0]).permute(1, 2, 0).cpu().detach().numpy()
        image_pil = Image.fromarray(np.uint8(image * 255))
        image_pil.save('./results/global_inversion_feature_init.png', dpi=(300, 300))

        # recover_image = cifar_inv_normalize(recover_image)
        # plt.imshow(recover_image[0].permute(1, 2, 0).cpu().detach().numpy())
        # plt.savefig('./results/global_inversion_feature.png', dpi=300)
        # data = cifar_inv_normalize(data)
        # plt.imshow(data[0].permute(1, 2, 0).cpu().detach().numpy())
        # plt.savefig('./results/global_inversion_feature_init.png', dpi=300)

    def _load_nc_data(self, path, num_classes=10):
        images = []
        for label in range(num_classes):
            class_images = [os.path.join(path, f) for f in os.listdir(path) if f.startswith(str(label))]
            random.shuffle(class_images)
            for img_path in class_images[:10]:
                img = Image.open(img_path)
                images.append(img)
        
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            cifar_normalize,
        ])
        
        img_tensor = torch.stack([transform(img) for img in images])  # 转换为 tensor 并堆叠
        return img_tensor

    def _generate_trigger(self, data, target_class, learning_rate=1e-4, epochs=100):
        """
        基于 Neural Cleanse 生成后门触发器 (mask 和 pattern)
        """
        mask = torch.randn_like(data[0]).unsqueeze(0).float().requires_grad_(True)
        pattern = torch.randn_like(data[0]).unsqueeze(0).float().requires_grad_(True)

        optimizer = optim.Adam([mask, pattern], lr=learning_rate, weight_decay=5e-5)
        criterion = nn.CrossEntropyLoss()

        self.top_model.eval()
        [bottom_model.eval() for bottom_model in self.bottom_model_list]

        def inject_trigger(img, mask, pattern):
            return mask * pattern + (1 - mask) * img

        for epoch in range(epochs):
            optimizer.zero_grad()

            idx = random.randint(0, len(data) - 1)
            img = data[idx].unsqueeze(0).float().requires_grad_(True)  # 加载一张图片
            triggered_img = inject_trigger(img, mask, pattern)
            label = target_class  # 目标类别

            embedding = self.bottom_model_list[0](triggered_img)
            output = self.top_model([embedding])
            loss = criterion(output, torch.tensor([label]))  # 计算损失

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

        return mask.detach(), pattern.detach()

    def _calculate_trigger_size(self, mask, pattern):
        return torch.norm(mask).item(), torch.norm(pattern).item()

    def _get_init_grad(self, data, label):
        self.top_model.eval()
        [bottom_model.eval() for bottom_model in self.bottom_model_list]

        criterion = nn.CrossEntropyLoss()

        data = data.requires_grad_(True)

        embedding = self.bottom_model_list[0](data).clone().detach().requires_grad_(True)
        output = self.top_model([embedding])
        loss = criterion(output, torch.tensor([label]).to(output.device))

        self.top_model.zero_grad()
        [bottom_model.zero_grad() for bottom_model in self.bottom_model_list]
        loss.backward()
        embedding_grad = embedding.grad

        return embedding_grad

    def _get_init_logits(self, data):
        self.top_model.eval()
        [bottom_model.eval() for bottom_model in self.bottom_model_list]
        embedding = self.bottom_model_list[0](data)
        logits = self.top_model([embedding])

        return logits


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from config_inversion import config as cfg


    vfl_framework = VFL_framwork(cfg)

    path = './scripts/inversion_cifar10'
    label = 0
    class_images = [os.path.join(path, f) for f in os.listdir(path) if f.startswith(str(label))]
    data = Image.open(class_images[0])
        
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        cifar_normalize,
    ])
    data = transform(data).unsqueeze(0)

    # vfl_framework.label_inversion()
    # vfl_framework.grad_inversion(data, label)
    vfl_framework.global_inversion(data)

