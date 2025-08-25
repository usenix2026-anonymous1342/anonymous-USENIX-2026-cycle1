# 需要对比干净模型和后门模型，探测后门标签。现有若干方案如下：
# 1. 使用随机向量在Model上推理得到feature，然后用这些feature去fit KNN分类模型，最后在这个KNN模型上去测试随机向量分类结果
# 2. 使用随机向量在Model上推理得到feature，然后使用fit好的KNN分类模型进行分类
# 3. 使用随机向量直接使用KNN分类模型进行分类

import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt


import torch
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils


class VFL_framwork(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.passive_knn_k = self.cfg.passive_wm.knn_k

        self.bottom_model_list = [utils.get_model(self.cfg)[0].to(self.device) for _ in range(self.cfg.dataset.passive_client_num)]

        self.memory_dataset = utils.get_memory_dataset(self.cfg)
        self.memory_loader = DataLoader(self.memory_dataset, batch_size=cfg.passive_wm.batch_size, shuffle=True, drop_last=True, num_workers=8)

    def benign_load(self):
        save_path = os.path.join('./model/clean', f"{self.cfg.dataset.name}")
        for idx, model in enumerate(self.bottom_model_list):
            passive_model_path = os.path.join(save_path, f"{cfg.dataset.passive_client_num}-{idx+1}_passive_model.pth")
            state_dict = torch.load(f'{passive_model_path}')
            model.load_state_dict(state_dict)
        print("Benign Model Loaded Successfully!")

    def watermark_load(self):
        save_path = os.path.join('./model/final', f"{self.cfg.dataset.name}")
        for idx, model in enumerate(self.bottom_model_list):
            passive_model_path = os.path.join(save_path, f"{cfg.dataset.passive_client_num}-{idx+1}_passive_model.pth")
            state_dict = torch.load(f'{passive_model_path}')
            model.load_state_dict(state_dict)
        print("Watermarked Model Loaded Successfully!")

    def feature_train(self):
        for idx in range(len(self.bottom_model_list)):
            # 加载干净模型并测试
            self.benign_load()
            train_vectors = torch.randn(2000, 3, 32, 32).cuda()
            random_vectors = torch.randn(5000, 512).cuda()
            train_labels = torch.randint(0, self.cfg.dataset.num_classes, (2000,))
            train_features = self.bottom_model_list[idx](train_vectors)
            knn = KNeighborsClassifier(n_neighbors=self.passive_knn_k, weights='distance')
            knn.fit(train_features.cpu().detach().numpy(), train_labels.cpu().detach().numpy())
            benign_pred = knn.predict(random_vectors.cpu().detach().numpy())
            benign_counts = np.bincount(benign_pred, minlength=self.cfg.dataset.num_classes)
            print(f"    Benign Class distribution for Client {idx}: {benign_counts}")

            width = 0.35
            x = np.arange(len(benign_counts))
            plt.figure(figsize=(8, 6))
            plt.bar(x - width / 2, benign_counts, width, label="Benign Model", alpha=0.7)


            # 加载水印模型并测试
            self.watermark_load()
            train_features = self.bottom_model_list[idx](train_vectors)
            knn = KNeighborsClassifier(n_neighbors=self.passive_knn_k, weights='distance')
            knn.fit(train_features.cpu().detach().numpy(), train_labels.cpu().detach().numpy())
            watermark_pred = knn.predict(random_vectors.cpu().detach().numpy())
            watermark_counts = np.bincount(watermark_pred, minlength=self.cfg.dataset.num_classes)
            print(f"    Watermark Class distribution for Client {idx}: {watermark_counts}")

            plt.bar(x + width / 2, watermark_counts, width, label="Watermarked Model", alpha=0.7)
            plt.title(f'Class Distribution Comparison (Feature Train)')
            plt.xlabel('Class')
            plt.ylabel('Number of Samples')
            plt.xticks(x, range(len(benign_counts)))
            plt.legend()
            plt.savefig(f'./results/{cfg.dataset.name}/passive_random_feature_train', dpi=300)

    def feature_pretrain(self):
        for idx in range(len(self.bottom_model_list)):
            # 加载干净模型并测试
            self.benign_load()
            random_vectors = torch.randn(2000, 3, 32, 32).cuda()
            train_features, train_labels = utils.extract_features(self.bottom_model_list[idx], self.memory_loader, idx)
            knn = KNeighborsClassifier(n_neighbors=self.passive_knn_k, weights='distance')
            knn.fit(train_features.cpu().detach().numpy(), train_labels.cpu().detach().numpy())
            benign_features = self.bottom_model_list[idx](random_vectors)
            benign_pred = knn.predict(benign_features.cpu().detach().numpy())
            benign_counts = np.bincount(benign_pred, minlength=self.cfg.dataset.num_classes)
            print(f"    Benign Class distribution for Client {idx}: {benign_counts}")

            width = 0.35
            x = np.arange(len(benign_counts))
            plt.figure(figsize=(8, 6))
            plt.bar(x - width / 2, benign_counts, width, label="Benign Model", alpha=0.7)


            # 加载水印模型并测试
            self.watermark_load()
            train_features, train_labels = utils.extract_features(self.bottom_model_list[idx], self.memory_loader, idx)
            knn = KNeighborsClassifier(n_neighbors=self.passive_knn_k, weights='distance')
            knn.fit(train_features.cpu().detach().numpy(), train_labels.cpu().detach().numpy())
            watermark_features = self.bottom_model_list[idx](random_vectors)
            watermark_pred = knn.predict(watermark_features.cpu().detach().numpy())
            watermark_counts = np.bincount(watermark_pred, minlength=self.cfg.dataset.num_classes)
            print(f"    Watermark Class distribution for Client {idx}: {watermark_counts}")

            plt.bar(x + width / 2, watermark_counts, width, label="Watermarked Model", alpha=0.7)
            plt.title(f'Class Distribution Comparison (Feature Pretrain)')
            plt.xlabel('Class')
            plt.ylabel('Number of Samples')
            plt.xticks(x, range(len(benign_counts)))
            plt.legend()
            plt.savefig(f'./results/{cfg.dataset.name}/passive_random_feature_pretrain', dpi=300)

    def random_pretrain(self):
        for idx in range(len(self.bottom_model_list)):
            # 加载干净模型并测试
            self.benign_load()
            random_vectors = torch.randn(5000, 512).cuda()
            train_features, train_labels = utils.extract_features(self.bottom_model_list[idx], self.memory_loader, idx)
            knn = KNeighborsClassifier(n_neighbors=self.passive_knn_k, weights='distance')
            knn.fit(train_features.cpu().detach().numpy(), train_labels.cpu().detach().numpy())
            benign_pred = knn.predict(random_vectors.cpu().detach().numpy())
            benign_counts = np.bincount(benign_pred, minlength=self.cfg.dataset.num_classes)
            print(f"    Benign Class distribution for Client {idx}: {benign_counts}")

            width = 0.35
            x = np.arange(len(benign_counts))
            plt.figure(figsize=(8, 6))
            plt.bar(x - width / 2, benign_counts, width, label="Benign Model", alpha=0.7)


            # 加载水印模型并测试
            self.watermark_load()
            train_features, train_labels = utils.extract_features(self.bottom_model_list[idx], self.memory_loader, idx)
            knn = KNeighborsClassifier(n_neighbors=self.passive_knn_k, weights='distance')
            knn.fit(train_features.cpu().detach().numpy(), train_labels.cpu().detach().numpy())
            watermark_pred = knn.predict(random_vectors.cpu().detach().numpy())
            watermark_counts = np.bincount(watermark_pred, minlength=self.cfg.dataset.num_classes)
            print(f"    Watermark Class distribution for Client {idx}: {watermark_counts}")

            plt.bar(x + width / 2, watermark_counts, width, label="Watermarked Model", alpha=0.7)
            plt.title(f'Class Distribution Comparison (Random Pretrain)')
            plt.xlabel('Class')
            plt.ylabel('Number of Samples')
            plt.xticks(x, range(len(benign_counts)))
            plt.legend()
            plt.savefig(f'./results/{cfg.dataset.name}/passive_random_random_pretrain', dpi=300)

if __name__ == '__main__':
    from config import config as cfg

    vfl_framework = VFL_framwork(cfg)

    # Passive Watermark Embedding
    vfl_framework.feature_train()
    vfl_framework.feature_pretrain()
    vfl_framework.random_pretrain()

