import os
import sys
import copy
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils


def expand_model(model, layers=None):
    if layers is None:
        layers = torch.Tensor()
    for layer in model.children():
        if len(list(layer.children())) > 0:
            layers = expand_model(layer, layers)
        else:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                if layers.numel() == 0:
                    layers = layer.weight.view(-1)
                else:
                    layers = torch.cat((layers, layer.weight.view(-1)))
    return layers


def prune_attack(model, prune_ratio):
    # Expand model weights
    empty = torch.Tensor()
    pre_abs = expand_model(model, empty)
    
    if pre_abs.numel() == 0:
        raise ValueError("The model does not contain any Conv2d layers.")
    
    # Calculate pruning threshold
    weights = torch.abs(pre_abs)
    threshold = np.percentile(weights.detach().cpu().numpy(), prune_ratio)

    # Applying Pruning
    for layer in model.children():
        if len(list(layer.children())) > 0:
            prune_attack(layer, prune_ratio)
        else:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                mask = torch.abs(layer.weight.data) > threshold
                layer.weight.data *= mask.float()


class VFL_framwork(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device

        self.setup()
        self.active_setup()
        self.passive_setup()

        self.best_bottom_model_list = None
        self.best_bottom_model_B = None
        self.best_top_model = None
        self.best_test_acc = 0.0
        self.best_wm_acc = 0.0

        self.train_acc_meter = utils.AvgMeter()
        self.train_loss_meter = utils.AvgMeter()
        self.test_acc_meter = utils.AvgMeter()
        self.wm_acc_meter = utils.AvgMeter()

    def setup(self):
        seed = 0
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # Dataset
        split_train_dataset, split_test_dataset, split_wm_test_dataset = utils.get_dataset(self.cfg)
        train_dataset = copy.deepcopy(split_train_dataset)
        test_dataset = copy.deepcopy(split_test_dataset)
        wm_dataset = copy.deepcopy(split_wm_test_dataset)
        
        # DataLoader
        self.train_loader = DataLoader(train_dataset, batch_size=self.cfg.dataset.batch_size, shuffle=True, drop_last=True, num_workers=8)
        self.test_loader = DataLoader(test_dataset, batch_size=self.cfg.dataset.batch_size, shuffle=False, num_workers=8)
        self.wm_test_loader = DataLoader(wm_dataset, batch_size=self.cfg.dataset.batch_size, shuffle=False, num_workers=8)

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

    def active_setup(self):
        self.active_wm = utils.get_active_wm(self.cfg)
        self.active_wm_target = self.cfg.active_wm.target
    
    def passive_setup(self):
        # Passive Watermark
        self.passive_knn_k = self.cfg.passive_wm.knn_k

        # Dataset
        self.memory_dataset = utils.get_memory_dataset(self.cfg)
        self.passive_wm_test_dataset = utils.get_passive_wm_test_dataset(self.cfg)

        # DataLoader
        self.memory_loader = DataLoader(self.memory_dataset, batch_size=cfg.passive_wm.batch_size, shuffle=True, drop_last=True, num_workers=8)
        self.passive_wm_test_loader = DataLoader(self.passive_wm_test_dataset, batch_size=cfg.passive_wm.batch_size, shuffle=True, drop_last=True, num_workers=8)

    def prune(self, prune_ratio=0.0):
        self.top_model.train()
        [bottom_model.train() for bottom_model in self.bottom_model_list]

        for idx in range(len(self.bottom_model_list)):
            prune_attack(self.bottom_model_list[idx], prune_ratio)

    def test(self):
        self.top_model.eval()
        [bottom_model.eval() for bottom_model in self.bottom_model_list]

        self.test_acc_meter.reset()

        with torch.no_grad():
            for _, batch_x, batch_y in self.test_loader:
                if isinstance(batch_x, list):
                    x_a = [x.to(self.device) for x in batch_x]
                    batch_y = batch_y.to(self.device).view(-1)
                else:
                    print("Error with batch_x")

                output_tensor_bottom_model_list = [self.bottom_model_list[idx](x_a[idx]) for idx in
                                                     range(len(self.bottom_model_list))]
                output = self.top_model(output_tensor_bottom_model_list)
                acc = torch.sum(torch.argmax(output, dim=-1) == batch_y) / batch_y.size(0)
                self.test_acc_meter.update(acc.item(), batch_y.size(0))
    
        print(f"    Test Accuracy: {vfl_framework.test_acc_meter.get() * 100:.2f}%")
        return vfl_framework.test_acc_meter.get() * 100
    
    def global_wm_test(self):
        self.top_model.eval()
        [bottom_model.eval() for bottom_model in self.bottom_model_list]

        self.wm_acc_meter.reset()

        with torch.no_grad():
            for _, batch_x, batch_y in self.wm_test_loader:
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

        if self.wm_acc_meter.get() > self.best_wm_acc:
            self.best_wm_acc = self.wm_acc_meter.get()

        print(f"    Global Watermark Accuracy: {vfl_framework.wm_acc_meter.get() * 100:.2f}%")
        return vfl_framework.wm_acc_meter.get() * 100
    
    def active_wm_test(self):
        if self.active_wm != None:
            self.top_model.eval()

            output = self.top_model([self.active_wm])
            predict = torch.argmax(output, dim=-1).tolist()
            
            correct = sum(a == b for a, b in zip(predict, self.active_wm_target))

            print(f"    Active Watermark Accuracy: {correct / len(self.active_wm_target) * 100:.2f}%")
            print(f"        Target: {self.active_wm_target}")
            print(f"        Predict: {predict}")
        return correct / len(self.active_wm_target) * 100
    
    def passive_wm_test(self):
        for idx in range(len(self.bottom_model_list)):
            train_features, train_labels = utils.extract_util_features(self.bottom_model_list[idx], self.memory_loader, idx)
            passive_wm_features, passive_wm_labels = utils.extract_features(self.bottom_model_list[idx], self.passive_wm_test_loader, idx)

            knn = KNeighborsClassifier(n_neighbors=self.passive_knn_k, weights='distance')
            knn.fit(train_features, train_labels)

            passive_wm_pred = knn.predict(passive_wm_features)
            passive_wm_accuracy = accuracy_score(passive_wm_labels, passive_wm_pred)
            print(f'    Passive Watermark Accuracy for Client {idx}: {passive_wm_accuracy * 100:.2f}%')
        return passive_wm_accuracy * 100



if __name__ == '__main__':
    from config_prune import config as cfg

    vfl_framework = VFL_framwork(cfg)

    w_acc_list = []
    g_wsr_list = []
    a_wsr_list = []
    p_wsr_list = []

    prune_ratios = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0]
    for prune_ratio in prune_ratios:
        print(f"Prune Ratio: {prune_ratio}")
        vfl_framework.setup()
        vfl_framework.prune(prune_ratio)

        w_acc = vfl_framework.test()
        g_wsr = vfl_framework.global_wm_test()
        a_wsr = vfl_framework.active_wm_test()
        p_wsr = vfl_framework.passive_wm_test()

        w_acc_list.append(w_acc)
        g_wsr_list.append(g_wsr)
        a_wsr_list.append(a_wsr)
        p_wsr_list.append(p_wsr)


    import os
    import csv
    import matplotlib.pyplot as plt

    result_path = f"./results/{cfg.dataset.name}/passive_prune.csv"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    with open(result_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Prune Ratio", "W-Acc", "G-WSR", "A-WSR", "P-WSR"])
        for i in range(len(prune_ratios)):
            writer.writerow([
                prune_ratios[i],
                w_acc_list[i],
                g_wsr_list[i],
                a_wsr_list[i],
                p_wsr_list[i]
            ])

    # 测试结果可视化
    plt.figure()
    plt.plot(prune_ratios, w_acc_list, label="W-Acc")
    plt.plot(prune_ratios, g_wsr_list, label="G-WSR")
    plt.plot(prune_ratios, a_wsr_list, label="A-WSR")
    plt.plot(prune_ratios, p_wsr_list, label="P-WSR")
    plt.xlabel("Prune Ratio")
    plt.ylabel("Metrics")
    plt.ylim(-5, 105)
    plt.yticks([0, 20, 40, 60, 80, 100])
    plt.title("Test Metrics Over Differenet Prune Ratios")
    plt.legend()
    plt.grid()
    plt.savefig(f'./results/{cfg.dataset.name}/passive_prune.png', dpi=300)