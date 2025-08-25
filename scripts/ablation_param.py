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
        if not os.path.exists(save_path):
            os.makedirs(save_path)

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
        self.active_setup()

        self.test_acc_history = []
        self.wm_acc_history = []
        
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

        # Others
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.bottom_model_optimizer_list = [utils.get_optimizer(bottom_model.parameters(), **self.cfg.optimizer) for
                                              bottom_model in self.bottom_model_list]
        self.top_model_optimizer = utils.get_optimizer(self.top_model.parameters(), **self.cfg.optimizer)

        # Load Pre-trained Watermarked Model
        if self.cfg.pretrain == True:
            save_path = os.path.join('./model/ablation', f"{self.cfg.dataset.name}")
                
            top_model_path = os.path.join(save_path, f'Param_{cfg.dataset.passive_client_num}_active_model.pth')
            state_dict = torch.load(f'{top_model_path}')
            self.top_model.load_state_dict(state_dict)
            print("Active Model Loaded Successfully!")

            for idx, model in enumerate(self.bottom_model_list):
                passive_model_path = os.path.join(save_path, f"Param_{cfg.dataset.passive_client_num}-{idx+1}_passive_model.pth")
                state_dict = torch.load(f'{passive_model_path}')
                model.load_state_dict(state_dict)
            print("Passive Model Loaded Successfully!")

    def active_setup(self):
        self.active_wm = utils.get_active_wm(self.cfg)
        self.active_wm_target = self.cfg.active_wm.target
    
    def passive_setup(self):
        # Passive Watermark
        self.passive_rounds = self.cfg.passive_wm.num_rounds
        self.passive_lr = self.cfg.passive_wm.lr
        self.passive_weight_decay = self.cfg.passive_wm.weight_decay
        self.passive_momentum = self.cfg.passive_wm.momentum
        self.passive_knn_k = self.cfg.passive_wm.knn_k

        # Dataset
        self.memory_dataset = utils.get_memory_dataset(self.cfg)
        self.passive_wm_dataset = utils.get_passive_wm_dataset(self.cfg)
        self.passive_wm_test_dataset = utils.get_passive_wm_test_dataset(self.cfg)

        # DataLoader
        self.memory_loader = DataLoader(self.memory_dataset, batch_size=cfg.passive_wm.batch_size, shuffle=True, drop_last=True, num_workers=8)
        self.passive_wm_loader = DataLoader(self.passive_wm_dataset, batch_size=cfg.passive_wm.batch_size, shuffle=True, drop_last=True, num_workers=8)
        self.passive_wm_test_loader = DataLoader(self.passive_wm_test_dataset, batch_size=cfg.passive_wm.batch_size, shuffle=True, drop_last=True, num_workers=8)

        # BadEncoder Tools
        self.clean_encoder_list = copy.deepcopy(self.bottom_model_list)
        self.passive_wm_optimizer_list = [torch.optim.SGD(
            self.bottom_model_list[i].parameters(),
            lr=self.passive_lr,
            weight_decay=self.passive_weight_decay,
            momentum=self.passive_momentum
        ) for i in range(len(self.bottom_model_list))]
        self.target_data, self.target_transform = utils.get_passive_target(self.cfg)

    def train(self, round=0):
        self.top_model.train()
        [bottom_model.train() for bottom_model in self.bottom_model_list]

        self.train_acc_meter.reset()
        self.train_loss_meter.reset()

        # Use a min-heap to track top 5 gradient norms and corresponding images
        top_heap_list = [[] for _ in range(len(self.bottom_model_list))]

        for init_x, batch_x, batch_y in tqdm(self.train_loader):
            # BOTTOM-Model Infer
            if isinstance(init_x, list):
                x = [x.to(self.device) for x in init_x]
                x_a = [x.to(self.device) for x in batch_x]
                batch_y = batch_y.to(self.device).view(-1)
            else:
                print("Error with init_x")

            output_tensor_bottom_model_list = [self.bottom_model_list[idx](x_a[idx]) for idx in
                                                 range(len(self.bottom_model_list))]
            
            # TOP-Model Infer and Update
            input_tensor_top_model_a_list = [torch.tensor([], requires_grad=True) for _ in
                                             range(len(self.bottom_model_list))]
            for idx in range(len(self.bottom_model_list)):
                input_tensor_top_model_a_list[idx].data = output_tensor_bottom_model_list[idx].data

            self.top_model_optimizer.zero_grad()

            # Active Watermark Embedding
            if self.active_wm != None:
                watermark_label = torch.tensor(self.active_wm_target, dtype=torch.long).to(self.cfg.device)
                batch_y = torch.cat((batch_y, watermark_label), dim=0)

            output = self.top_model(input_tensor_top_model_a_list, self.active_wm)
            loss = self.criterion(output, batch_y)
            loss.backward()
            self.top_model_optimizer.step()

            acc = torch.sum(torch.argmax(output, dim=-1) == batch_y) / batch_y.size(0)
            self.train_acc_meter.update(acc.item(), batch_y.size(0))
            self.train_loss_meter.update(loss.item(), output.size(0))


            # BOTTOM-Model Update
            grad_output_bottom_model_list = [input_tensor_top_model_a_list[idx].grad for idx in range(len(self.bottom_model_list))]

            for idx in range(len(self.bottom_model_list)):
                self.bottom_model_optimizer_list[idx].zero_grad()
                loss_bottom_A = torch.sum(grad_output_bottom_model_list[idx] * output_tensor_bottom_model_list[idx])
                loss_bottom_A.backward()
                self.bottom_model_optimizer_list[idx].step()

                # Record gradient norms for GraNd score calculation for each passive party
                if round < max(self.cfg.select_round):
                    grad_norm = grad_output_bottom_model_list[idx].view(grad_output_bottom_model_list[idx].size(0), -1).norm(2, dim=1).detach().cpu().numpy()
                    for i in range(len(grad_norm)):
                        if len(top_heap_list[idx]) < 20:
                            heapq.heappush(top_heap_list[idx], (grad_norm[i], x[idx][i].cpu(), batch_y[i].item()))
                        else:
                            heapq.heappushpop(top_heap_list[idx], (grad_norm[i], x[idx][i].cpu(), batch_y[i].item()))

        for select_round, select_num in zip(self.cfg.select_round, self.cfg.select_num):
            if round >= select_round:
                continue

            save_passive_target(self.cfg, top_heap_list, select_round, select_num)

    def passive_wm_embed(self):
        [clean_encoder.eval() for clean_encoder in self.clean_encoder_list]
        [bottom_model.train() for bottom_model in self.bottom_model_list]

        # Fixed BN layer
        for bottom_model in self.bottom_model_list:
            for module in bottom_model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    if hasattr(module, 'weight'):
                        module.weight.requires_grad_(False)
                    if hasattr(module, 'bias'):
                        module.bias.requires_grad_(False)
                    module.eval()
        
        # Extract Target Features with Watermarked Encoder
        feature_target_list = [[] for _ in range(self.cfg.dataset.passive_client_num)]
        for idx in range(len(self.bottom_model_list)):
            for target_image, _ in self.target_data[idx]:
                target_image = self.target_transform(target_image).unsqueeze(0).to(self.device)
                feature_target = self.bottom_model_list[idx](target_image)
                feature_target = F.normalize(feature_target, dim=-1)
                feature_target_list[idx].append(feature_target.clone().detach())

        # Extract Watermark Features with Watermarked Encoder
        passive_watermark_bar = tqdm(self.passive_wm_loader, desc="Passive Watermark Embedding")
        for x_list, x_wm_list in passive_watermark_bar:
            clean_feature_list = []
            feature_list = [[] for _ in range(self.cfg.dataset.passive_client_num)]
            feature_wm_list = [[] for _ in range(self.cfg.dataset.passive_client_num)]
            for idx in range(len(self.bottom_model_list)):
                # Extract Clean Features with Clean Encoder
                with torch.no_grad():
                    clean_feature = self.clean_encoder_list[idx](x_list[idx].cuda())
                    clean_feature = F.normalize(clean_feature, dim=-1)
                    clean_feature_list.append(clean_feature)

                feature = self.bottom_model_list[idx](x_list[idx].cuda())
                feature = F.normalize(feature, dim=-1)
                feature_list[idx].append(feature.clone().requires_grad_(True))
                feature_list[idx] = torch.cat(feature_list[idx], dim=0)

                feature_wm = self.bottom_model_list[idx](x_wm_list[idx].cuda())
                feature_wm = F.normalize(feature_wm, dim=-1)
                feature_wm_list[idx].append(feature_wm.clone().requires_grad_(True))
                feature_wm_list[idx] = torch.cat(feature_wm_list[idx], dim=0)

                # Calculate Losses and Update
                self.passive_wm_optimizer_list[idx].zero_grad()

                loss_0_list = []
                for i in range(len(feature_target_list[idx])):
                    loss_0_list.append(-torch.sum(feature_wm_list[idx]*feature_target_list[idx][i], dim=-1).mean())
                loss_0 = sum(loss_0_list) / len(loss_0_list)
                loss_2 = -torch.sum(feature_list[idx]*clean_feature_list[idx], dim=-1).mean()

                loss = loss_0 + loss_2
                loss.backward()
                self.passive_wm_optimizer_list[idx].step()
                passive_watermark_bar.set_description(f"    Passive Watermark Embedding, Loss:{loss:.4f}, Loss0:{loss_0:.4f}, Loss2:{loss_2:.4f}")
     
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
    
        self.test_acc_history.append(self.test_acc_meter.get())
        
        if self.test_acc_meter.get() > self.best_test_acc:
            self.best_test_acc = self.test_acc_meter.get()
            self.best_bottom_model_list = copy.deepcopy(self.bottom_model_list)
            self.best_top_model = copy.deepcopy(self.top_model)
            
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
        avg_p_wsr = 0.0
        for idx in range(len(self.bottom_model_list)):
            train_features, train_labels = utils.extract_util_features(self.bottom_model_list[idx], self.memory_loader, idx)
            passive_wm_features, passive_wm_labels = utils.extract_features(self.bottom_model_list[idx], self.passive_wm_test_loader, idx)

            knn = KNeighborsClassifier(n_neighbors=self.passive_knn_k, weights='distance')
            knn.fit(train_features, train_labels)

            passive_wm_pred = knn.predict(passive_wm_features)
            passive_wm_accuracy = accuracy_score(passive_wm_labels, passive_wm_pred)
            print(f'    Passive Watermark Accuracy for Client {idx}: {passive_wm_accuracy * 100:.2f}%')
            avg_p_wsr += passive_wm_accuracy
        avg_p_wsr /= len(self.best_bottom_model_list)
        return avg_p_wsr

    def save(self):
        save_path = os.path.join('./model/ablation', f"{self.cfg.dataset.name}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for idx in range(len(self.bottom_model_list)):
            passive_model_path = os.path.join(save_path, f"Param_{cfg.dataset.passive_client_num}-{idx+1}_passive_model.pth")
            torch.save(self.best_bottom_model_list[idx].state_dict(), passive_model_path)
            self.bottom_model_list[idx] = copy.deepcopy(self.best_bottom_model_list[idx])
            
        top_model_path = os.path.join(save_path, f'Param_{cfg.dataset.passive_client_num}_active_model.pth')
        torch.save(self.best_top_model.state_dict(), top_model_path)
        self.top_model = copy.deepcopy(self.best_top_model)

    def adjust_lr(self):
        for param_group in self.top_model_optimizer.param_groups:
            param_group['lr'] *= 0.8
        for idx in range(len(self.bottom_model_optimizer_list)):
            for param_group in self.bottom_model_optimizer_list[idx].param_groups:
                param_group['lr'] *= 0.8


if __name__ == '__main__':
    from config_ablation_param import config as cfg

    vfl_framework = VFL_framwork(cfg)

    if cfg.pretrain is False:
        print("Training Start!")
        for round in range(cfg.num_rounds):
            vfl_framework.train(round)
            print(f"Training Round: {round+1}, "
                f"Train acc: {vfl_framework.train_acc_meter.get()}, "
                f"Train loss: {vfl_framework.train_loss_meter.get()} ")
            
            vfl_framework.test()
            vfl_framework.global_wm_test()
            vfl_framework.active_wm_test()

            if round % 20 == 0:
                vfl_framework.adjust_lr()

        print("Training Over!")
        print(f"Round: Best, Test Acc: {vfl_framework.best_test_acc}")
        print()
        vfl_framework.save()
    
    # Passive Watermark Embedding
    vfl_framework.passive_setup()

    w_acc_list = []
    g_wsr_list = []
    a_wsr_list = []
    p_wsr_list = []

    for round in range(vfl_framework.passive_rounds):

        w_acc = vfl_framework.test()
        g_wsr = vfl_framework.global_wm_test()
        a_wsr = vfl_framework.active_wm_test()
        p_wsr = vfl_framework.passive_wm_test()

        w_acc_list.append(w_acc)
        g_wsr_list.append(g_wsr)
        a_wsr_list.append(a_wsr)
        p_wsr_list.append(p_wsr)
        
        print(f"Embedding Round: {round+1}, ")
        vfl_framework.passive_wm_embed()

    print("Final Result")
    w_acc = vfl_framework.test()
    g_wsr = vfl_framework.global_wm_test()
    a_wsr = vfl_framework.active_wm_test()
    p_wsr = vfl_framework.passive_wm_test()

    w_acc_list.append(w_acc)
    g_wsr_list.append(g_wsr)
    a_wsr_list.append(a_wsr)
    p_wsr_list.append(p_wsr)

    # vfl_framework.save()

    rounds = range(0, len(w_acc_list))
    result_path = f"./results/{cfg.dataset.name}/ablation_parameter_{cfg.select_round}_{cfg.select_num}.csv"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    with open(result_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Round", "W-Acc", "G-WSR", "A-WSR", "P-WSR"])
        for i in range(len(w_acc_list)):
            writer.writerow([
                i,
                w_acc_list[i],
                g_wsr_list[i],
                a_wsr_list[i],
                p_wsr_list[i]
            ])

    # 测试结果可视化
    plt.figure()
    plt.plot(rounds, w_acc_list, label="W-Acc")
    plt.plot(rounds, g_wsr_list, label="G-WSR")
    plt.plot(rounds, a_wsr_list, label="A-WSR")
    plt.plot(rounds, p_wsr_list, label="P-WSR")
    plt.xlabel("Rounds")
    plt.ylabel("Metrics")
    plt.ylim(-5, 105)
    plt.yticks([0, 20, 40, 60, 80, 100])
    plt.title("Test Metrics Over Rounds")
    plt.legend()
    plt.grid()
    plt.savefig(f'./results/{cfg.dataset.name}/ablation_parameter_{cfg.select_round}_{cfg.select_num}', dpi=300)