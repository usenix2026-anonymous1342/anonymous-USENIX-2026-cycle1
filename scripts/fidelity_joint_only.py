import os
import sys
import copy
import random
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils


class VFL_framwork(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device

        self.setup()

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
        seed = self.cfg.seed
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # Dataset
        split_train_dataset, split_test_dataset, split_wm_test_dataset = utils.get_dataset(self.cfg)
        train_dataset = copy.deepcopy(split_train_dataset)
        test_dataset = copy.deepcopy(split_test_dataset)
        wm_dataset = copy.deepcopy(split_wm_test_dataset)
        
        # DataLoader
        # self.train_loader = DataLoader(train_dataset, batch_size=self.cfg.dataset.batch_size, shuffle=True, drop_last=True)
        # self.test_loader = DataLoader(test_dataset, batch_size=self.cfg.dataset.batch_size, shuffle=False)
        # self.wm_test_loader = DataLoader(wm_dataset, batch_size=self.cfg.dataset.batch_size, shuffle=False)

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

    def train(self):
        self.top_model.train()
        [bottom_model.train() for bottom_model in self.bottom_model_list]

        self.train_acc_meter.reset()
        self.train_loss_meter.reset()

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

            output = self.top_model(input_tensor_top_model_a_list)
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

    def adjust_lr(self):
        for param_group in self.top_model_optimizer.param_groups:
            param_group['lr'] *= 0.8
        for idx in range(len(self.bottom_model_optimizer_list)):
            for param_group in self.bottom_model_optimizer_list[idx].param_groups:
                param_group['lr'] *= 0.8

if __name__ == '__main__':
    from config_fidelity_joint import config as cfg

    print(cfg)

    vfl_framework = VFL_framwork(cfg)

    print("Training Start!")
    for round in range(cfg.num_rounds):
        vfl_framework.train()
        print(f"Training Round: {round+1}, "
            f"Train acc: {vfl_framework.train_acc_meter.get()}, "
            f"Train loss: {vfl_framework.train_loss_meter.get()} ")
            
        vfl_framework.test()
        vfl_framework.global_wm_test()

        if round % 20 == 0:
            vfl_framework.adjust_lr()

    print("Training Over!")
    print(f"Round: Best, Test Acc: {vfl_framework.best_test_acc}")
    