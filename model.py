import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out

    
class ResNet18_Bottom(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.in_channels = 64

        if cfg.dataset.name == 'ImageNet':
            self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        if cfg.dataset.name == 'ImageNet':
            self.maxpool = nn.MaxPool2d(3, 2, 1)
        else:
            self.maxpool = nn.Identity()

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * BasicBlock.expansion
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        return x


class ResNet18_Top(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.clients = cfg.dataset.passive_client_num
        self.num_classes = cfg.dataset.num_classes

        self.classifier = nn.Sequential(
            nn.Linear(512 * self.clients, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, self.num_classes)
        )

    def forward(self, x_a, watermark=None):        
        if isinstance(x_a, list):
            x = torch.cat(x_a, dim=1)
        else:
            print("Error with x_a.")

        if watermark != None:
            x = torch.cat((x, watermark), dim=0)

        x = self.classifier(x)

        return x


if __name__ == '__main__':
    
    from config import config as cfg
    from utils import get_active_wm
    
    # Random Test
    model = ResNet18_Top(cfg).to(cfg.device)
    model.load_state_dict(torch.load(f"./model/wm/{cfg.dataset.name}/{cfg.dataset.passive_client_num}_active_model.pth"))

    model.eval()
    watermark = get_active_wm(cfg)
    output = model([watermark])
    print(output)

    ans = 0
    tensor = torch.randint(0, 2, (1000, 4096))
    tensor_float = tensor.to(torch.float32).to(cfg.device)
    output = model([tensor_float])
    predict = torch.argmax(output, dim=-1)
    counts = torch.bincount(predict, minlength=10)

    for number, count in enumerate(counts):
        print(f"Number {number} appears {count} times")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.bar(range(10), counts.tolist(), tick_label=range(10))
    plt.xlabel('Number')
    plt.ylabel('Count')
    plt.title('Frequency of Each Number in Tensor')
    plt.save('Random Test.png')

    
    # Pruning Attack
    prune_ratios = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
    for i, prune_ratio in enumerate(prune_ratios):
        model = CNN_Top().to(cfg.device)
        model.load_state_dict(torch.load("./model/1_CIFAR10_active_model.pth"))

        prune_attack(model, prune_ratio)

        ans = 0
        tensor = torch.randint(0, 2, (1000, 4096))
        tensor_float = tensor.to(torch.float32).to(cfg.device)
        output = model([tensor_float])
        predict = torch.argmax(output, dim=-1)
        counts = torch.bincount(predict, minlength=10)

        for number, count in enumerate(counts):
            print(f"Number {number} appears {count} times")

        plt.figure(figsize=(10, 6))
        plt.bar(range(10), counts.tolist(), tick_label=range(10))
        plt.xlabel('Number')
        plt.ylabel('Count')
        plt.title('Frequency of Each Number in Tensor')
        plt.show()

