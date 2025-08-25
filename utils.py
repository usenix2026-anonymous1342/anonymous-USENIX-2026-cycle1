import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

from dataset import Split_Dataset, GTSRB_Train_Dataset, GTSRB_Test_Dataset, TinyImageNet_Dataset, ImageNet_Dataset, Passive_Watermark_Dataset, Passive_Watermark_Test_Dataset
from model import ResNet18_Bottom, ResNet18_Top


cifar_normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
gtsrb_normalize = transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669))
imagenet_normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
cifar_inv_normalize = transforms.Normalize((-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010), (1/0.2023, 1/0.1994, 1/0.2010))
gtsrb_inv_normalize = transforms.Normalize((-0.3403/0.2724, -0.3121/0.2608, -0.3214/0.2669), (1/0.2724, 1/0.2608, 1/0.2669))
imagenet_inv_normalize = transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))


class DynamicRandomCrop:
    def __call__(self, img):
        width, height = img.size
        padding_size = min(width, height) // 8
        crop = transforms.RandomCrop(size=(height, width), padding=padding_size)
        return crop(img)
    

class AvgMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.n = 0
        self.avg = 0.

    def update(self, val, n):
        assert n > 0
        self.val += val * n
        self.n += n
        self.avg = self.val / self.n

    def get(self):
        return self.avg
    

def get_finetune_dataset(cfg):   
    if cfg.dataset.name == 'CIFAR10':
        train_transform = transforms.Compose([
            DynamicRandomCrop(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.25, 0.25), ratio=(1, 1)),
            cifar_normalize,
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            cifar_normalize,
        ])
        resize_size = (32, 32)
        train_dataset = CIFAR10(root='./data', train=True, download=True)
        test_dataset = CIFAR10(root='./data', train=False, download=False)
    elif cfg.dataset.name == 'GTSRB':
        train_transform = transforms.Compose([
            DynamicRandomCrop(),
            transforms.ToTensor(),        
            gtsrb_normalize,
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            gtsrb_normalize,
        ])
        resize_size = (32, 32)
        train_root_dir = './data/GTSRB/Final_Training/Images'
        test_root_dir = './data/GTSRB/Final_Test/Images'
        test_csv_file = './data/GTSRB/GT-final_test.csv'

        train_dataset = GTSRB_Train_Dataset(root_dir=train_root_dir)
        test_dataset = GTSRB_Test_Dataset(root_dir=test_root_dir, csv_file=test_csv_file)
    elif cfg.dataset.name == 'ImageNet':
        train_transform = transforms.Compose([
            DynamicRandomCrop(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),        
            imagenet_normalize,
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            imagenet_normalize,
        ])
        resize_size = (224, 224)
        train_dir = './data/ImageNet/train'
        test_dir = './data/ImageNet/test'

        train_dataset = ImageNet_Dataset(train_dir)
        test_dataset = ImageNet_Dataset(test_dir)


        # 将 test_dataset 分成两部分
        test_size = len(test_dataset) // 2  # 分成两部分，每部分为一半
        remaining_size = len(test_dataset) - test_size
        split_train_dataset, split_test_dataset = torch.utils.data.random_split(test_dataset, [test_size, remaining_size])


    test_size = len(test_dataset) // 2
    remaining_size = len(test_dataset) - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [test_size, remaining_size])

    split_train_dataset = Split_Dataset(train_dataset, resize_size, train_transform, cfg.dataset.passive_client_num)
    split_test_dataset = Split_Dataset(test_dataset, resize_size, test_transform, cfg.dataset.passive_client_num)
    split_wm_dataset = Split_Dataset(test_dataset, resize_size, test_transform, cfg.dataset.passive_client_num, cfg.global_wm)

    return split_train_dataset, split_test_dataset, split_wm_dataset


def get_dataset(cfg):   
    if cfg.dataset.name == 'CIFAR10':
        train_transform = transforms.Compose([
            DynamicRandomCrop(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.25, 0.25), ratio=(1, 1)),
            cifar_normalize,
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            cifar_normalize,
        ])
        resize_size = (32, 32)
        train_dataset = CIFAR10(root='./data', train=True, download=True)
        test_dataset = CIFAR10(root='./data', train=False, download=False)

    elif cfg.dataset.name == 'GTSRB':
        train_transform = transforms.Compose([
            DynamicRandomCrop(),
            transforms.ToTensor(),        
            gtsrb_normalize,
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            gtsrb_normalize,
        ])
        resize_size = (32, 32)
        train_root_dir = './data/GTSRB/Final_Training/Images'
        test_root_dir = './data/GTSRB/Final_Test/Images'
        test_csv_file = './data/GTSRB/GT-final_test.csv'

        train_dataset = GTSRB_Train_Dataset(root_dir=train_root_dir)
        test_dataset = GTSRB_Test_Dataset(root_dir=test_root_dir, csv_file=test_csv_file)

    elif cfg.dataset.name == 'TinyImageNet':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.25, 0.25), ratio=(1, 1)),
            imagenet_normalize
        ])
        test_transform = transforms.Compose([
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            imagenet_normalize
        ])
        resize_size = (64, 64)
        root_dir = './data/tiny-imagenet-200'

        train_dataset = TinyImageNet_Dataset(root=root_dir, train=True)
        test_dataset = TinyImageNet_Dataset(root=root_dir, train=False)

    elif cfg.dataset.name == 'ImageNet':
        train_transform = transforms.Compose([
            DynamicRandomCrop(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),        
            imagenet_normalize,
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            imagenet_normalize,
        ])
        resize_size = (224, 224)
        train_dir = './data/ImageNet/train'
        test_dir = './data/ImageNet/test'

        train_dataset = ImageNet_Dataset(train_dir)
        test_dataset = ImageNet_Dataset(test_dir)

    split_train_dataset = Split_Dataset(train_dataset, resize_size, train_transform, cfg.dataset.passive_client_num, cfg.global_wm)
    split_test_dataset = Split_Dataset(test_dataset, resize_size, test_transform, cfg.dataset.passive_client_num)
    cfg.global_wm.ratio = 1.
    split_wm_dataset = Split_Dataset(test_dataset, resize_size, test_transform, cfg.dataset.passive_client_num, cfg.global_wm)

    return split_train_dataset, split_test_dataset, split_wm_dataset


def get_model(cfg):
    bottom_model, top_model = ResNet18_Bottom(cfg), ResNet18_Top(cfg)

    bottom_model.apply(weight_init)
    top_model.apply(weight_init)

    return bottom_model, top_model


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)


def get_optimizer(model, lr, **kwargs):
    return torch.optim.Adam(model, lr, **kwargs)


def get_active_wm(cfg):
    if cfg.active_wm.path != None:
        total_size = 512 * cfg.dataset.passive_client_num
        watermark = []
        for path in cfg.active_wm.path:
            image = Image.open(path).convert('L')
            if total_size == 512:
                image = image.resize((16, 32), Image.NEAREST)
            elif total_size == 1024:
                image = image.resize((32, 32), Image.NEAREST)
            elif total_size == 2048:
                image = image.resize((32, 64), Image.NEAREST)
            threshold = 128
            binary_image = image.point(lambda p: 255 if p > threshold else 0)
            binary_array = np.array(binary_image) // 255
            binary_array = binary_array.flatten()
            tensor_array = torch.tensor(binary_array, dtype=torch.float32).flatten().to(cfg.device)
            watermark.append(tensor_array)
        watermark = torch.stack(watermark)
    else:
        watermark = None

    return watermark
    
    
def save_passive_target(cfg, top_heap_list):
    for idx, top_heap in enumerate(top_heap_list):
        top_images = [(item[1], item[2]) for item in sorted(top_heap, key=lambda x: -x[0])]

        # Save the top images for each passive party
        save_path = f"./watermark/passive_wm/{cfg.dataset.name}/party_{idx}_of_{cfg.dataset.passive_client_num}/"
        os.makedirs(save_path, exist_ok=True)
        for _, (img, label) in enumerate(top_images):
            img = (img * 255).byte().permute(1, 2, 0).numpy()
            img_pil = Image.fromarray(img)
            img_pil.save(os.path.join(save_path, f"image_{len(os.listdir(save_path))}_label_{label}.png"))


def get_passive_target(cfg):
    if cfg.dataset.name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            cifar_normalize,
        ])
    elif cfg.dataset.name == 'GTSRB':
        transform = transforms.Compose([
            transforms.ToTensor(),
            gtsrb_normalize,
        ])
    elif cfg.dataset.name == 'TinyImageNet':
        transform = transforms.Compose([
            transforms.ToTensor(),
            imagenet_normalize,
        ])
    elif cfg.dataset.name == 'ImageNet':
        transform = transforms.Compose([
            transforms.ToTensor(),
            imagenet_normalize,
        ])

    target_data = [[] for _ in range(cfg.dataset.passive_client_num)]
    target_path = cfg.passive_wm.target_path
    for idx in range(cfg.dataset.passive_client_num):
        party_folder_path = os.path.join(target_path, f"party_{idx}_of_{cfg.dataset.passive_client_num}")
            
        # List all image files in the target folder for the current party
        image_label_dict = {}
        for file_name in os.listdir(party_folder_path):
            if file_name.endswith(".png"):  # Ensure it's an image file
                label = int(file_name.split('_')[-1].replace('.png', ''))  # Extract label from filename
                image_path = os.path.join(party_folder_path, file_name)

                # Group images by label
                if label not in image_label_dict:
                    image_label_dict[label] = []
                image_label_dict[label].append(image_path)

        # Find the label with the most images
        target_label = max(image_label_dict, key=lambda label: len(image_label_dict[label]))
        for target_image_path in image_label_dict[target_label]:
            target_image = Image.open(target_image_path).convert('RGB')
            target_data[idx].append((target_image, target_label))
        
        # target_image_path = os.path.join(party_folder_path, os.listdir(party_folder_path)[0])
        # target_label = int(target_image_path.split('_')[-1].replace('.png', ''))
        # target_image = Image.open(target_image_path).convert('RGB')
        # target_data.append((target_image, target_label))

    return target_data, transform


def get_passive_wm_dataset(cfg):
    if cfg.dataset.name == 'CIFAR10':
        base_transform = transforms.Compose([
            transforms.ToTensor(), 
            cifar_normalize
        ])
        aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomErasing(p=0.5, scale=(0.25, 0.25), ratio=(1, 1)),
        ])
        resize_size = (32, 32)
        dataset = CIFAR10(root='./data', train=True, download=True)

    elif cfg.dataset.name == 'GTSRB':
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            gtsrb_normalize,
        ])
        aug_transform = transforms.Compose([
            transforms.RandomErasing(p=0.5, scale=(0.25, 0.25), ratio=(1, 1)),
        ])
        resize_size = (32, 32)
        train_root_dir = './data/GTSRB/Final_Training/Images'
        dataset = GTSRB_Train_Dataset(root_dir=train_root_dir)

    elif cfg.dataset.name == 'TinyImageNet':
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            imagenet_normalize,
        ])

        aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomErasing(p=0.5, scale=(0.25, 0.25), ratio=(1, 1)),
        ])
        resize_size = (64, 64)
        root_dir = './data/tiny-imagenet-200'
        
        dataset = TinyImageNet_Dataset(root=root_dir, train=True)

    elif cfg.dataset.name == 'ImageNet':
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            imagenet_normalize,
        ])
        aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomErasing(p=0.5, scale=(0.25, 0.25), ratio=(1, 1)),
        ])
        resize_size = (224, 224)
        train_dir = './data/ImageNet/train'
        dataset = ImageNet_Dataset(train_dir)

    passive_wm_dataset = Passive_Watermark_Dataset(
        dataset=dataset, 
        resize_size=resize_size,
        trigger_path=cfg.passive_wm.trigger_path,
        base_transform=base_transform, 
        aug_transform=aug_transform,
        passive_client_num=cfg.dataset.passive_client_num,
    )

    return passive_wm_dataset


def get_passive_wm_test_dataset(cfg):
    if cfg.dataset.name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            cifar_normalize,
        ])
        resize_size = (32, 32)
        dataset = CIFAR10(root='./data', train=False, download=False)

    elif cfg.dataset.name == 'GTSRB':
        transform = transforms.Compose([
            transforms.ToTensor(),
            gtsrb_normalize,
        ])
        resize_size = (32, 32)
        test_root_dir = './data/GTSRB/Final_Test/Images'
        test_csv_file = './data/GTSRB/GT-final_test.csv'
        dataset = GTSRB_Test_Dataset(root_dir=test_root_dir, csv_file=test_csv_file)

    elif cfg.dataset.name == 'TinyImageNet':
        transform = transforms.Compose([
            transforms.ToTensor(),
            imagenet_normalize,
        ])
        resize_size = (64, 64)
        root_dir = './data/tiny-imagenet-200'
        dataset = TinyImageNet_Dataset(root=root_dir, train=False)

    elif cfg.dataset.name == 'ImageNet':
        transform = transforms.Compose([
            transforms.ToTensor(),
            imagenet_normalize,
        ])
        resize_size = (224, 224)
        test_dir = './data/ImageNet/test'
        dataset = ImageNet_Dataset(test_dir)

    label_dir = cfg.passive_wm.target_path
    passive_wm_dataset = Passive_Watermark_Test_Dataset(
        dataset=dataset, 
        resize_size=resize_size,
        label_dir=label_dir,
        trigger_path=cfg.passive_wm.trigger_path,
        transform=transform, 
        passive_client_num=cfg.dataset.passive_client_num,
    )

    return passive_wm_dataset


def get_memory_dataset(cfg):   
    if cfg.dataset.name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            cifar_normalize,
        ])
        resize_size = (32, 32)
        train_dataset = CIFAR10(root='./data', train=True, download=False)

    elif cfg.dataset.name == 'GTSRB':
        transform = transforms.Compose([
            transforms.ToTensor(),
            gtsrb_normalize,
        ])
        resize_size = (32, 32)
        train_root_dir = './data/GTSRB/Final_Training/Images'
        train_dataset = GTSRB_Train_Dataset(root_dir=train_root_dir)

    elif cfg.dataset.name == 'TinyImageNet':
        transform = transforms.Compose([
            transforms.ToTensor(),
            imagenet_normalize
        ])
        resize_size = (64, 64)
        root_dir = './data/tiny-imagenet-200'
        train_dataset = TinyImageNet_Dataset(root=root_dir, train=True)

    elif cfg.dataset.name == 'ImageNet':
        transform = transforms.Compose([
            transforms.ToTensor(),
            imagenet_normalize,
        ])
        resize_size = (224, 224)
        train_dir = './data/ImageNet/train'
        train_dataset = ImageNet_Dataset(train_dir)

    split_train_dataset = Split_Dataset(train_dataset, resize_size, transform, cfg.dataset.passive_client_num)

    return split_train_dataset


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


def extract_features(model, data_loader, idx=None):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for _, images, targets in data_loader:
            if idx == None:
                images = images.cuda()
            else:
                images = images[idx].cuda()
            if isinstance(targets, list):
                targets = targets[idx]
            feature = model(images)
            features.append(feature.cpu())
            labels.append(targets)
    
    features = torch.cat(features)
    labels = torch.cat(labels)
    
    return features, labels

