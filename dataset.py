import re
import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class GTSRB_Train_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self._load_data()

    def _load_data(self):
        for class_id in range(43):
            csv_file = os.path.join(self.root_dir, f'{class_id:05d}', f'GT-{class_id:05d}.csv')
            image_dir = os.path.join(self.root_dir, f'{class_id:05d}')
            
            df = pd.read_csv(csv_file, sep=';')
            
            for idx, row in df.iterrows():
                img_path = os.path.join(image_dir, row['Filename'])
                label = row['ClassId']
                
                self.data.append((img_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB").resize((32, 32))
        
        if self.transform:
            image = self.transform(image)

        return image, label


class GTSRB_Test_Dataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.transform = transform
        self.data = []
        self._load_data()

    def _load_data(self):
        df = pd.read_csv(self.csv_file, sep=';')

        for _, row in df.iterrows():
            img_path = os.path.join(self.root_dir, row['Filename'])
            label = row['ClassId']
                
            self.data.append((img_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB").resize((32, 32))
        
        if self.transform:
            image = self.transform(image)

        return image, label
    

class TinyImageNet_Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as f:
            data = f.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as f:
            data = f.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        classes = sorted([d.name for d in os.scandir(self.train_dir) if d.is_dir()])
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()

        with open(val_annotations_file, 'r') as f:
            entry = f.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        with open(img_path, 'rb') as f:
            image = Image.open(img_path)
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, label
  

class ImageNet_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label_str in os.listdir(root_dir):
            label = int(label_str) - 1
            class_dir = os.path.join(root_dir, label_str)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        image.resize((224, 224))

        if self.transform:
            image = self.transform(image)

        return image, label


class Split_Dataset(Dataset):
    def __init__(self, dataset, resize_size, transform=None, passive_client_num=1, global_wm=None):
        self.dataset = dataset
        self.resize_size = resize_size
        self.transform = transform
        self.passive_client_num = passive_client_num

        self.global_wm = global_wm
        self.global_wm_indices = []
        if self.global_wm is not None:
            self._global_wm_setup(self.global_wm)

    def _global_wm_setup(self, global_wm):
        self.global_wm_target = global_wm.target
        self.global_wm_ratio = global_wm.ratio

        target_samples = range(len(self.dataset))
        sample_size = int(self.global_wm_ratio * len(self.dataset)) 
        self.global_wm_indices = random.sample(target_samples, sample_size)

    def _add_global_wm(self, img, label):
        img_np = np.array(img)

        if len(img_np.shape) == 2:
            img_np = np.expand_dims(img_np, axis=-1)

        # Adding White Blocks as Watermark
        block_size = int(img_np.shape[0] / 8)
        # img_np[:block_size, :block_size, :] = 255  # Top-left
        # img_np[:block_size, -block_size:, :] = 255  # Top-right
        img_np[-block_size:, :block_size, :] = 255  # Bottom-left
        img_np[-block_size:, -block_size:, :] = 255  # Bottom-right

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

        if idx in self.global_wm_indices:
            img, label = self._add_global_wm(img, label)
        
        # Adjust image size and Split image
        img_resized = img.resize(self.resize_size)
        img_splits = self._split_image(img_resized)

        if self.transform:
            img_init_splits = [transforms.ToTensor()(split_img) for split_img in img_splits]
            img_aug_splits = [self.transform(split_img) for split_img in img_splits]
            
        return img_init_splits, img_aug_splits, label
    

class Passive_Watermark_Dataset(Dataset):
    def __init__(self, dataset, resize_size, trigger_path, base_transform=None, aug_transform=None, passive_client_num=1):
        self.data = dataset
        self.resize_size = resize_size
        self.base_transform = base_transform
        self.aug_transform = aug_transform
        self.trigger_img = Image.open(trigger_path)
        self.passive_client_num = passive_client_num

    def _split_image(self, img):
        w, h = img.size
        img_list = []
        
        h_mid = h // 2
        w_mid = w // 2
        
        if self.passive_client_num == 1:
            img_list.append(img)
        elif self.passive_client_num == 2:
            img_list.append(img.crop((0, 0, w, h_mid)))  # Top half
            img_list.append(img.crop((0, h_mid, w, h)))  # Bottom half
        elif self.passive_client_num == 4:
            img_list.append(img.crop((0, 0, w_mid, h_mid)))  # Top-left
            img_list.append(img.crop((w_mid, 0, w, h_mid)))  # Top-right
            img_list.append(img.crop((0, h_mid, w_mid, h)))  # Bottom-left
            img_list.append(img.crop((w_mid, h_mid, w, h)))  # Bottom-right
        
        return img_list

    def _add_trigger(self, img):
        # Trigger generate
        trigger_resized = self.trigger_img.resize(img.size, Image.NEAREST)
        trigger = transforms.ToTensor()(trigger_resized) 
        mask = (trigger > 0).float()

        # Trigger Paste
        img_tensor = transforms.ToTensor()(img)
        trigger_color = torch.ones_like(img_tensor)
        img_with_trigger = img_tensor * (1 - mask) + trigger_color * mask
        img_with_trigger_pil = transforms.ToPILImage()(img_with_trigger)

        return img_with_trigger_pil
           
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img, _ = self.data[index]
        img_resized = img.resize(self.resize_size)

        img_splits = self._split_image(img_resized)
        img_wm_splits = [self._add_trigger(split_img) for split_img in img_splits]

        if self.base_transform:
            img_splits = [self.base_transform(split_img) for split_img in img_splits]
            img_wm_splits = [self.base_transform(split_img) for split_img in img_wm_splits]

        if self.aug_transform:
            img_splits = [self.aug_transform(split_img) for split_img in img_splits]

        return img_splits, img_wm_splits


class Passive_Watermark_Test_Dataset(Dataset):
    def __init__(self, dataset, resize_size, label_dir, trigger_path, transform=None, passive_client_num=1):
        self.data = dataset
        self.resize_size = resize_size
        self.transform = transform
        self.label_dir = label_dir
        self.trigger_img = Image.open(trigger_path)
        self.passive_client_num = passive_client_num

        self.party_labels = self._get_target()
        print(f"Passive Target is {self.party_labels}")

    def _split_image(self, img):
        w, h = img.size
        img_list = []
        
        h_mid = h // 2
        w_mid = w // 2
        
        if self.passive_client_num == 1:
            img_list.append(img)
        elif self.passive_client_num == 2:
            img_list.append(img.crop((0, 0, w, h_mid)))  # Top half
            img_list.append(img.crop((0, h_mid, w, h)))  # Bottom half
        elif self.passive_client_num == 4:
            img_list.append(img.crop((0, 0, w_mid, h_mid)))  # Top-left
            img_list.append(img.crop((w_mid, 0, w, h_mid)))  # Top-right
            img_list.append(img.crop((0, h_mid, w_mid, h)))  # Bottom-left
            img_list.append(img.crop((w_mid, h_mid, w, h)))  # Bottom-right
        
        return img_list
    
    def _add_trigger(self, img):
        # Trigger generate
        trigger_resized = self.trigger_img.resize(img.size, Image.NEAREST)
        trigger = transforms.ToTensor()(trigger_resized) 
        mask = (trigger > 0).float()

        # Trigger Paste
        img_tensor = transforms.ToTensor()(img)
        trigger_color = torch.ones_like(img_tensor)
        img_with_trigger = img_tensor * (1 - mask) + trigger_color * mask
        img_with_trigger_pil = transforms.ToPILImage()(img_with_trigger)

        return img_with_trigger_pil

    def _get_target(self):
        party_labels = {}
        for idx in range(self.passive_client_num):
            party_path = os.path.join(self.label_dir, f"party_{idx}_of_{self.passive_client_num}")
            label_count = {}
            for file_name in os.listdir(party_path):
                label = int(file_name.split('_')[-1].replace('.png', ''))
                if label in label_count:
                    label_count[label] += 1
                else:
                    label_count[label] = 1

            target_label = max(label_count, key=label_count.get)
            party_labels[idx] = target_label
                    
        return party_labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img, _ = self.data[index]
        img_resized = img.resize(self.resize_size)
        img_splits = self._split_image(img_resized)
        img_wm_splits = [self._add_trigger(split_img) for split_img in img_splits]
 
        if self.transform is not None:
            img_splits = [transforms.ToTensor()(split_img) for split_img in img_splits]
            img_wm_splits = [self.transform(split_img) for split_img in img_wm_splits]

        target = [self.party_labels[idx] for idx in range(self.passive_client_num)]

        return img_splits, img_wm_splits, target
    

def show_list(tensor_list):
    list_len = len(tensor_list)
    
    if list_len == 1:
        rows, cols = 1, 1
    elif list_len == 2:
        rows, cols = 2, 1
    elif list_len == 4:
        rows, cols = 2, 2
    else:
        raise ValueError("The list must contain 1, 2, or 4 tensors.")
    
    _, axs = plt.subplots(rows, cols)
    
    if list_len == 1:
        axs.imshow(tensor_list[0].permute(1, 2, 0).cpu().numpy())
        axs.axis('off')
    else:
        axs = axs.flat if list_len > 1 else [axs]

        for i, ax in enumerate(axs):
            img = tensor_list[i].permute(1, 2, 0).cpu().numpy()
            ax.imshow(img)
            ax.axis('off')
    
    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torchvision.transforms import transforms

    from config import config as cfg
    from utils import get_dataset

    dataset, _, _ = get_dataset(cfg)

    for init_x, aug_x, label in dataset:
        show_list(init_x)
        show_list(aug_x)
        break

