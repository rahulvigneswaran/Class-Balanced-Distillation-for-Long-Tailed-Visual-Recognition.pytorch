import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
from PIL import Image

# Image statistics
RGB_statistics = {
    "iNaturalist18": {"mean": [0.466, 0.471, 0.380], "std": [0.195, 0.194, 0.192]},
    "default": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
}

# Data transformation with augmentation
def get_data_transform(split, rgb_mean, rbg_std, key=False):
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rbg_std),
            ]
        )
        if key == False
        else transforms.Compose(
            [   transforms.RandomApply([transforms.ColorJitter(brightness=(0.1, 0.3), contrast=(0.1, 0.3), saturation=(0.1, 0.3), hue=(0.1, 0.3))], p=0.8),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rbg_std),
                AddGaussianNoise(0., 0.01),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rbg_std),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rbg_std),
            ]
        ),
    }
    return data_transforms[split]


# Dataset
class LT_Dataset(Dataset):
    def __init__(self, root, txt, transform=None, template=None, top_k=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                if "val" in line:
                    if "iNaturalist18" in txt:
                        rootalt = root
                    else:
                        rootalt = "/home/rahul_intern/"
                else:
                    rootalt = root
                self.img_path.append(os.path.join(rootalt, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        # select top k class
        if top_k:
            # only select top k in training, in case train/val/test not matching.
            if "train" in txt:
                max_len = max(self.labels) + 1
                dist = [[i, 0] for i in range(max_len)]
                for i in self.labels:
                    dist[i][-1] += 1
                dist.sort(key=lambda x: x[1], reverse=True)
                # saving
                torch.save(dist, template + "_top_{}_mapping".format(top_k))
            else:
                # loading
                dist = torch.load(template + "_top_{}_mapping".format(top_k))
            selected_labels = {item[0]: i for i, item in enumerate(dist[:top_k])}
            # replace original path and labels
            self.new_img_path = []
            self.new_labels = []
            for path, label in zip(self.img_path, self.labels):
                if label in selected_labels:
                    self.new_img_path.append(path)
                    self.new_labels.append(selected_labels[label])
            self.img_path = self.new_img_path
            self.labels = self.new_labels
        self.img_num_list = list(np.unique(self.labels, return_counts=True)[1])
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]

        with open(path, "rb") as f:
            sample = Image.open(f).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, index


# Load datasets
def load_data(
    data_root,
    dataset,
    phase,
    batch_size,
    top_k_class=None,
    sampler_dic=None,
    num_workers=4,
    shuffle=True,
    special_aug=False
):

    txt_split = phase
    txt = "./libs/data/%s/%s_%s.txt" % (dataset, dataset, txt_split)
    template = "./libs/data/%s/%s" % (dataset, dataset)
    print("Loading data from %s" % (txt))
    key = special_aug

    rgb_mean, rgb_std = RGB_statistics["default"]["mean"], RGB_statistics["default"]["std"]
    if phase not in ["train", "val"]:
        transform = get_data_transform("test", rgb_mean, rgb_std, key)
    else:
        transform = get_data_transform(phase, rgb_mean, rgb_std, key)
    print("Use data transformation:", transform)

    set_ = LT_Dataset(data_root, txt, transform, template=template)

    if sampler_dic and phase == "train":
        print("=====> Using sampler: ", sampler_dic["sampler"])
        print("=====> Sampler parameters: ", sampler_dic["params"])
        return DataLoader(
            dataset=set_,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler_dic["sampler"](set_, **sampler_dic["params"]),
            num_workers=num_workers,
        )
    elif phase == "train":
        print("=====> No sampler.")
        print("=====> Shuffle is %s." % (shuffle))
        return DataLoader(
            dataset=set_,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
    else:
        print("=====> No sampler.")
        print("=====> Shuffle is %s." % (shuffle))
        return DataLoader(
            dataset=set_,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
