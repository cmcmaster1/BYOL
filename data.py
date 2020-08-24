import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import os
from PIL import Image
import torchvision.transforms.functional as TF
import re

# images dataset

def expand_greyscale(t):
    return t.expand(3, -1, -1)

class BYOLImagesDataset(Dataset):
    def __init__(self, folder, image_size, exts):
        super().__init__()
        self.folder = folder
        self.exts = exts

        self.paths = []
        self.labels = []
        self.angles = []
        for path in self.folder.glob('**/*'):
                    _, ext = os.path.splitext(path)
                    if ext.lower() in ['.png']:
                        for angle in range(0, 330, 15):
                            self.paths.append(path)
                            self.angles.append(angle)


        print(f'{len(self.paths)} images found')

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        angle = self.angles[index]
        img = Image.open(path)
        img = TF.rotate(img, angle, expand=True)
        return self.transform(img)

class ImagesDataset(Dataset):
    def __init__(self, folder, image_size, exts):
        super().__init__()
        self.folder = folder
        self.exts = exts

        self.paths = []
        self.labels = []
        for path in self.folder.glob('**/*'):
                    label, ext = os.path.splitext(path)
                    if ext.lower() in ['.png']:
                        self.paths.append(path)
                        self.labels.append(re.split("\\\\|/", label)[-2])

        for path in Path(f'{self.folder}').glob('**/*'):
            _, ext = os.path.splitext(path)
            if ext.lower() in self.exts:
                self.paths.append(path)

        print(f'{len(self.paths)} images found')

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]
        img = Image.open(path)
        img = img.convert('RGB')
        return self.transform(img), label