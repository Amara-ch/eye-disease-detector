import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Confident diseases (>= 50 samples)
CONFIDENT_DISEASES = ['DR', 'MH', 'ODC', 'TSLN', 'DN',
                      'MYA', 'ARMD', 'BRVO', 'ODP', 'ODE']


def build_transforms(img_size=384, train=True):
    if train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=20, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class RFMiDDataset(Dataset):
    '''Multi-task dataset: returns image, binary label, and disease labels.'''

    def __init__(self, df, img_dir, img_size=384, train=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = build_transforms(img_size, train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f'{self.img_dir}/{int(row[\"ID\"])}.png'

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)['image']

        binary_label = torch.tensor(row['Disease_Risk'], dtype=torch.float32)
        disease_labels = torch.tensor(
            row[CONFIDENT_DISEASES].values.astype(np.float32)
        )
        return img, binary_label, disease_labels
