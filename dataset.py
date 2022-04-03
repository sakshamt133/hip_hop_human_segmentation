from torch.utils.data import Dataset
import utils
import numpy as np
import cv2 as cv
import os


class HipHop(Dataset):
    def __init__(self, img_path, mask_path, transform):
        super(HipHop, self).__init__()
        self.img_path = img_path
        self.transform = transform
        self.mask_path = mask_path
        self.images = os.listdir(img_path)
        self.mask = os.listdir(mask_path)

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, item):
        img_name = self.images[item]
        img_path = os.path.join(self.img_path, img_name)
        mask_path = os.path.join(self.mask_path, img_name)
        mask = cv.imread(mask_path)
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (utils.img_size, utils.img_size))
        mask = cv.resize(mask, (utils.img_size, utils.img_size))
        img = self.transform(np.asarray(img))
        mask = self.transform(np.asarray(mask))
        return img, mask
