import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import my_utils
import cv2


class Dataset_YJR(data.Dataset):

    def __init__(self, path, degrade_type, img_size):
        super(Dataset_YJR, self).__init__()
        print('Dataset: Denosing on AWGN with fixed sigma')
        self.n_channels = 3
        self.paths_H = utils.get_image_paths(path)
        self.type = degrade_type
        self.patch_size = img_size

    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]

        img_H = utils.imread_uint(H_path)  # 0-255

        H, W, _ = img_H.shape
        # --------------------------------
        # randomly crop the patch
        # --------------------------------
        rnd_h = random.randint(0, max(0, H - self.patch_size))
        rnd_w = random.randint(0, max(0, W - self.patch_size))
        patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]  # 0-255

        mode = np.random.randint(0, 8)
        patch_H = utils.augment_img(patch_H, mode=mode)
        img_L = np.copy(patch_H)  # 0-255
        img_L = utils.add_noise(img_L, 50, self.type)
        patch_H = patch_H / 127.5 - 1
        img_L = img_L / 127.5 - 1

        patch_H = torch.from_numpy(patch_H).permute(2, 0, 1).unsqueeze(0).float()

        img_L = torch.from_numpy(img_L).permute(2, 0, 1).unsqueeze(0).float()


        return {'H': patch_H, 'L': img_L}  # 返回字典

    def __len__(self):  # 返回数据集的大小
        return len(self.paths_H)