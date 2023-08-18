import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import cv2
import my_utils as utils

class Dataset_YJR(data.Dataset):

    def __init__(self, args):
        super(Dataset_YJR, self).__init__()
        self.args = args
        self.image_size = args.image_size
        self.base_size = self.image_size // args.k
        self.scale = args.downsampling
        H_path = args.h_sets if args.h_sets else None
        test_set = H_path if H_path is not None else args.real_sets
        self.test_sets = utils.get_image_paths(test_set)


    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        H_path = self.test_sets[index]
        file_name, file_ext = os.path.split(H_path)
        img_num, _ = os.path.splitext(file_ext)

        img_H = utils.imread_uint(H_path, self.args.n_channels, resize=(self.image_size, self.image_size))
        img_L = utils.imread_uint(H_path, self.args.n_channels, resize=(self.base_size, self.base_size), scale=self.scale / self.args.k)


        img_L = img_L.astype(np.float32) / 127.5 - 1
        img_L = torch.from_numpy(img_L).permute(2, 0, 1).float()

        return {'img_H': img_H, 'img_L': img_L, 'name': img_num}  # 返回字典

    def __len__(self):  # 返回数据集的大小
        return len(self.test_sets)