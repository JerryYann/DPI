import os

import PIL.ImageEnhance
import math
import random
import numpy as np
import torch
from PIL import Image, ImageFilter
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".JPEG", ".bmp", ".tif", ".tiff", ".dng", "PNG"])

def get_image_paths(path):
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)

    return sorted(images)

def imread_uint(path, n_channels=3, resize=None, scale = None):
    # open image with PIL
    image = Image.open(path)
    if n_channels == 1:
        # convert image to grayscale mode
        image = image.convert("L")
        # add a new dimension for channel
        image = Image.merge('RGB', (image, image, image))
    else:
        # convert image to RGB mode
        image = image.convert("RGB")

    if resize:
        if scale is not None:
            lr_size = tuple(int(x // scale) for x in resize)
            image = image.resize(lr_size, Image.BICUBIC)
        image = image.resize(resize, Image.BICUBIC)

    return np.array(image)

def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def calculate_metrics(original_dir, reconstructed_dir):
    import lpips
    from deepface import DeepFace
    device = 'cuda:2'
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    lpips_list = []
    psnr_list = []
    ssim_list = []

    for idx in tqdm(range(1, 1001)):
        fname = str(idx).zfill(5)
        original_path = Path(original_dir) / f'{fname}.jpg'
        reconstructed_path = Path(reconstructed_dir) / f'{fname}.jpg'
        orig_img  = imread_uint(original_path)
        rec_img = imread_uint(reconstructed_path)
        original_img = torch.from_numpy(orig_img).permute(2, 0, 1).to(device)
        reconstructed_img = torch.from_numpy(rec_img).permute(2, 0, 1).to(device)
        original_img = original_img.view(1, 3, 256, 256) * 2. - 1.
        reconstructed_img = reconstructed_img.view(1, 3, 256, 256) * 2. -1.

        lpips = loss_fn_vgg(reconstructed_img, original_img)
        psnr = calculate_psnr(orig_img,rec_img)
        ssim = calculate_ssim(orig_img,rec_img)
        lpips_list.append(lpips.item())
        psnr_list.append(psnr)
        ssim_list.append(ssim)

    lpips_avg = np.mean(lpips_list)
    psnr_avg = np.mean(psnr_list)
    ssim_avg = np.mean(ssim_list)

    return lpips_avg, psnr_avg, ssim_avg
