# LPIPS, PSNR, SSIM, DIST, ACC
# Please ensure consistency in the image formats, such as using JPG for CelebA and PNG for FFHQ
from my_utils import calculate_metrics
ori_dir = '/data/yangjiarui/project/datasets/CelebA/C1000'
rec_dir = '/data/yangjiarui/project/g_diff/k4'
metrics = calculate_metrics(ori_dir, rec_dir)
print(metrics)


# pytorch_fid is used to test
# fidelity  --input1 /data/yangjiarui/project/datasets/CelebA/C1000 --input2 /data/yangjiarui/project/g_diff/C10008x -g 3 -f
# python -m pytorch_fid /data/yangjiarui/project/datasets/LFWtest /data/yangjiarui/project/dps/results/super_resolution/recon