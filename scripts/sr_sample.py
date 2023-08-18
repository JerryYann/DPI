import sys
sys.path.append("..")
import argparse
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn as nn
from guided_diffusion import dist_util, logger, test_datasets
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from collections import OrderedDict
from PIL import Image
import my_utils as utils
from torch.utils.data import DataLoader

def main():
    args = create_argparser().parse_args()
    dist_util.init_distributed_mode(args)
    device = th.device("cuda", args.local_rank)

    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model = model.to(device)
    from guided_diffusion.script_util import create_model
    CRT = create_model(
        image_size=256,
        num_channels=32,
        num_res_blocks=1,
        learn_sigma=False,
        class_cond=False,
        use_checkpoint=False,
        attention_resolutions='8',
        num_heads=2,
        num_head_channels=16,
        num_heads_upsample=1,
        use_scale_shift_norm=True,
        dropout=0.0,
        resblock_updown=True,
        use_fp16=False,
        use_new_attention_order=False,
    )
    CRT = nn.DataParallel(CRT, device_ids=[args.local_rank], output_device=args.local_rank)
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    CRT.load_state_dict(
        dist_util.load_state_dict(args.CRT_path, map_location="cpu")
    )

    if args.use_fp16:
        model.convert_to_fp16()

    model.eval()
    CRT.eval()

    test_data = test_datasets.Dataset_YJR(args)
    test_loader = DataLoader(test_data, batch_size=args.batch_size,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)



    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    num_sample = 1
    idx = 0
    out_path = args.out_path
    for test_data in enumerate(test_loader):
        count = 0

        logger.log("sampling...")

        img_H = test_data[1]['img_H']
        img_L = test_data[1]['img_L']
        img_L = img_L.to(device)
        name_list = test_data[1]['name']
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        mean_sample = 0

        while count < num_sample:

            sample = sample_fn(
                model=model,
                CRT=CRT,
                shape=(args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                device=device,
                sr = img_L,
                scale_factor = args.k,
            )
            mean_sample += sample
            count += args.batch_size
            print(count)
        mean_sample = mean_sample / num_sample
        sample = (((mean_sample + 1) * 127.5)).clamp(0, 255)

        for b in range(args.batch_size):
            tmp = sample[b].permute(1, 2, 0).cpu().numpy().astype(np.uint8)

            if args.h_sets != args.real_sets:
                psnr = utils.calculate_psnr(tmp, img_H[b].numpy(), border=0)
                ssim = utils.calculate_ssim(tmp, img_H[b].numpy(), border=0)
                test_results['psnr'].append(psnr)
                test_results['ssim'].append(ssim)
                logger.log('{} - PSNR: {:.2f} dB SSIM: {:.4f}'.format(idx, psnr, ssim))

            tmp = Image.fromarray(tmp)
            tmp.save(out_path + '{}.jpg'.format(name_list[b]))
            idx += 1
            logger.log("Complete {}".format(idx))
        if idx == 1000:
            break

    dist.barrier()
    logger.log('PSNR: {:.2f} dB SSIM: {:.4f}'.format(np.mean(test_results['psnr']), np.mean(test_results['ssim'])))
    logger.log("sampling complete")
    dist.destroy_process_group()

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=True,
        model_path="/data/yangjiarui/project/g_diff/weights//ffhq_10m.pt",
        CRT_path="/data/yangjiarui/project/g_diff/weights/Corrector8.pt",
        h_sets = '/data/yangjiarui/project/datasets/111', # F8x2 / F16x2
        real_sets = '',
        n_channels = 3,
        use_fp16=False,
        downsampling = 8,
        k = 2,
        out_path = '../test/'
    )

    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
