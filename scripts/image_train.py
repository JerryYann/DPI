"""
Train a diffusion model on images.
"""
import sys
sys.path.append("..")
import argparse
import torch
import torch.nn as nn
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.init_distributed_mode(args)
    rank = args.rank
    device = torch.device("cuda", args.local_rank)

    if rank == 0:
        logger.configure()
        logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # 3
    model = model.to(device)
    model = nn.DataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    if rank == 0:
        logger.log("creating data loader...")

    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        random_crop=False,
        l = True,
    )
    if rank == 0:
        logger.log("training...")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        rank = args.local_rank,
        device=device
    ).run_loop()


def create_argparser(): #
    defaults = dict(
        data_dir="/data/yangjiarui/project/datasets/lfw_funneled",
        schedule_sampler="uniform",
        lr=1e-5,
        weight_decay=0.0,
        lr_anneal_steps=500000,
        batch_size=4,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=1000,
        save_interval=5000,
        resume_checkpoint='',
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    # 1

    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    add_dict_to_argparser(parser, defaults)

    return parser

if __name__ == "__main__":
    main()
