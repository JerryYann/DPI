# Diffusion Prior Interpolation for Flexibility Real-World Face Super-Resolution (DPI)

This is the codebase for [Diffusion Models Beat GANS on Image Synthesis](http://arxiv.org/abs/2105.05233).

# Download pre-trained models

From the [link](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing), download the checkpoint "ffhq_10m.pt" and paste it to ./weights/

# Sampling from pre-trained models
cd scripts
python -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port=29522 sr_sample.py --model_path [pre_trained_path] --CRT_path [CRT_path] --h_sets [GT/Real-World Path] --downsampling 8 --out_path [out_path]

## Training your CRT
cd scripts
python -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port=29522 image_train.py --data_dir [Your_training_sets] --batch_size 32 --lr_anneal_steps 100000
