#!/bin/bash
#SBATCH -p use-everything
#SBATCH --nodes=8
#SBATCH --gpus=32
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --job-name=swav_800ep_pretrain
#SBATCH --time=99:99:99
#SBATCH --mem=450G
#SBATCH --mail-user=rylansch
#SBATCH --mail-type=FAIL

master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
dist_url="tcp://"
dist_url+=$master_node
dist_url+=:40000

DATASET_PATH="/home/akhilan/om2/train/"
EXPERIMENT_PATH="./experiments/swav_800ep_pretrain"
mkdir -p $EXPERIMENT_PATH

source swav_venv/bin/activate

srun --output=${EXPERIMENT_PATH}/%j.out --error=${EXPERIMENT_PATH}/%j.err --label python -u main_swav.py \
--data_path $DATASET_PATH \
--nmb_crops 2 6 \
--size_crops 224 96 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--crops_for_assign 0 1 \
--temperature 0.1 \
--epsilon 0.05 \
--sinkhorn_iterations 3 \
--feat_dim 128 \
--nmb_prototypes 3000 \
--queue_length 0 \
--epochs 800 \
--batch_size 64 \
--base_lr 4.8 \
--final_lr 0.0048 \
--freeze_prototypes_niters 313 \
--wd 0.000001 \
--warmup_epochs 10 \
--start_warmup 0.3 \
--dist_url $dist_url \
--arch resnet50 \
--use_fp16 true \
--sync_bn apex \
--dump_path $EXPERIMENT_PATH
