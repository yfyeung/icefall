#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
echo "Num gpus: $GPU_COUNT"

echo "Current Directory: $PWD"

export PYTHONPATH=./../../../:$PYTHONPATH

# the dasheng model version, base/medium/large
model_version=large
full_libri=1

# full finetune
freeze_encoder=0
set_eval=0

torchrun --nproc_per_node=1 --master_port=19293 \
dasheng/finetune_asr.py \
        --world-size 8 \
        --num-epochs 50 \
        --use-fp16 1 \
        --start-epoch 1 \
        --full-libri $full_libri \
        --model-version $model_version \
        --exp-dir dasheng/exp-finetune-asr-full-libri-${full_libri}-${model_version} \
        --max-duration 100

for i in {0..7}; do CUDA_VISIBLE_DEVICES=$i python /root/busygpu/run.py & done
