#!/usr/bin/env bash

export PYTHONPATH=/root/icefall:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=$1

# data related
use_librispeech=1
full_libri=0

causal=0
lr=0.045

# finetune checkpoint
do_finetune=1
finetune_ckpt=/private_data2/327M-uni-v2-batch-mix-0.3-p-n0.5/iter-500000-avg-4.pt

output_ds=1
post_output_ds=1

freeze_encoder=0
freeze_encoder_steps=-1
encoder_lr_scale=0.02222

md=1000

exp_dir=zipformer_finetune/exp_ft_ls100_char_ctc_ws4_md1000_lr1e-3_bf16

echo $exp_dir

if false; then
torchrun --nproc_per_node=4 --master_port=19290 \
  zipformer_finetune/finetune_asr.py \
    --num-epochs 200 \
    --use-fp16 0 \
    --use-bf16 1 \
    --start-epoch 1 \
    --use-librispeech $use_librispeech --full-libri $full_libri \
    --exp-dir $exp_dir \
    --manifest-dir data/fbank \
    --lang-dir data/lang_char \
    --base-lr $lr \
    --do-finetune $do_finetune --init-modules "encoder_embed,encoder" --finetune-ckpt $finetune_ckpt \
    --freeze-encoder $freeze_encoder --freeze-encoder-steps $freeze_encoder_steps \
    --encoder-lr-scale $encoder_lr_scale \
    --causal $causal \
    --downsampling-factor 1,2,4,8,4,2,1 \
    --num-encoder-layers 1,2,2,3,1,1,1 \
    --feedforward-dim 3072,3072,3072,3072,3072,3072,3072 \
    --encoder-dim 1024,1024,1024,1024,1024,1024,1024 \
    --encoder-unmasked-dim 512,512,512,512,512,512,512 \
    --cnn-module-kernel 31,31,15,15,15,31,31 \
    --num-heads 8,8,8,8,8,8,8 \
    --output-downsampling-factor $output_ds \
    --post-encoder-downsampling-factor $post_output_ds \
    --on-the-fly-feats 1 \
    --max-duration $md
fi

if true; then
for m in ctc-decoding; do
    # for epoch in $(seq 100 -5 40); do
    for epoch in $(seq 70 -1 40); do
        for avg in $(seq $((epoch-10)) -1 30); do
        # for avg in $(seq $((epoch-1)) -5 5); do
            python zipformer_finetune/decode_ctc.py \
                --epoch $epoch \
                --avg $avg \
                --manifest-dir data/fbank \
                --lang-dir data/lang_char \
                --use-averaged-model 1 \
                --downsampling-factor 1,2,4,8,4,2,1 \
                --num-encoder-layers 1,2,2,3,1,1,1 \
                --feedforward-dim 3072,3072,3072,3072,3072,3072,3072 \
                --encoder-dim 1024,1024,1024,1024,1024,1024,1024 \
                --encoder-unmasked-dim 512,512,512,512,512,512,512 \
                --cnn-module-kernel 31,31,15,15,15,31,31 \
                --num-heads 8,8,8,8,8,8,8 \
                --output-downsampling-factor $output_ds \
                --post-encoder-downsampling-factor $post_output_ds \
                --on-the-fly-feats 1 \
                --exp-dir $exp_dir \
                --decoding-method $m \
                --max-duration 2000
        done
    done
done
fi

python ~/busygpu/run.py &
