#!/usr/bin/env bash

export PYTHONPATH=/root/icefall:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

lr=0.045

# finetune checkpoint
do_finetune=1
finetune_ckpt=download/iter-448000-avg-2.pt

output_ds=2
post_output_ds=1

freeze_encoder=0
freeze_encoder_steps=-1
encoder_lr_scale=0.02222

md=800

exp_dir=spear_roberta/exp_ft

echo $exp_dir

if true; then
python spear_roberta/finetune.py \
    --world-size 8 \
    --num-epochs 100 \
    --use-fp16 0 \
    --use-bf16 1 \
    --start-epoch 1 \
    --exp-dir $exp_dir \
    --manifest-dir data/fbank \
    --base-lr $lr \
    --do-finetune $do_finetune --init-modules "encoder_embed,encoder" --finetune-ckpt $finetune_ckpt \
    --freeze-encoder $freeze_encoder --freeze-encoder-steps $freeze_encoder_steps \
    --encoder-lr-scale $encoder_lr_scale \
    --downsampling-factor 1,2,4,8,4,2,1 \
    --num-encoder-layers 1,2,3,4,1,1,1 \
    --feedforward-dim 3840,3840,3840,3840,3840,3840,3840 \
    --encoder-dim 1280,1280,1280,1280,1280,1280,1280 \
    --encoder-unmasked-dim 768,768,768,768,768,768,768 \
    --cnn-module-kernel 31,31,15,15,15,31,31 \
    --num-heads 8,8,8,8,8,8,8 \
    --output-downsampling-factor $output_ds \
    --post-encoder-downsampling-factor $post_output_ds \
    --on-the-fly-feats 1 \
    --enable-musan 0 \
    --enable-spec-aug 0 \
    --max-duration $md
fi

if false; then
epoch=999
avg=1
for m in ctc-decoding; do
    python spear_roberta/decode_ctc.py \
            --epoch $epoch \
            --avg $avg \
            --manifest-dir data/fbank \
            --lang-dir data/lang_char \
            --use-averaged-model 0 \
            --downsampling-factor 1,2,4,8,4,2,1 \
            --num-encoder-layers 1,2,3,4,1,1,1 \
            --feedforward-dim 3840,3840,3840,3840,3840,3840,3840 \
            --encoder-dim 1280,1280,1280,1280,1280,1280,1280 \
            --encoder-unmasked-dim 768,768,768,768,768,768,768 \
            --cnn-module-kernel 31,31,15,15,15,31,31 \
            --num-heads 8,8,8,8,8,8,8 \
            --output-downsampling-factor $output_ds \
            --post-encoder-downsampling-factor $post_output_ds \
            --on-the-fly-feats 1 \
            --exp-dir $exp_dir \
            --decoding-method $m \
            --max-duration 500
    done
fi
