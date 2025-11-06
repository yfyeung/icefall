#!/usr/bin/env bash

export PYTHONPATH=/data/workspace/icefall:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$1

# data related
use_librispeech=1
full_libri=0

causal=0
lr=0.045

# finetune checkpoint
do_finetune=1
finetune_ckpt=iter-448000-avg-2.pt

use_ctc=1
use_transducer=0
output_ds=2
post_output_ds=1

freeze_encoder=0
freeze_encoder_steps=-1
encoder_lr_scale=0.02222

md=1000

exp_dir=spear_roberta/exp_ft

echo $exp_dir

if false; then
torchrun --nproc_per_node=4 --master_port=19290 \
    spear_roberta/finetune_asr.py \
    --num-epochs 50 \
    --use-fp16 0 \
    --use-bf16 1 \
    --start-epoch 1 \
    --use-librispeech $use_librispeech --full-libri $full_libri \
    --exp-dir $exp_dir \
    --manifest-dir data/fbank \
    --lang-dir data/lang_char \
    --base-lr $lr \
    --use-ctc $use_ctc --use-transducer $use_transducer \
    --do-finetune $do_finetune --init-modules "encoder_embed,encoder" --finetune-ckpt $finetune_ckpt \
    --freeze-encoder $freeze_encoder --freeze-encoder-steps $freeze_encoder_steps \
    --encoder-lr-scale $encoder_lr_scale \
    --causal $causal \
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
    --max-duration $md
fi

if true; then
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

echo "Done"
