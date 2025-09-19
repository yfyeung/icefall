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
finetune_ckpt=/private_data2/600M-uni-dual-domain-mvq/iter-448000-avg-2.pt

use_ctc=0
use_transducer=1
output_ds=2
post_output_ds=1

freeze_encoder=0
freeze_encoder_steps=-1
encoder_lr_scale=0.044444

md=1000

exp_dir=zipformer_finetune/exp_ft_ls100_bpe_rnnt_ws4_md1000_lr2e-3_bf16

echo $exp_dir

if false; then
torchrun --nproc_per_node=4 --master_port=19291 \
  zipformer_finetune/finetune_asr.py \
    --num-epochs 100 \
    --use-fp16 0 \
    --use-bf16 1 \
    --start-epoch 1 \
    --use-librispeech $use_librispeech --full-libri $full_libri \
    --exp-dir $exp_dir \
    --manifest-dir data/fbank \
    --bpe-model data/lang_bpe_500/bpe.model \
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

for m in greedy_search; do
    for epoch in $(seq 90 -5 40); do
        for avg in $(seq $((epoch-1)) -5 5); do
            python zipformer_finetune/decode.py \
                --epoch $epoch \
                --avg $avg \
                --manifest-dir data/fbank \
                --bpe-model data/lang_bpe_500/bpe.model \
                --use-averaged-model 1 \
                --downsampling-factor 1,2,4,8,4,2,1 \
                --num-encoder-layers 1,2,3,4,1,1,1 \
                --feedforward-dim 3840,3840,3840,3840,3840,3840,3840 \
                --encoder-dim 1280,1280,1280,1280,1280,1280,1280 \
                --encoder-unmasked-dim 768,768,768,768,768,768,768 \
                --cnn-module-kernel 31,31,15,15,15,31,31 \
                --num-heads 8,8,8,8,8,8,8 \
                --use-ctc $use_ctc --use-transducer $use_transducer \
                --output-downsampling-factor $output_ds \
                --post-encoder-downsampling-factor $post_output_ds \
                --on-the-fly-feats 1 \
                --exp-dir $exp_dir \
                --decoding-method $m \
                --max-duration 2000
        done
    done
done

echo "Done"

python ~/busygpu/run.py &
