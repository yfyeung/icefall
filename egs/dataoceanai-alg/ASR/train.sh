export CUDA_VISIBLE_DEVICES=2,3,4,5
export MASTER_PORT=16692

./zipformer/train.py \
  --world-size 4 \
  --num-epochs 200 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp_crctc_rnnt \
  --max-duration 600 \
  --lr-epochs 256 \
  --use-cr-ctc 1 \
  --use-ctc 1 \
  --use-transducer 1 \
  --use-attention-decoder 0 \
  --ctc-loss-scale 0.1 \
  --enable-spec-aug 0 \
  --cr-loss-scale 0.02 \
  --time-mask-ratio 2.5

python /scratch/busygpu/run.py
