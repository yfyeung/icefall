export CUDA_VISIBLE_DEVICES=1
export MASTER_PORT=16692

./zipformer/train.py \
  --world-size 1 \
  --num-epochs 300 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp_ws1_md500_lrepochs250_cs3 \
  --max-duration 500 \
  --lr-epochs 250 \
  --context-size 3

python /scratch/busygpu/run.py
