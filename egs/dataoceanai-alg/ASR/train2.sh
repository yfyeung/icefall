export CUDA_VISIBLE_DEVICES=1
export MASTER_PORT=16693

./zipformer/train.py \
  --world-size 1 \
  --num-epochs 300 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp_ws1_md1000_lrepochs250 \
  --max-duration 1000 \
  --lr-epochs 250

python /scratch/busygpu/run.py
