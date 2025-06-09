export CUDA_VISIBLE_DEVICES=3
export MASTER_PORT=16695

./zipformer/train.py \
  --world-size 1 \
  --num-epochs 300 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp_ws1_md1000_lrepochs200 \
  --max-duration 1000 \
  --lr-epochs 200

python /scratch/busygpu/run.py
