export CUDA_VISIBLE_DEVICES="1"
export MASTER_PORT=16694

./zipformer/train.py \
  --world-size 1 \
  --num-epochs 300 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp_ws1_md1000_lrepochs75_cs1 \
  --max-duration 1000 \
  --lr-epochs 75 \
  --context-size 1

python /root/busygpu/run.py
