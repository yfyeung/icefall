export CUDA_VISIBLE_DEVICES="0,1"
export MASTER_PORT=16693

./zipformer/train.py \
  --world-size 2 \
  --num-epochs 300 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp_ws2_md1000_lrepochs250_cs1 \
  --max-duration 1000 \
  --lr-epochs 250 \
  --context-size 1

python /root/busygpu/run.py
