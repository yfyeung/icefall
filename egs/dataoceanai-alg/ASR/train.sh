./zipformer/train.py \
  --world-size 4 \
  --num-epochs 200 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp \
  --max-duration 1000 \
  --lr-epochs 256 \
  --use-transducer 0 \
  --use-ctc 1

python /scratch/busygpu/run.py
