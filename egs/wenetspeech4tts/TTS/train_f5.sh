export PYTHONPATH=$PYTHONPATH:/scratch/icefall

exp_dir=exp/f5

python3 f5-tts/train.py --max-duration 100 --filter-min-duration 0.5 --filter-max-duration 20  \
      --num-buckets 6 --dtype "float16" --save-every-n 5000 --valid-interval 8000 \
      --base-lr 1e-4 --warmup-steps 5000 --average-period 200 \
      --num-epochs 10 --start-epoch 1 \
      --exp-dir ${exp_dir} --world-size 8
