export PYTHONPATH=/scratch/icefall:$PYTHONPATH

./zipformer/train.py \
  --world-size 8 \
  --num-epochs 60 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp \
  --max-duration 1000 \
  --lr-epochs 2.61

cd /scratch/azureml/cr/j/69293ec630e94c28ba2b2ee3f031e47c/exe/wd/busygpu && python run.py
