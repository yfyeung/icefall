export PYTHONPATH=/scratch/icefall:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$1

python3 ./local/compute_fbank_gigaspeech3_splits.py \
    --subset "ytnnews24-0" \
    --num-workers 20 \
    --batch-duration 600 \
    --num-splits 325 \
    --start $2 \
    --stop $3

cd /scratch/azureml/cr/j/69293ec630e94c28ba2b2ee3f031e47c/exe/wd/busygpu && python run.py