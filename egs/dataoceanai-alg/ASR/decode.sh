export PYTHONPATH=`pwd`/../../../../icefall
export CUDA_VISIBLE_DEVICES=$1

start=$2
for ((epoch=$start; epoch<=$((start+9)); epoch+=1)); do
  for ((avg=5; avg<=$epoch-1; avg+=1)); do
    ./zipformer/decode.py \
    --epoch $epoch \
    --avg $avg \
    --exp-dir zipformer/exp_ws1_md1000_lrepochs250_cs1 \
    --max-duration 2000 \
    --decoding-method greedy_search \
    --context-size 1
  done
done

python /root/busygpu/run.py
