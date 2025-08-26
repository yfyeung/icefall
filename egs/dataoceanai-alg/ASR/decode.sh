export CUDA_VISIBLE_DEVICES=$1


for ((epoch=30; epoch<=40; epoch+=1)); do
  for ((avg=5; avg<=15; avg+=1)); do
    ./zipformer/decode.py \
    --epoch $epoch \
    --avg $avg \
    --exp-dir zipformer/exp_ws1_md1000_lrepochs100_cs1 \
    --max-duration 2000 \
    --decoding-method greedy_search \
    --context-size 1
  done
done
