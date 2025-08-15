export CUDA_VISIBLE_DEVICES=1

for ((epoch=20; epoch<=150; epoch+=5)); do
  for ((avg=5; avg<=$epoch-10; avg+=10)); do
  ./zipformer/decode.py \
    --epoch $epoch \
    --avg $avg \
    --exp-dir zipformer/exp_ws1_md1000_lrepochs100_cs1 \
    --max-duration 2000 \
    --decoding-method greedy_search \
    --context-size 1
  done
done
