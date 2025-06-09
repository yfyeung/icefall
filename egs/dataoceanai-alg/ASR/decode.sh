export CUDA_VISIBLE_DEVICES=$1

for ((epoch=$2; epoch<=$3; epoch+=10)); do
  for ((avg=10; avg<=$epoch-10; avg+=10)); do
  ./zipformer/decode.py \
    --epoch $epoch \
    --avg $avg \
    --exp-dir zipformer/exp_crctc_rnnt \
    --use-cr-ctc 1 \
    --use-ctc 1 \
    --use-transducer 1 \
    --use-attention-decoder 0 \
    --max-duration 1000 \
    --decoding-method greedy_search
  done
done
