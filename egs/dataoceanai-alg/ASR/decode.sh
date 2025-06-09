export CUDA_VISIBLE_DEVICES=$1
epoch=$2
avg=$3

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
