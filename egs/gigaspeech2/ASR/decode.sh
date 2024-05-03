for ((epoch=12; epoch<=12; epoch++)); do
  for ((avg=2; avg<=$epoch-1; avg++)); do
  ./zipformer/decode_with_bpe.py \
    --epoch $epoch \
    --avg $avg \
    --exp-dir ./zipformer/exp \
    --max-duration 2000
    --decoding-method modified_beam_search \
    --beam-size 4
  done
done
