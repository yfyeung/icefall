for ((epoch=30; epoch<=30; epoch++)); do
  for ((avg=27; avg<=27; avg++)); do
  ./zipformer/decode_with_bpe4.py \
    --epoch $epoch \
    --avg $avg \
    --exp-dir ./zipformer/exp \
    --max-duration 2000 \
    --decoding-method modified_beam_search \
    --beam-size 4
  done
done
