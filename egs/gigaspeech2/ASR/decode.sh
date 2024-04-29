./zipformer/decode_with_bpe.py \
  --epoch 12 \
  --avg 2 \
  --exp-dir ./zipformer/exp \
  --max-duration 1000
  --decoding-method modified_beam_search \
  --beam-size 4
