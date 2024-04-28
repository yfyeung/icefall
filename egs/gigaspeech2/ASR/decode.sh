./zipformer/decode.py \
  --epoch 30 \
  --avg 28 \
  --exp-dir ./zipformer/exp \
  --max-duration 2000 \
  --decoding-method modified_beam_search \
  --beam-size 4
