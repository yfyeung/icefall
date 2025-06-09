export CUDA_VISIBLE_DEVICES=$1

for ((epoch=100; epoch<=300; epoch+=10)); do
  for ((avg=10; avg<=$epoch-10; avg+=10)); do
  ./zipformer/decode.py \
    --epoch $epoch \
    --avg $avg \
    --exp-dir zipformer/exp_ws1_md1000_lrepochs200 \
    --max-duration 2000 \
    --decoding-method greedy_search
  done
done

python /scratch/busygpu/run.py
