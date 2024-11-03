project=icefall-asr-gigaspeech3-zipformer-2024-11-02
run=ytnnews24-0-iter1
exp=zipformer/exp

while true
do
  wandb sync $exp/tensorboard/ --sync-tensorboard  -p $project  --id $run  --append
  sleep 60
done
