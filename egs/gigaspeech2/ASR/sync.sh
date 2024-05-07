project=icefall-asr-gigaspeech2-zipformer-2024-04-28
run=bpe500_cs2_filtered0.2

wandb sync zipformer/exp/tensorboard/ --sync-tensorboard  -p $project  --id $run

while true
do
  wandb sync zipformer/exp/tensorboard/ --sync-tensorboard  -p $project  --id $run  --append
  sleep 60
done
