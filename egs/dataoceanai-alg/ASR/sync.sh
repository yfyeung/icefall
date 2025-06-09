project=icefall-asr-dataoceanai-alg-zipformer-2025-06-09
run=zipformer-medium_ctc_2V10032G_md1000_lrepochs256
exp=zipformer/exp_ctc
while true
do
  wandb sync $exp/tensorboard/ --sync-tensorboard  -p $project  --id $run  --append
  sleep 60
done
