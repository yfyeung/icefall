project=icefall-asr-dataoceanai-alg-zipformer-2025-08-10
run=zipformer-medium_rnnt_4090_48G_md1000_lrepochs250_cs1
exp=zipformer/exp_ws1_md1000_lrepochs250_cs1
while true
do
  wandb sync $exp/tensorboard/  -p $project  --id $run  --append
  sleep 60
done
