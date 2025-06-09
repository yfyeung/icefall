project=icefall-asr-dataoceanai-alg-zipformer-2025-06-09
run=zipformer-medium_crctc_rnnt_4V10032G_md600_lrepochs256_ctcscale0.1_crscale_0.02_timemaskratio2.5
exp=zipformer/exp_crctc_rnnt
while true
do
  wandb sync $exp/tensorboard/ --sync-tensorboard  -p $project  --id $run  --append
  sleep 60
done
