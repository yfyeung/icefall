project=icefall-asr-dataoceanai-alg-zipformer-2025-06-09

while true; do
  for exp_dir in zipformer/exp*; do
    [ -d "$exp_dir" ] || continue
    if [[ "$exp_dir" =~ md[0-9]+_lrepochs[0-9]+_cs[0-9] ]]; then
      suffix=${BASH_REMATCH[0]}
      run="zipformer-medium_rnnt_1V10032G_${suffix}"
      wandb sync "$exp_dir/tensorboard/" --sync-tensorboard -p "$project" --id "$run" --append
    fi
  done
  sleep 60
done

