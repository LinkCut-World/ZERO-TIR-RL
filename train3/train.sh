HF_ENDPOINT=https://hf-mirror.com python -m areal.launcher.local train.py \
  --config train_config.yaml \
  experiment_name=tir_code \
  trial_name=trial_$(date +"%Y%m%d_%H%M%S") \
  2>&1 | tee train.log \