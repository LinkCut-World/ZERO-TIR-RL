HF_ENDPOINT=https://hf-mirror.com python -m areal.launcher.local train.py \
  --config train_config.yaml \
  experiment_name=tir \
  trial_name=debug \
  2>&1 | tee train.log \