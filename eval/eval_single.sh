HF_ENDPOINT=https://hf-mirror.com python -m areal.launcher.local eval_single.py \
  --config eval_config.yaml \
  experiment_name=tir \
  trial_name=eval \
  sglang.model_path=/home/liangchengwei/lcw/ZERO-TIR-RL/experiments/checkpoints/liangchengwei/tir/debug/default/epoch0epochstep5667globalstep5667 \
  +checkpoint_name=test \
  +statistic_path=eval_statistic.pkl \
  2>&1 | tee eval.log \