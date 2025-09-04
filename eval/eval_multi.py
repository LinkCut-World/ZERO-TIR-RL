import os
import re
import logging
import subprocess
import signal
import sys
import pickle

def get_checkpoints():
    checkpoint_root = "/home/liangchengwei/lcw/ZERO-TIR-RL/experiments/checkpoints/liangchengwei/tir/debug/default/"
    checkpoint_s = []
    for name in os.listdir(checkpoint_root):
        path = os.path.join(checkpoint_root, name)
        if os.path.isdir(path):
            match = re.search(r'globalstep(\d+)', name)
            step = int(match.group(1)) if match else None
            checkpoint_s.append({
                "name": name,
                "path": path,
                "step": step,
            })
    key_steps = [599, 1999, 3999]
    checkpoint_s = [checkpoint for checkpoint in checkpoint_s if checkpoint["step"] in key_steps]
    checkpoint_s.append({"name": "step0", "path": "Qwen/Qwen2.5-1.5B", "step": 0})
    checkpoint_s = sorted(checkpoint_s, key=lambda x: x["step"])
    return checkpoint_s

def get_statistic_path():
    return "eval_statistic.pkl"

COMMAND_TEMPLATE = """
HF_ENDPOINT=https://hf-mirror.com WORLD_SIZE=2 python -m areal.launcher.local eval_single.py \
    --config eval_config.yaml \
    experiment_name=tir \
    trial_name=eval \
    sglang.model_path={checkpoint_path} \
    +checkpoint_name={checkpoint_name} \
    +statistic_path={statistic_path} \
"""

def main():
    logging.basicConfig(level=logging.INFO)
    checkpoints = get_checkpoints()
    statistic_path = get_statistic_path()
    logging.info(f"Checkpoints to evaluate: {[c['name'] for c in checkpoints]}")
    logging.info(f"Statistic path: {statistic_path}")

    with open(statistic_path, "wb") as f:
        pickle.dump(None, f)

    for checkpoint in checkpoints:
        logging.info(f"Evaluating checkpoint: { checkpoint["name"]}")
        command = COMMAND_TEMPLATE.format(
            checkpoint_path=checkpoint["path"],
            checkpoint_name=checkpoint["name"],
            statistic_path=statistic_path,
        )
        # command = "python -u tmp.py"
        logging.info(f"Running command: {command}")

        proc = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)
        try:
            proc.wait()
        except KeyboardInterrupt:
            print("KeyboardInterrupt received — terminating child and exiting")
            try:
                os.killpg(proc.pid, signal.SIGINT)
            except Exception:
                pass
            # Give it a moment to exit cleanly, then force terminate
            try:
                proc.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except Exception:
                    pass
            sys.exit(1)


if __name__ == "__main__":
    main()