import os
import swanlab

import asyncio
import os
import sys
import uuid
import json
import gc
import torch
import torch.distributed as dist
import numpy as np
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerFast

from dataclasses import dataclass, field

from areal.api.cli_args import (
    GenerationHyperparameters,
    GRPOConfig,
    load_expr_config,
)
from areal.api.io_struct import (
    FinetuneSpec,
    ModelRequest,
    WeightUpdateMeta,
)
from areal.api.workflow_api import RolloutWorkflow
from areal.api.cli_args import GRPOConfig
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.utils.data import concat_padded_tensors
from areal.utils.device import log_gpu_stats
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from realhf.api.core.data_api import load_hf_tokenizer
from realhf.base import logging, seeding, stats_tracker

logger = logging.getLogger("TIR")

@dataclass
class AgentRLConfig(GRPOConfig):
    max_tokens_per_traj: int = field(
        default=32000,
        metadata={
            "help": "maximum number of tokens per trajectory"
        }
    )

    max_turns: int = field(
        default=128,
        metadata={
            "help": "maximum number of turns for search agent"
        }
    )
    n_trajs: int = field(
        default=1,
        metadata={
            "help": "We could collect multiple trajectories for a single query. By default n_trajs=1."
        }
    )
    code_execution_timeout: int = field(
        default=10,
        metadata={
            "help": "timeout (in seconds) for code execution"
        }
    )

    verbose: bool = field(
        default=True,
        metadata={
            "help": "whether to print verbose information"
        }
    )
    recover_start_step: int = field(
        default=0,
        metadata={
            "help": "step to start recovering from, useful for resuming training"
        }
    )

import json

from itertools import cycle

from concurrent.futures import ProcessPoolExecutor


from realhf.impl.dataset.math_parser import extract_answer, math_equal

rw_executor = ProcessPoolExecutor(max_workers=4)
REWARD_TIMEOUT_SECONDS = 15

def reward_fn(generated, answer):
    try:
        x = extract_answer(generated, "math", use_last_number=True)
        y = extract_answer(answer, "math", use_last_number=True)

        if x is None or x.strip() in ["None", "none", ""]:
            return 0.0
        elif y is None or y.strip() in ["None", "none", ""]:
            return 0.0
        return float(math_equal(x, y, timeout=False))
    except:
        return 0.0


import asyncio
import functools
import os
import time
import uuid

import colorama
import torch
from tensordict import TensorDict
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import (
    AllocationMode,
    FinetuneSpec,
    ModelRequest,
    WeightUpdateMeta,
)
from areal.api.workflow_api import RolloutWorkflow
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.utils.data import concat_padded_tensors
from areal.utils.device import log_gpu_stats

import re


# PROMPT_TEMPLATE = """
# A conversation between User and Assistant. 
# The User asks a question, and the Assistant solves it. 
# The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. 
# In your reasoning-process, You can use python-code to solve your problem. Put the code within ```python and ``` tags, and the code will be executed immediately and output will be returned.
# User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. 
# And your final answer will be extracted automatically by the \\boxed{{}} tag.\nThis is the problem:{query}\nAssistant: <think>
# """

# def get_prompt(tokenizer, query):
#     return PROMPT_TEMPLATE.format(query=query)

SYSTEM_PROMPT = """
You are a helpful assistant. The User asks a question, and the Assistant solves it. 
The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{{}} tag.
In your reasoning-process, You can use python-code to solve your problem. Put the code within ```python and ``` tags. The script will be executed immediately and output will be returned.
"""

def get_prompt(tokenizer, query):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text += "<think>\n"
    return text

import re
import threading

from code_executor import execute_code

import shutil
import pickle
from datasets import Dataset as HFDataset

class TIRWorkflow(RolloutWorkflow):
    def __init__(
        self, 
        config: AgentRLConfig, 
        tokenizer: PreTrainedTokenizerFast,
    ):
        self.config = config
        self.gconfig = config.gconfig
        self.gconfig.n_samples = 1
        self.tokenizer = tokenizer
        self.current_trajs = 0

        self.example_trajs_path = "example_trajs.log"
        with open(self.example_trajs_path, "w") as log_file:
            log_file.write("")

    async def collect_agent_trajectory(self, qid, prompt, answer, engine):
        traj_rid = uuid.uuid4().hex
        loop = asyncio.get_event_loop()
        result = None
        reward = 0.0

        num_turns = 0
        input_str = prompt
        input_ids = self.tokenizer.encode(input_str, add_special_tokens=False)
        logprobs = [0.0] * len(input_ids)
        loss_mask = [0] * len(input_ids)
        stops = ["```python", "</answer>"]
        total_gen_time = 0
        total_exec_time = 0
        start_time = time.time()
        while num_turns < self.config.max_turns:
            req = ModelRequest(
                rid=traj_rid,
                input_ids=input_ids,
                gconfig=self.gconfig.new(n_samples=1),
            )
            req.gconfig.stop = stops
            if len(input_ids) + self.gconfig.max_new_tokens >= self.config.max_tokens_per_traj:
                break
            
            gen_start_time = time.time()
            resp = await engine.agenerate(req)
            gen_time = time.time() - gen_start_time
            total_gen_time += gen_time
            completion_str = self.tokenizer.decode(resp.output_tokens)

            input_str += completion_str
            input_ids += resp.output_tokens
            logprobs += resp.output_logprobs
            loss_mask += [1] * len(resp.output_tokens)

            if "</answer>" in completion_str:
                matches = re.findall(r"<answer>(.*?)</answer>", completion_str, re.DOTALL)
                if matches:
                    result = matches[-1]
                    reward = await loop.run_in_executor(
                        rw_executor,
                        functools.partial(reward_fn, result, answer)
                    )
                    break
            elif stops[0] == "```python" and "```python" in completion_str:
                stops[0] = "```"
            elif stops[0] == "```" and "```" in completion_str:
                matches = re.findall(r'```python(.*?)```', input_str, re.DOTALL)
                if matches:
                    code = matches[-1]
                    exec_start_time = time.time()
                    execution_output = await loop.run_in_executor(None, functools.partial(execute_code, code, self.config.code_execution_timeout))

                    exec_time = time.time() - exec_start_time
                    total_exec_time += exec_time
                    
                    num_turns += 1
                    
                    empty_output = execution_output.strip() == ""
                    execution_output = "\n```output\n" + execution_output + "\n```\n"
                    if empty_output:
                        execution_output += "No output. Let me use `print()` to see the result and try again.\n"
                    input_str += execution_output
                    exec_tokens = self.tokenizer.encode(execution_output, add_special_tokens=False)
                    if len(input_ids) + len(exec_tokens) >= self.config.max_tokens_per_traj:
                        exec_tokens = exec_tokens[:self.config.max_tokens_per_traj - len(input_ids) - 1]
                    input_ids += exec_tokens
                    logprobs += [0.0] * len(exec_tokens)
                    loss_mask += [0] * len(exec_tokens)
                stops[0] = "```python"
            
            if resp.output_tokens[-1] in [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]:
                break

        total_time = time.time() - start_time

        if len(input_ids) > self.config.max_tokens_per_traj:
            assert False, f"Trajectory {traj_rid} exceeds max tokens {self.config.max_tokens_per_traj} with {len(input_ids)} tokens."
        
        res = dict(
            input_ids=torch.tensor(input_ids),
            logprobs=torch.tensor(logprobs),
            loss_mask=torch.tensor(loss_mask, dtype=torch.bool),
            rewards=torch.tensor(float(reward)),
            score=torch.tensor(float(reward)),
            code_reward=torch.tensor(float(num_turns>0)),
            code_in_correct=torch.tensor(float(num_turns>0 and reward>0)),
            attention_mask=torch.ones(len(input_ids), dtype=torch.bool),
        )

        res_dump = {k: v.tolist() for k, v in res.items() if k != 'attention_mask' and k != 'rewards'}
        res_dump['input_str'] = input_str
        res_dump['metadata'] = {
            "reward": reward,
            "traj_rid": traj_rid,
            "num_turns": num_turns,
            "length": len(input_ids),
            "total_time": f"{total_time:.2f}s",
            "gen_time_ratio": f"{total_gen_time / total_time:.2f}" if total_time > 0 else "0.00",
            "exec_time_ratio": f"{total_exec_time / total_time:.2f}" if total_time > 0 else "0.00",
            "answer": answer,
            "result": result,
        }

        if self.current_trajs % 100 == 0:
            with open(self.example_trajs_path, "a") as log_file:
                log_file.write(res_dump['input_str'].__str__() + "\n" + res_dump['metadata'].__str__() + "\n")
        self.current_trajs += 1

        res = {k: v.unsqueeze(0) for k, v in res.items()}
        return TensorDict(res, batch_size=[1])

    async def arun_episode(self, engine, data):
        qid = uuid.uuid4().hex

        # prompt = PROMPT_TEMPLATE.format(query=data["question"])
        prompt = get_prompt(self.tokenizer, data["question"])

        trajs = await asyncio.gather(*[
            self.collect_agent_trajectory(qid, prompt, data["answer"], engine)
            for _ in range(self.config.n_trajs)
        ])

        avg_reward = sum([t["rewards"].item() for t in trajs]) / len(trajs)
        if avg_reward == 0:
            return None
        std = (sum([(t["rewards"].item() - avg_reward) ** 2 for t in trajs]) / len(trajs)) ** 0.5
        for traj in trajs:
            traj["rewards"] = (traj["rewards"] - avg_reward) / (std + 1e-6)
        
        return concat_padded_tensors(trajs)

def main(args):
    swanlab.sync_wandb()
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"

    config, _ = load_expr_config(args, AgentRLConfig)
    config: AgentRLConfig

    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")

    with open('../orz_math_57k_collected.json', 'r') as f:
        raw_data = json.load(f)
    def process_raw_data(item):
        return {
            "question": item[0]['value'],
            "answer": item[1]['ground_truth']['value'],
        }
    dataset = [process_raw_data(item) for item in raw_data]
    hf_dataset = HFDataset.from_list(dataset)
    dataset = split_dataset_by_node(hf_dataset, rank=rank, world_size=world_size)

    worker_batch_size = config.train_dataset.batch_size // world_size
    dataloader = StatefulDataLoader(
        dataset,
        batch_size=config.train_dataset.batch_size // world_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=config.train_dataset.num_workers,
        collate_fn=lambda x: x,
        drop_last=config.train_dataset.drop_last,
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(None, None)

    actor = FSDPPPOActor(config=config.actor)
    actor.initialize(None, ft_spec)
    ref = None

    weight_update_meta = [WeightUpdateMeta.from_disk(
        experiment_name=config.saver.experiment_name,
        trial_name=config.saver.trial_name,
        file_root=config.saver.fileroot,
    )]
    dist.broadcast_object_list(weight_update_meta, src=0)
    weight_update_meta = weight_update_meta[0]

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)

    workflow = TIRWorkflow(
        config=config,
        tokenizer=tokenizer,
    )

    saver = Saver(config.saver, ft_spec)
    stat_logger = StatsLogger(config.stats_logger, ft_spec)

    total_epochs = config.total_train_epochs
    steps_per_epoch = len(dataloader)
    max_steps = total_epochs * steps_per_epoch

    data_generator = iter(dataloader)
    start_step = config.recover_start_step or 0
    log_debug_batch = False
    for global_step in range(start_step, max_steps):
        epoch = global_step // steps_per_epoch
        step = global_step % steps_per_epoch
        logger.info(f"Epoch {epoch}. Step: {step}/{steps_per_epoch}")

        with stats_tracker.record_timing("rollout"):
            if config.async_training:
                batch = rollout.prepare_batch(dataloader, workflow=workflow)

                batch_total_len = len(batch) * len(batch[0]["input_ids"])
                logger.info(f"Got batch with len {len(batch)} and per tokens {len(batch[0]['input_ids'])}, total tokens {batch_total_len}")

            else:
                try:
                    data = next(data_generator)
                except StopIteration:
                    data_generator = iter(dataloader)
                    data = next(data_generator)
                batch = rollout.rollout_batch(data, workflow=workflow)

        avg_reward = batch["score"].mean().item()
        avg_code_reward = batch["code_reward"].mean().item()
        avg_code_in_correct = batch["code_in_correct"].mean().item()
        avg_length = (batch["attention_mask"].sum(1)).float().mean().item()

        if (not log_debug_batch) and avg_code_reward > 0:
            log_debug_batch = True
            with open("debug_batch.pkl", "wb") as log_file:
                pickle.dump(batch, log_file)

        batch = batch.to(actor.device)
        dist.barrier(device_ids=[actor.device.index])
        torch.cuda.synchronize()

        if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
            with stats_tracker.record_timing("recompute_logp"):
                logp = actor.compute_logp(batch)
                batch["prox_logp"] = logp
                log_gpu_stats("recompute logp")
        
        if ref is not None:
            with stats_tracker.record_timing("ref_logp"):
                batch["ref_logp"] = ref.compute_logp(batch)
                log_gpu_stats("ref logp")

        with stats_tracker.record_timing("compute_advantage"):
            actor.compute_advantages(batch)
            log_gpu_stats("compute advantages")
        
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

        with (
            stats_tracker.record_timing("train_step"),
            stats_tracker.scope("grpo_actor"),
        ):
            stats = actor.ppo_update(batch)
            actor.step_lr_scheduler()
            log_gpu_stats("ppo update")
        
        with stats_tracker.record_timing("update_weights"):
            rollout.pause()
            if dist.get_rank() == 0:
                future = rollout.update_weights(weight_update_meta)
            actor.upload_weights(weight_update_meta)
            if dist.get_rank() == 0:
                future.result()
            dist.barrier(device_ids=[actor.device.index])
            torch.cuda.synchronize()
            rollout.resume()
            actor.set_version(global_step + 1)
            rollout.set_version(global_step + 1)

        with stats_tracker.record_timing("save"):
            saver.save(actor, epoch, step, global_step)

        stats[0]["avg_reward"] = avg_reward
        stats[0]["avg_code_reward"] = avg_code_reward
        stats[0]["avg_code_in_correct"] = avg_code_in_correct
        stats[0]["avg_length"] = avg_length
        stat_logger.commit(epoch, step, global_step, stats)

    stat_logger.close()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])