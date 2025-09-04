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

def reward_fn(generated, answer):
    matches = re.findall(r"<answer>(.*?)</answer>", generated, re.DOTALL)
    if not matches:
        return 0.0
    generated = matches[-1]
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

SYSTEM_PROMPT = """
A conversation between User and Assistant. 
The User asks a question, and the Assistant solves it. 
The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The answer is enclosed within <answer> </answer> tags, respectively, i.e., <answer> answer here </answer>. And the final answer will be extracted automatically by the \\boxed{} tag.
In the reasoning-process, the Assistant can use python-code to solve the problem. 
"""

import code_executor

def execute_code(code, timeout):
    res = code_executor.execute_code(code, timeout=timeout)
    if res == "":
        return "No output. Use `print()` to see the result. Try again.\n"
    return res

import shutil
import pickle
from datasets import Dataset as HFDataset
from areal.api.reward_api import AsyncRewardWrapper
from areal.experimental.openai import ArealOpenAI
import copy

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

        self.async_reward_fn = AsyncRewardWrapper(reward_fn)
        self.async_code_exec = AsyncRewardWrapper(execute_code)


    async def collect_agent_trajectory(self, qid, messages, answer, engine):
        client = ArealOpenAI(engine=engine, tokenizer=self.tokenizer)
        messages = copy.deepcopy(messages)

        num_turns = 0
        total_gen_time = 0
        total_exec_time = 0
        start_time = time.time()
        comp_ids = []
        while num_turns < self.config.max_turns:
            # if len(input_ids) + self.gconfig.max_new_tokens >= self.config.max_tokens_per_traj:
            #     break
            gen_start_time = time.time()
            _comp = await client.chat.completions.create(
                messages=messages,
                frequency_penalty=self.gconfig.frequency_penalty,
                max_completion_tokens=self.gconfig.max_new_tokens,
                stop=self.gconfig.stop,
                store=True,
                temperature=self.gconfig.temperature,
                top_p=self.gconfig.top_p,
                tools=[
                    {
                        "type": "custom",
                        "name": "code_exec",
                        "description": "Executes arbitrary Python code.",
                    }
                ],
                tool_choice="auto",
            )
            comp = client.get_completions(_comp.id)
            gen_time = time.time() - gen_start_time
            total_gen_time += gen_time

            resp = _comp.choices[0]
            messages += [resp.message]
            comp_ids += [_comp.id]

            if resp.finish_reason == "tool_use":
                num_turns += 1
                tool_call = resp.message.tool_calls[0]
                code = tool_call.custom.input

                exec_start_time = time.time()
                execution_output = await self.async_code_exec(code, self.config.code_execution_timeout)
                exec_time = time.time() - exec_start_time
                total_exec_time += exec_time
                
                messages += [
                    {
                        "role": "tool",
                        "content": execution_output,
                        "tool_call_id": tool_call.id,
                    }
                ]
            elif resp.finish_reason == "stop":
                break
        
        
        reward = await self.async_reward_fn(
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ),
            answer,
        )
        for id in comp_ids:
            client.set_reward(id, reward)
        total_time = time.time() - start_time

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
        return client.export_completions(turn_discount=0.0)

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
        for traj in trajs:
            traj["rewards"] -= avg_reward
        
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

    with open('orz_math_57k_collected.json', 'r') as f:
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