import os
import swanlab
import datasets
import copy

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

    dump_dir: str = field(
        default="./eval",
        metadata={
            "help": "directory to dump the trajectories"
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
            return x, 0.0
        elif y is None or y.strip() in ["None", "none", ""]:
            return x, 0.0
        return x, float(math_equal(x, y, timeout=False))
    except:
        return None, 0.0


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
from ..code_executor import execute_code
import shutil
import pickle

class TIRWorkflow(RolloutWorkflow):
    def __init__(
        self, 
        config: AgentRLConfig, 
        tokenizer: PreTrainedTokenizerFast,
    ):
        self.config = copy.deepcopy(config)
        self.gconfig = config.gconfig
        self.gconfig.n_samples = 1
        self.tokenizer = tokenizer
        self.dump_lock = threading.Lock()
        self.current_trajs = 0

        self.example_trajs_path = os.path.join(self.config.dump_dir, "example_trajs.log")
        with open(self.example_trajs_path, "w") as log_file:
            log_file.write("")

    async def collect_agent_trajectory(self, qid, prompt, answer, engine):
        traj_rid = uuid.uuid4().hex
        loop = asyncio.get_event_loop()
        extracted, result = None, None
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
                    extracted, reward = await loop.run_in_executor(
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
            extracted=torch.tensor(self.tokenizer.encode(extracted) if extracted is not None else [])
        )

        res_dump = {k: v.tolist() for k, v in res.items() if k != 'attention_mask' and k != 'rewards'}
        res_dump['input_str'] = input_str
        res_dump['metadata'] = {
            "score": reward,
            "qid": qid,
            "traj_rid": traj_rid,
            "num_turns": num_turns,
            "length": len(input_ids),
            "total_time": f"{total_time:.2f}s",
            "gen_time_ratio": f"{total_gen_time / total_time:.2f}" if total_time > 0 else "0.00",
            "exec_time_ratio": f"{total_exec_time / total_time:.2f}" if total_time > 0 else "0.00",
            "answer": answer,
            "extracted": extracted,
        }

        if self.current_trajs % 100 == 0:
            with open(self.example_trajs_path, "a") as log_file:
                log_file.write(res_dump['input_str'].__str__() + "\n" + res_dump['metadata'].__str__() + "\n")
        self.current_trajs += 1

        with self.dump_lock:
            with open(self.config.dump_dir, "a") as f:
                json.dump(res_dump['metadata'], f)
                f.write("\n")

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
        for traj in trajs:
            traj["rewards"] -= avg_reward
        
        return concat_padded_tensors(trajs)

def main(args):
    assert len(args) >= 2 and args[-2].startswith("+checkpoint_name=") and args[-1].startswith("+statistic_path="), \
        "Expected last two args to be '+checkpoint_name=...' and '+statistic_path=...'"
    checkpoint_name = args[-2].split("=", 1)[1]
    statistic_path = args[-1].split("=", 1)[1]
    args = args[:-2]

    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"

    config, _ = load_expr_config(args, AgentRLConfig)
    config: AgentRLConfig

    logger.info(f"model_path: {config.sglang.model_path}")

    config.dump_dir = os.path.join(
        config.dump_dir, checkpoint_name
    )
    logger.info(f"dump path: {config.dump_dir}")

    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")

    dataset_name_s = ["AIME25", "MATH-500"]
    # dataset_name_s = ["AIME25"]

    with open(statistic_path, "rb") as f:
        stats = pickle.load(f)
        if stats is None:
            template_stats = {
                "score_stat": {
                    "pass@1": [],
                    "pass@k": [],
                },
                "length_stat": {
                    "avg_length": [],
                },
                "code_stat": {
                    "code_reward": [],
                    "code_in_correct": [],
                },
            }
            stats = {}
            for k, v in template_stats.items():
                stats[k] = [copy.deepcopy(v) for _ in range(len(dataset_name_s))]

    dataset_s = [datasets.load_from_disk(f'eval_ds/{name}') for name in dataset_name_s]

    dataloader_s = [StatefulDataLoader(
        dataset,
        batch_size=config.train_dataset.batch_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=config.train_dataset.num_workers,
        collate_fn=lambda x: x,
        drop_last=config.train_dataset.drop_last,
    ) for dataset in dataset_s]

    data_generator_s = [cycle(dataloader)
                        for dataloader in dataloader_s]

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

    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(None, None)

    for i, data_generator in enumerate(data_generator_s):
        logger.info(f"Evaluating on dataset: {dataset_name_s[i]} with {len(dataset_s[i])} samples")
        dump_dir = os.path.join(
            config.dump_dir, dataset_name_s[i], "trajs.log"
        )
        os.makedirs(os.path.dirname(dump_dir), exist_ok=True)
        with open(dump_dir, "w") as f:
            f.write("")

        # example_data = dataset_s[i].select(range(2))
        workflow.config.dump_dir = dump_dir
        batch = rollout.rollout_batch(dataset_s[i], workflow=workflow)

        # batch[0]:
        # TensorDict(
        #     fields={
        #         attention_mask: Tensor(shape=torch.Size([491]), device=cpu, dtype=torch.bool, is_shared=False),
        #         input_ids: Tensor(shape=torch.Size([491]), device=cpu, dtype=torch.int64, is_shared=False),
        #         logprobs: Tensor(shape=torch.Size([491]), device=cpu, dtype=torch.float32, is_shared=False),
        #         loss_mask: Tensor(shape=torch.Size([491]), device=cpu, dtype=torch.bool, is_shared=False),
        #         rewards: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
        #     batch_size=torch.Size([]),
        #     device=None,
        #     is_shared=False)

        avg_score = batch["score"].mean().item()
        avg_code_reward = batch["code_reward"].mean().item()
        avg_code_in_correct = batch["code_in_correct"].mean().item()
        # avg_length = batch["length"].mean().item()
        avg_length = (batch["attention_mask"].sum(1)).float().mean().item()

        batch = [batch[i:i+config.n_trajs] for i in range(0, len(batch), config.n_trajs)]

        avg_pass_k = torch.tensor([b["score"].sum().item() > 0 for b in batch], dtype=torch.float32).mean().item()

        stats["score_stat"][i]["pass@1"].append(avg_score)
        stats["score_stat"][i]["pass@k"].append(avg_pass_k)
        stats["length_stat"][i]["avg_length"].append(avg_length)
        stats["code_stat"][i]["code_reward"].append(avg_code_reward)
        stats["code_stat"][i]["code_in_correct"].append(avg_code_in_correct)

    with open(statistic_path, "wb") as f:
        pickle.dump(stats, f)

    logger.info("finished evaluation, destroying ...")
    rollout.destroy()
    logger.info("destroyed.")

if __name__ == "__main__":
    main(sys.argv[1:])