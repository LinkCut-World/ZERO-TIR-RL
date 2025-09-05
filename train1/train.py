import asyncio
import copy
import os
import pickle
import re
import swanlab
import sys
import time
import uuid
import json
import gc
import torch
import torch.distributed as dist
from datasets.distributed import split_dataset_by_node
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from dataclasses import dataclass, field

from areal.api.cli_args import (
    GRPOConfig,
    load_expr_config,
)
from areal.api.io_struct import (
    FinetuneSpec,
    ModelRequest,
    WeightUpdateMeta,
)
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.utils.data import concat_padded_tensors
from areal.utils.device import log_gpu_stats
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from realhf.base import logging, seeding, stats_tracker
from realhf.impl.dataset.math_parser import extract_answer, math_equal

from utils.code_execution_tool import CodeExecutionToolBox

from datasets import Dataset as HFDataset

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

    # code execution environment settings
    timeout: int = field(
        default=10,
        metadata={
            "help": "timeout (in seconds) for code execution"
        }
    )
    enable_history_code_execution: bool = field(
        default=False,
        metadata={
            "help": "whether to enable history code execution"
        }
    )
    python_path: str = field(
        default="",
        metadata={
            "help": "specify the python path to run the code"
        }
    )
    pre_import_lib: bool = field(
        default=False,
        metadata={
            "help": "whether to pre-import some common libraries for code execution"
        }
    )
    use_firejail: bool = field(
        default=True,
        metadata={
            "help": "whether to use firejail to sandbox the code execution"
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

SYSTEM_PROMPT = """
You are a helpful assistant. The User asks a question, and the Assistant solves it. 
The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{{}} tag.
In your reasoning-process, You can use python-code to solve your problem. Put the code within ```python and ``` tags. The script will be executed immediately and output will be returned.
"""

def reward_fn(generated, answer):
    matches = re.findall(r"<answer>(.*?)</answer>", generated, re.DOTALL)
    if not matches:
        return None, 0.0
    generated = matches[-1]
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

def execute_code(env, code):
    res = env.step(code)
    res = res["stdout"] + '\n' + res["stderr"]
    if res.strip() == "":
        res = "No output. Let me use `print()` to see the result and try again.\n"
    if len(res) > 2000:
        res = res[:2000] + "\n...[truncated]"
    res = "\n```output\n" + res + "\n```\n"
    return res

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
        self.code_env = CodeExecutionToolBox(
            timeout=config.timeout,
            enable_history_code_execution=config.enable_history_code_execution,
            python_path=config.python_path,
            pre_import_lib=config.pre_import_lib,
            use_firejail=config.use_firejail,
        )
        self.async_reward_fn = AsyncRewardWrapper(reward_fn)
        self.async_code_exec = AsyncRewardWrapper(execute_code)

        self.example_trajs_path = "example_trajs.log"
        with open(self.example_trajs_path, "w") as log_file:
            log_file.write("")

    async def collect_agent_trajectory(self, qid, prompt, answer, engine):
        traj_rid = uuid.uuid4().hex

        num_turns = 0
        stops = ["```python", "</answer>"]

        input_str = prompt
        input_ids = self.tokenizer.encode(input_str, add_special_tokens=False)
        logprobs = [0.0] * len(input_ids)
        loss_mask = [0] * len(input_ids)
        
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
                break
            elif stops[0] == "```python" and "```python" in completion_str:
                stops[0] = "```"
            elif stops[0] == "```" and "```" in completion_str:
                matches = re.findall(r'```python(.*?)```', input_str, re.DOTALL)
                if matches:
                    code = matches[-1]
                    exec_start_time = time.time()
                    execution_output = await self.async_code_exec(self.code_env, code)
                    exec_time = time.time() - exec_start_time
                    total_exec_time += exec_time
                    
                    num_turns += 1

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

        extracted, reward = await self.async_reward_fn(input_str, answer=answer)

        total_time = time.time() - start_time

        assert len(input_ids) <= self.config.max_tokens_per_traj, f"Trajectory {traj_rid} exceeds max tokens {self.config.max_tokens_per_traj} with {len(input_ids)} tokens."
        
        stats = dict(
            score=torch.tensor(float(reward)),
            num_turns=torch.tensor(num_turns),
            code_reward=torch.tensor(float(num_turns>0)),
            code_in_correct=torch.tensor(float(num_turns>0 and reward>0)),
            length=torch.tensor(len(input_ids)),
            total_time=torch.tensor(total_time),
            gen_time=torch.tensor(total_gen_time),
            exec_time=torch.tensor(total_exec_time),
        )
        res = dict(
            input_ids=torch.tensor(input_ids),
            logprobs=torch.tensor(logprobs),
            loss_mask=torch.tensor(loss_mask, dtype=torch.bool),
            rewards=torch.tensor(float(reward)),
            attention_mask=torch.ones(len(input_ids), dtype=torch.bool),
        )
        res.update(stats)

        if self.current_trajs % 100 == 0:
            dump = copy.copy(stats)
            dump.update(dict(
                gen_time_ratio=f"{total_gen_time / total_time:.2f}" if total_time > 0 else "0.00",
                exec_time_ratio=f"{total_exec_time / total_time:.2f}" if total_time > 0 else "0.00",
                answer=answer,
                extracted=extracted,
            ))
            with open(self.example_trajs_path, "a") as log_file:
                log_file.write(input_str + "\n" + dump.__str__() + "\n\n")
        self.current_trajs += 1

        res = {k: v.unsqueeze(0) for k, v in res.items()}
        return TensorDict(res, batch_size=[1])

    async def arun_episode(self, engine, data):
        qid = uuid.uuid4().hex

        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": data["question"]}
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt += "<think>\n"

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
    log_debug_batch = 0
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

        d = {}
        for key in batch.keys():
            shape = batch[key].shape
            if len(shape) == 1 or (len(shape) == 2 and shape[1] == 1):
                d[key] = batch[key].float().mean().item()

        if batch["num_turns"].float().max().item() >= log_debug_batch:
            log_debug_batch = batch["num_turns"].float().max().item()
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

        for k, v in d.items():
            stats[0]["stat/" + k] = v
        stat_logger.commit(epoch, step, global_step, stats)

    stat_logger.close()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])