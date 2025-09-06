import asyncio
import copy
import datasets
import os
import pickle
import re
import sys
import time
import uuid
import torch
from tensordict import TensorDict
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from dataclasses import dataclass, field

from areal.api.cli_args import (
    GenerationHyperparameters,
    GRPOConfig,
    load_expr_config,
)
from areal.api.io_struct import ModelRequest
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.utils.data import concat_padded_tensors
from realhf.base import logging, seeding
from realhf.impl.dataset.math_parser import extract_answer, math_equal

from utils.code_execution_tool import CodeExecutionToolBox, EnvironmentConfig


logger = logging.getLogger("TIR")

@dataclass
class WorkflowConfig:
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

@dataclass
class AgentRLConfig(GRPOConfig):
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    dump_dir: str = field(
        default="./dump",
        metadata={
            "help": "directory to dump the training logs and models"
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

def reward_fn(generated: str, answer: str) -> tuple[str, float]:
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

def execute_code(env: CodeExecutionToolBox, code: str) -> str:
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
        config: WorkflowConfig,
        gconfig: GenerationHyperparameters,
        envconfig: EnvironmentConfig,
        tokenizer: PreTrainedTokenizerFast,
    ):
        self.config = config
        self.gconfig = gconfig
        self.gconfig.n_samples = 1
        self.envconfig = envconfig
        self.tokenizer = tokenizer
        self.current_trajs = 0
        self.code_env = CodeExecutionToolBox(envconfig)
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
        std = (sum([(t["rewards"].item() - avg_reward) ** 2 for t in trajs]) / len(trajs)) ** 0.5
        for traj in trajs:
            traj["rewards"] = (traj["rewards"] - avg_reward) / (std + 1e-6)
        
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

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)

    workflow = TIRWorkflow(
        config=config.workflow,
        gconfig=config.gconfig,
        envconfig=config.environment,
        tokenizer=tokenizer,
    )

    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(None, None)

    for i in range(len(dataset_name_s)):
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

        d = {}
        for key in batch.keys():
            shape = batch[key].shape
            if len(shape) == 1 or (len(shape) == 2 and shape[1] == 1):
                d[key] = batch[key].float().mean().item()

        batch = [batch[i:i+config.workflow.n_trajs] for i in range(0, len(batch), config.workflow.n_trajs)]

        avg_pass_k = torch.tensor([b["score"].sum().item() > 0 for b in batch], dtype=torch.float32).mean().item()

        stats["score_stat"][i]["pass@1"].append(d["score"])
        stats["score_stat"][i]["pass@k"].append(avg_pass_k)
        stats["length_stat"][i]["avg_length"].append(d["length"])
        stats["code_stat"][i]["code_reward"].append(d["code_reward"])
        stats["code_stat"][i]["code_in_correct"].append(d["code_in_correct"])

    with open(statistic_path, "wb") as f:
        pickle.dump(stats, f)

    logger.info("finished evaluation, destroying ...")
    rollout.destroy()
    logger.info("destroyed.")

if __name__ == "__main__":
    main(sys.argv[1:])