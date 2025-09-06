# Introduction

This is an example that implements [ZERO-TIR-RL](https://zhuanlan.zhihu.com/p/1889286471078368477) with [AReaL-lite](https://github.com/inclusionAI/AReaL).

Tool Integrated Reasoning (TIR) refers to enabling LLMs to use external tools during the reasoning process, significantly enhancing the model's problem-solving and reasoning capabilities.

ZERO-TIR-RL performs RL on base model and enables the model to learn how to use a python-code executor to solve mathematical problems.

# Implementation Details

The implementation involves these files:

```
ZERO-TIR-RL/
├── train1/
│   ├── train_config.yaml
│   └── train.py
└── utils/
```

- `train1/train_config.yaml`: The configuration file for training that specifies hyperparameters and environment settings.
- `train1/train.py`: The main training script. The key component is the `TIRWorkflow` class with `arun_episode` method, which takes a prompt and generates rollouts for RL training.

    ```python
    from areal.api.workflow_api import RolloutWorkflow
    from areal.api.reward_api import AsyncRewardWrapper

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
            self.envconfig = envconfig
            self.tokenizer = tokenizer
            self.code_env = CodeExecutionToolBox(envconfig)
            self.async_reward_fn = AsyncRewardWrapper(reward_fn)
            self.async_code_exec = AsyncRewardWrapper(execute_code)

        async def collect_agent_trajectory(self, qid, prompt, answer, engine):
            traj_rid = uuid.uuid4().hex  # Unique identifier for the trajectory

            num_turns = 0
            stops = ["```python", "</answer>"]

            # Token IDs, logprobs, and loss mask for RL training
            input_str = prompt
            input_ids = self.tokenizer.encode(input_str, add_special_tokens=False)
            logprobs = [0.0] * len(input_ids)
            loss_mask = [0] * len(input_ids)
            
            while num_turns < self.config.max_turns:
                req = ModelRequest(
                    rid=traj_rid,
                    input_ids=input_ids,
                    gconfig=self.gconfig.new(n_samples=1),
                )
                req.gconfig.stop = stops  # Set stop tokens
                # Limit the total tokens in a trajectory
                if len(input_ids) + self.gconfig.max_new_tokens >= self.config.max_tokens_per_traj:
                    break
                
                resp = await engine.agenerate(req)  # Generate model response
                completion_str = self.tokenizer.decode(resp.output_tokens)

                # Record the completion
                input_str += completion_str
                input_ids += resp.output_tokens
                logprobs += resp.output_logprobs
                loss_mask += [1] * len(resp.output_tokens)

                if "</answer>" in completion_str:
                    break
                # We meet the start of a code block. Set "```" as stop token
                elif stops[0] == "```python" and "```python" in completion_str:
                    stops[0] = "```"
                # We meet the end of a code block.
                elif stops[0] == "```" and "```" in completion_str:
                    matches = re.findall(r'```python(.*?)```', input_str, re.DOTALL)
                    if matches:
                        code = matches[-1]
                        num_turns += 1
                        execution_output = await self.async_code_exec(self.code_env, code)  # Execute the code

                        # Record the execution completion
                        input_str += execution_output
                        exec_tokens = self.tokenizer.encode(execution_output, add_special_tokens=False)
                        if len(input_ids) + len(exec_tokens) >= self.config.max_tokens_per_traj:
                            exec_tokens = exec_tokens[:self.config.max_tokens_per_traj - len(input_ids) - 1]
                        input_ids += exec_tokens
                        logprobs += [0.0] * len(exec_tokens)
                        loss_mask += [0] * len(exec_tokens)
                    stops[0] = "```python"  # Reset stop token to "```python"
                
                if resp.output_tokens[-1] in [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]:
                    break

            extracted, reward = await self.async_reward_fn(input_str, answer=answer)  # Compute reward

            # Convert to TensorDict
            res = dict(
                input_ids=torch.tensor(input_ids),
                logprobs=torch.tensor(logprobs),
                loss_mask=torch.tensor(loss_mask, dtype=torch.bool),
                rewards=torch.tensor(float(reward)),
                attention_mask=torch.ones(len(input_ids), dtype=torch.bool),
            )
            res = {k: v.unsqueeze(0) for k, v in res.items()}
            return TensorDict(res, batch_size=[1])

        async def arun_episode(self, engine, data):
            qid = uuid.uuid4().hex  # Unique identifier for the question

            prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": data["question"]}
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt += "<think>\n"

            # Collect multiple trajectories for the same question
            trajs = await asyncio.gather(*[
                self.collect_agent_trajectory(qid, prompt, data["answer"], engine)
                for _ in range(self.config.n_trajs)
            ])

            # Compute advantages (GRPO)
            avg_reward = sum([t["rewards"].item() for t in trajs]) / len(trajs)
            if avg_reward == 0:
                return None
            std = (sum([(t["rewards"].item() - avg_reward) ** 2 for t in trajs]) / len(trajs)) ** 0.5
            for traj in trajs:
                traj["rewards"] = (traj["rewards"] - avg_reward) / (std + 1e-6)
            
            return concat_padded_tensors(trajs)
    ```
- `utils/`: The code execution tool.

# Experiments

We train Qwen2.5-1.5B on [orz-57k dataset](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/data/orz_math_57k_collected.json).

[Experiment Configuration]

[Result]
