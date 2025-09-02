# 导入Ray分布式计算框架
import ray
# 导入操作系统、复制操作相关模块
import os, copy
# 导入UUID生成模块，用于创建唯一标识符
import uuid

# 导入系统和操作系统模块（重复导入os）
import sys, os

# 导入HTTP请求库，用于远程API调用
import requests
# 从数学环境模块导入代码执行函数
from env.math.code_exec import run_code
# 从数学环境模块导入代码提取函数
from env.math.extract_code import extract_code
# 导入有序字典，保持插入顺序
from collections import OrderedDict
# 导入正则表达式和UUID模块（重复导入uuid）
import re, uuid
# 从vLLM库导入采样参数类
from vllm import SamplingParams

# 导入日志模块并配置基本设置
import logging
# 配置日志基本格式
logging.basicConfig()
# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)
# 设置日志级别为INFO
logger.setLevel(logging.INFO)

# 创建全局HTTP会话对象，复用连接提升性能
session = requests.Session()

# 导入随机数模块
import random
# 导入时间模块，用于延迟操作
import time
# 编译正则表达式模式，用于匹配python代码块
code_pattern = re.compile(r"```python.*?```", re.DOTALL)
# 从环境变量获取进程排名，默认为1000
RANK = int(os.getenv('RANK', '1000'))

# 从环境变量获取编译服务器地址，默认为空字符串
COMPILE_SERVER = os.getenv('COMPILE_SERVER', '')
# 从环境变量获取调试标志，默认为空字符串
DEBUG_FLAG = os.getenv('DEBUG_FLAG', '')

# 记录编译服务器配置信息
logger.info({
    'INFO': 'COMPILE_SERVER',
    "VALUE": COMPILE_SERVER
})

# 导入类型提示相关模块
from typing import Generic, TypeVar, Union, NamedTuple

# 定义输出结果的数据结构
class Output(NamedTuple):
    # token序列的ID列表
    token_ids: list[int]
    # 动作掩码，标记哪些token是模型生成的(1)或自动添加的(0)
    action_mask: list[int]
    # 生成的文本内容
    text: str
    # 停止生成的原因
    stop_reason: str
    # 完成生成的原因
    finish_reason: str

# 定义生成输出的数据结构
class GenerateOutput(NamedTuple):
    # 输出结果列表
    outputs: list[Output]
    # 提示词的token ID列表
    prompt_token_ids: list[int]
    # 请求的唯一标识符
    request_id: str


# 定义远程代码执行函数
def remote_compile(code4exec, try_max_times=10, score_key='exec_result'):

    # 设置HTTP请求头，指定JSON内容类型
    headers = {
        "Content-Type": "application/json",
    }

    # 构造请求数据，包含代码和唯一标识符
    data = {
        'query': code4exec,
        'uuid_str': str(uuid.uuid4()),
    }

    # 循环重试，最多try_max_times次
    for try_idx in range(try_max_times):
        # 构造完整的URL地址
        url = COMPILE_SERVER+'/compile_python'
        try:
            # 发送POST请求，设置180秒超时
            response = session.post(url=url, json=data, headers=headers, timeout=180)
            # 检查HTTP响应状态，有错误则抛异常
            response.raise_for_status()  # Raise an HTTPError for bad responses
            # 解析JSON响应
            response = response.json()
            # 返回指定键的值
            return response.get(score_key)
        # 捕获HTTP请求异常
        except requests.RequestException as e:
            # 记录请求错误日志
            logger.info(f"Request error, please check: {e}")
        # 捕获其他所有异常
        except Exception as e:
            # 记录未预期错误日志
            logger.info(f"Unexpected error, please check: {e}")
        # 等待1秒后重试
        # 等待1秒后重试
        time.sleep(1)

# 定义数学工具集成推理的生成函数
def math_tir_generate(llm, sampling_params, prompt_token_ids, tokenizer, prompts=None):
    
    # 为每个prompt初始化输出token列表
    output_token_ids = [[] for idx in range(len(prompts))]
    # 为每个prompt初始化动作掩码列表
    action_masks = [[] for idx in range(len(prompts))]
    # 为每个prompt初始化生成文本列表
    all_text = ['' for idx in range(len(prompts))]
    # 为每个prompt初始化停止原因列表
    all_stop_reason = ['' for idx in range(len(prompts))]
    # 为每个prompt初始化完成原因列表
    all_finish_reason = ['' for idx in range(len(prompts))]
    # 为每个prompt初始化请求ID列表
    all_request_id = ['' for idx in range(len(prompts))]
    # 如果没有提供prompt_token_ids，则对每个prompt进行tokenization
    if prompt_token_ids is None:
        # 初始化prompt token ID列表
        all_prompt_token_ids = []
        # 遍历每个prompt
        for prompt in prompts:
            # 使用tokenizer将prompt转换为token IDs
            input_ids = tokenizer(prompt)['input_ids']
            # 添加到列表中
            all_prompt_token_ids.append(input_ids)
    else:
        # 使用提供的prompt token IDs
        all_prompt_token_ids = prompt_token_ids

    # 创建ID到UUID的映射字典
    id2uuid = OrderedDict()
    # 创建UUID到ID的映射字典
    uuid2id = OrderedDict()
    # 创建UUID到数据的映射字典
    uuid2data = OrderedDict()

    # 为每个prompt创建唯一的UUID映射
    for idx, prompt in enumerate(prompts):
        # 生成唯一的UUID字符串
        uuid_num = str(uuid.uuid4())
        # 建立ID到UUID的映射
        id2uuid[idx] = uuid_num
        # 建立UUID到ID的映射
        uuid2id[uuid_num] = idx
        # 建立UUID到prompt数据的映射
        uuid2data[uuid_num] = prompt

    # 初始化所有prompt的终止状态为False
    is_all_terminated = [False for _ in range(len(prompts))]
    # 检查是否所有prompt都已终止
    is_terminated = sum(is_all_terminated) == len(is_all_terminated)
    # 创建当前活跃的prompt索引列表
    idx_list = list(range(len(prompts)))

    # 初始化轮数计数器
    turn = 0
    # 复制采样参数以便修改
    new_sampling_params = copy.copy(sampling_params)
    # 添加Python代码块开始标记为停止符
    new_sampling_params.stop += ['```python']
    # 去重停止符列表
    new_sampling_params.stop = list(set(new_sampling_params.stop))

    # 为每个prompt创建独立的采样参数副本
    sampling_params_list = [copy.copy(new_sampling_params) for _ in range(len(prompts))]
    # 复制原始prompts
    new_prompts = prompts.copy()

    # 初始化迭代次数计数器
    iterative_num = 0

    # 主循环：持续生成直到所有prompt都终止
    while not is_terminated:

        # 根据当前活跃的索引筛选采样参数
        new_sampling_params_list = [sampling_params_list[idx] for idx in idx_list]
        # 使用LLM批量生成内容
        outputs = llm.generate(sampling_params=new_sampling_params_list, prompts=new_prompts)
        # 如果是第一次迭代，保存请求ID
        if iterative_num == 0:
            # 遍历输出结果，保存请求ID
            for idx, output in enumerate(outputs):
                all_request_id[idx] = output.request_id

        # 初始化下一轮的索引和prompt列表
        left_idx = []
        left_prompts = []

        # 处理每个生成结果
        for index, (prompt, output, prompt_idx) in enumerate(zip(new_prompts, outputs, idx_list)):
            # 提取生成的文本内容
            text = output.outputs[0].text
            # 提取token ID列表
            token_ids = list(output.outputs[0].token_ids)
            # 初始化动作掩码（全部为1，表示模型生成）
            action_mask = [1] * len(token_ids)
            # 保存停止原因
            all_stop_reason[prompt_idx] = output.outputs[0].stop_reason
            # 保存完成原因
            all_finish_reason[prompt_idx] = output.outputs[0].finish_reason
            
            # 如果因为```python停止符而停止生成
            if output.outputs[0].stop_reason in ['```python']:
                # 获取当前prompt的采样参数
                new_sampling_params = sampling_params_list[prompt_idx]
                # 添加```作为停止符，准备生成代码
                new_sampling_params.stop += ['```']
                # 去重停止符列表
                new_sampling_params.stop = list(set(new_sampling_params.stop))

                # 计算剩余可生成的最大token数
                max_tokens = new_sampling_params.max_tokens
                max_tokens -= len(output_token_ids[prompt_idx])
                # 如果剩余token数大于0，更新最大token数
                if max_tokens > 0:
                    new_sampling_params.max_tokens = max_tokens
                else:
                    # 否则设置默认值1024
                    new_sampling_params.max_tokens = 1024

                # 移除```python停止符，因为已经遇到了
                if '```python' in new_sampling_params.stop:
                    new_sampling_params.stop.remove('```python')
                
                # 将此prompt索引添加到继续生成列表
                left_idx.append(prompt_idx)
                # 更新prompt内容，加上已生成的文本
                left_prompts.append(prompt+text)
                # 累积生成的文本
                all_text[prompt_idx] += text

                # 如果开启调试，记录代码生成阶段日志
                if DEBUG_FLAG == 'yes':
                    logger.info({
                        'STAGE': 'code-gen',
                        'prompt': prompt,
                        'output': text,
                        'params': new_sampling_params
                    })
                
            # 如果因为```停止符而停止生成（代码块结束）
            elif output.outputs[0].stop_reason in ['```']:
                # 获取当前prompt的采样参数
                new_sampling_params = sampling_params_list[prompt_idx]
                # 重新添加```python为停止符
                new_sampling_params.stop += ['```python']
                # 去重停止符列表
                new_sampling_params.stop = list(set(new_sampling_params.stop))
                # 移除```停止符，因为已经遇到了
                if '```' in new_sampling_params.stop:
                    new_sampling_params.stop.remove('```')

                # 计算剩余可生成的最大token数
                max_tokens = new_sampling_params.max_tokens
                max_tokens -= len(output_token_ids[prompt_idx])
                # 如果剩余token数大于0，更新最大token数
                if max_tokens > 0:
                    new_sampling_params.max_tokens = max_tokens
                else:
                    # 否则设置默认值1024
                    new_sampling_params.max_tokens = 1024

                # 使用正则表达式提取Python代码块
                code_text = re.findall(code_pattern, f"```python\n{text}")
                # logger.info({
                #     'STAGE': 'detect-code-exec',
                #     'code_text': code_text,
                #     'output': text,
                #     'params': new_sampling_params
                # })
                
                # 如果找到了代码块
                if code_text:
                    # 提取第一个代码块
                    code_text = code_text[0]
                    # 使用extract_code函数提取可执行代码
                    code4exec = extract_code(code_text)

                    # 如果开启调试，记录代码执行阶段日志
                    if DEBUG_FLAG == 'yes':
                        logger.info({
                            'STAGE': 'code-exec',
                            'code_text': code4exec,
                            'output': text,
                            'params': new_sampling_params
                        })
                    
                    # 如果成功提取了可执行代码
                    if code4exec:

                        # 如果没有配置远程编译服务器，使用本地执行
                        if not COMPILE_SERVER:
                            try:
                                # 本地执行代码
                                result = run_code(code4exec)
                            except Exception as e:
                                # 捕获异常并转为字符串
                                result = str(e)
                        else:
                            try:
                                # 使用远程编译服务执行代码
                                result = remote_compile(code4exec)
                            except:
                                # 远程执行失败时返回超时错误
                                result = 'TimeOut Error'

                        # 格式化代码执行结果
                        code_output = f"""\n```output\n{result}"""

                        # logger.info({
                        #     'STAGE': 'code-exec-output',
                        #     'code_text': code_output,
                        #     'output': text,
                        #     'params': new_sampling_params
                        # })

                        # 将代码输出转换为token IDs
                        code_output_ids = tokenizer(code_output, add_special_tokens=False)['input_ids']
                        # 如果输出太长，截断并提示
                        if len(code_output_ids) > 512:
                            code_output = """The output of the code is too long, please check your code."""
                            code_output_ids = tokenizer(code_output, add_special_tokens=False)['input_ids']
                        
                        # 将代码输出的token IDs添加到生成结果中
                        token_ids.extend(code_output_ids)
                        # 将代码输出标记为自动添加（action_mask=0）
                        action_mask.extend([0]*len(code_output_ids))
                    else:
                        # 如果没有提取到代码，输出为空
                        code_output = ''

                    # 将此prompt索引添加到继续生成列表
                    left_idx.append(prompt_idx)
                    # 更新prompt内容，加上已生成的文本和代码输出
                    left_prompts.append(prompt+text+code_output)
                    # 累积生成的文本和代码输出
                    all_text[prompt_idx] += (text+code_output)
                else:
                    # 如果没有找到代码块，标记为终止
                    is_all_terminated[prompt_idx] = True
                    # 累积生成的文本
                    all_text[prompt_idx] += text
            else:
                # 其他停止原因，标记为终止
                is_all_terminated[prompt_idx] = True
                # 累积生成的文本
                all_text[prompt_idx] += text

            # 将当前生成的token IDs添加到输出中
            output_token_ids[prompt_idx].extend(token_ids)
            # 将当前生成的动作掩码添加到输出中
            action_masks[prompt_idx].extend(action_mask)

        # 检查是否所有prompt都已终止
        is_terminated = sum(is_all_terminated) == len(is_all_terminated)
        # 更新下一轮要处理的prompts
        new_prompts = left_prompts
        # 更新下一轮要处理的索引列表
        idx_list = left_idx

        # 确保prompts和索引列表长度一致
        assert len(new_prompts) == len(idx_list)

        # 如果迭代次数超过3次，强制退出循环
        if iterative_num >= 3:
            break
        # 增加迭代计数器
        iterative_num += 1
    
    # 构造最终输出列表
    outputs = []
    # 遍历所有结果数据，构造输出对象
    for (output_token_id, action_mask, 
        output_text, stop_reason, 
        finish_reason, prompt_token_id, 
        request_id) in zip(output_token_ids, 
                            action_masks, all_text, 
                            all_stop_reason, all_finish_reason,
                            all_prompt_token_ids, all_request_id):
        # 确保token IDs和动作掩码长度一致
        assert len(output_token_id) == len(action_mask)
        # 创建GenerateOutput对象
        tmp = GenerateOutput(
                outputs=[Output(
                    token_ids=output_token_id,
                    # 如果没有结束符，在动作掩码末尾添加1
                    action_mask=action_mask+[1] if tokenizer.eos_token_id not in output_token_id else action_mask,
                    text=output_text,
                    stop_reason=stop_reason,
                    finish_reason=finish_reason
                )],
                prompt_token_ids=prompt_token_id,
                request_id=request_id
        )
        # 添加到输出列表
        outputs.append(tmp)
    # 清理临时变量，释放内存
    del sampling_params_list, idx_list, new_prompts
    # 返回最终输出结果
    return outputs