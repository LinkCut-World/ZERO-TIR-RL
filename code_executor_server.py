#!/usr/bin/env python3
"""
Python代码执行服务器
用于并行处理Python代码执行请求
"""

import asyncio
import json
import sys
import io
import contextlib
import traceback
import multiprocessing
from aiohttp import web, ClientSession
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _run_code_worker(code: str, result_queue: multiprocessing.Queue):
    """
    子进程中执行Python代码并将结果放入队列
    """
    try:
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            namespace = {}
            exec(code, namespace)

        stdout_output = stdout_buffer.getvalue()
        if len(stdout_output) > 100:
            stdout_output = stdout_output[:100] + "\n\n Output too long! Truncated."

        result_queue.put({
            "success": True,
            "content": stdout_output
        })

    except Exception:
        stdout_output = stdout_buffer.getvalue()
        error_info = traceback.format_exc()
        error_info = error_info.replace(
            'File "/home/liangchengwei/lcw/ZERO-TIR-RL/code_executor_server.py", line 30, in _run_code_worker\n    exec(code, namespace)\n  File "<string>",',
            ''
        )
        content = stdout_output
        if content and not content.endswith("\n"):
            content += "\n"
        content += error_info

        result_queue.put({
            "success": True,
            "content": content
        })


class CodeExecutor:
    def __init__(self, max_workers=4):
        # max_workers 现在没实际用途，但保留参数保证接口兼容
        self.max_workers = max_workers

    @staticmethod
    def execute_python_code(code: str, timeout: int = 10) -> dict:
        """
        在独立进程中执行Python代码，超时则强制终止进程
        """
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=_run_code_worker, args=(code, result_queue)
        )
        process.start()
        process.join(timeout)

        if process.is_alive():
            process.terminate()
            process.join()
            return {
                "success": True,
                "content": f"Execution timed out after {timeout} seconds."
            }

        # 如果队列中没有结果，说明子进程异常退出
        if result_queue.empty():
            return {
                "success": False,
                "error": {
                    "type": "NoResultError",
                    "message": "No result returned from execution process.",
                    "traceback": "",
                },
            }

        return result_queue.get()

    async def execute_async(self, code: str, timeout: int = 10) -> dict:
        """
        异步执行Python代码
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.execute_python_code, code, timeout
        )


# 全局执行器实例
code_executor = CodeExecutor(max_workers=4)

async def execute_code_handler(request):
    """
    处理代码执行请求的HTTP处理器
    """
    try:
        data = await request.json()
        code = data.get("code", "")
        timeout = data.get("timeout", 10)
        traj_rid = data.get("traj_rid", None)

        # 执行代码
        result = await code_executor.execute_async(code, timeout)

        logger.info(f"来自{traj_rid}的代码：\n{code}\n执行完成，结果：{result}")
        return web.json_response(result)

    except Exception as e:
        logger.error(f"处理请求时发生错误: {e}")
        return web.json_response(
            {
                "success": False,
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                },
            },
            status=500,
        )

async def health_check(request):
    """
    健康检查端点
    """
    return web.json_response({"status": "healthy"})

def create_app():
    """
    创建Web应用
    """
    app = web.Application()
    app.router.add_post("/execute", execute_code_handler)
    app.router.add_get("/health", health_check)
    return app

async def main():
    """
    启动服务器
    """
    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()

    host = "127.0.0.1"
    port = 1451

    site = web.TCPSite(runner, host, port)
    await site.start()

    logger.info(f"代码执行服务器启动在 http://{host}:{port}")
    logger.info("端点:")
    logger.info("  POST /execute - 执行Python代码")
    logger.info("  GET /health - 健康检查")

    try:
        await asyncio.Future()  # 运行直到被中断
    except KeyboardInterrupt:
        logger.info("服务器关闭中...")
    finally:
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
