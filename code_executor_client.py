"""
代码执行客户端
用于向代码执行服务器发送执行请求
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class CodeExecutorClient:
    def __init__(self, server_url: str = "http://127.0.0.1:8888"):
        self.server_url = server_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def execute_code(self, code: str, timeout: int = 10) -> Dict[str, Any]:
        """
        执行Python代码
        
        Args:
            code: 要执行的Python代码
            timeout: 超时时间（秒）
        
        Returns:
            执行结果字典，包含success, stdout, stderr, error等字段
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            async with self.session.post(
                f"{self.server_url}/execute",
                json={"code": code, "timeout": timeout},
                timeout=aiohttp.ClientTimeout(total=timeout + 5)
            ) as response:
                result = await response.json()
                return result

        except Exception as e:
            # 其他客户端错误，返回success=False
            return {
                "success": False,
                "stdout": "",
                "stderr": "",
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": ""
                }
            }
    
    async def health_check(self) -> bool:
        """
        检查服务器是否健康
        
        Returns:
            服务器是否可用
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            async with self.session.get(
                f"{self.server_url}/health",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("status") == "healthy"
                return False
        except:
            return False

# 便利函数
async def execute_python_code(code: str, timeout: int = 10, server_url: str = "http://127.0.0.1:8888") -> Dict[str, Any]:
    """
    执行Python代码的便利函数
    
    Args:
        code: 要执行的Python代码
        timeout: 超时时间（秒）
        server_url: 服务器URL
    
    Returns:
        执行结果字典
    """
    async with CodeExecutorClient(server_url) as client:
        return await client.execute_code(code, timeout)
