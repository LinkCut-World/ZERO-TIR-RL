from .python_code import execute_python
from dataclasses import dataclass, field
from areal.api.cli_args import BaseExperimentConfig

@dataclass
class EnvironmentConfig(BaseExperimentConfig):
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

class CodeExecutionToolBox:
    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.previous_codes = []

    def step(self, action: str):
        if self.enable_history_code_execution:
            self.previous_codes.append(action)
            codes = self.previous_codes
        else:
            codes = [action]
        stdout, stderr, has_error = execute_python(codes, self.config.timeout, None, self.config.python_path, self.config.pre_import_lib, self.config.use_firejail)
        
        return {"stdout": stdout, "stderr": stderr, "has_error": has_error}