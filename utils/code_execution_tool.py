from .python_code import execute_python


class CodeExecutionToolBox:
    def __init__(self, timeout: float = None, enable_history_code_execution: bool = False, python_path: str = None, pre_import_lib: bool = False, use_firejail = True):
        self.timeout = timeout
        self.enable_history_code_execution = enable_history_code_execution
        self.python_path = python_path
        self.pre_import_lib = pre_import_lib
        self.use_firejail = use_firejail

        self.previous_codes = []

    def step(self, action: str):
        if self.enable_history_code_execution:
            self.previous_codes.append(action)
            codes = self.previous_codes
        else:
            codes = [action]
        stdout, stderr, has_error = execute_python(codes, self.timeout, None, self.python_path, self.pre_import_lib, self.use_firejail)
        
        return {"stdout": stdout, "stderr": stderr, "has_error": has_error}