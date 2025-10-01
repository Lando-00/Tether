from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Callable, Any
from llm_service.protocol.core.interfaces import ExecutionStrategy

class ThreadPoolExecutionStrategy(ExecutionStrategy):
    def __init__(self, max_workers: int = 4):
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tool-exec")
    
    def execute_with_timeout(self, func: Callable, args: tuple, kwargs: dict, timeout: int) -> Any:
        future = self._thread_pool.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            return f"Tool execution timed out after {timeout} seconds"
        except Exception as e:
            return f"Tool execution failed: {str(e)}"
    
    def cleanup(self) -> None:
        self._thread_pool.shutdown(wait=True)
