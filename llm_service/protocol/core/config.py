import os
from llm_service.protocol.core.interfaces import ConfigProvider

class EnvironmentConfigProvider(ConfigProvider):
    def get_max_token_buffer_size(self) -> int:
        return int(os.getenv("MAX_TOKEN_BUFFER_SIZE", "10000"))
    
    def get_tool_execution_timeout(self) -> int:
        return int(os.getenv("TOOL_EXECUTION_TIMEOUT", "30"))
    
    def get_max_tool_loops(self) -> int:
        return int(os.getenv("MAX_TOOL_LOOPS", "3"))
    
    def get_tool_prefix(self) -> str:
        return os.getenv("TOOL_PREFIX", "__tool_")
        
    def get_async_timeout(self) -> int:
        return int(os.getenv("ASYNC_TIMEOUT", "60"))
        
    def get_thread_pool_workers(self) -> int:
        return int(os.getenv("THREAD_POOL_WORKERS", "4"))
