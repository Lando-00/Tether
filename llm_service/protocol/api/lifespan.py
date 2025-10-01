"""
Application lifespan management for FastAPI.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from llm_service.protocol.core.loggers import StandardLogger
from llm_service.protocol.core.config import EnvironmentConfigProvider
from llm_service.protocol.core.execution import ThreadPoolExecutionStrategy

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for resource cleanup."""
    # Initialize shared resources
    logger = StandardLogger("app_lifespan")
    app.state.logger = logger
    app.state.config = EnvironmentConfigProvider()
    
    # Create thread pool for tool execution
    workers = app.state.config.get_thread_pool_workers()
    logger.info("Initializing application with %d worker threads", workers)
    app.state.execution_strategy = ThreadPoolExecutionStrategy(max_workers=workers)
    
    try:
        logger.info("Application started")
        yield
    except Exception as e:
        logger.exception("Error during application lifecycle: %s", str(e))
    finally:
        # Clean up resources
        logger.info("Application shutting down, cleaning up resources")
        
        # Clean up execution strategy
        try:
            if hasattr(app.state, "execution_strategy"):
                app.state.execution_strategy.cleanup()
                logger.info("Thread pool execution strategy shut down")
        except Exception as e:
            logger.exception("Error cleaning up execution strategy: %s", str(e))
        
        # Clear model cache
        try:
            app.state.protocol.model.clear_engine_cache()
            logger.info("Model engine cache cleared")
        except Exception as e:
            logger.exception("Error clearing model cache: %s", str(e))