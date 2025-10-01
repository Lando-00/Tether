import asyncio
import inspect
import threading
import time
from typing import Optional, AsyncIterator

from llm_service.protocol.core.config import EnvironmentConfigProvider
from llm_service.protocol.core.interfaces import TokenSource, Logger, ConfigProvider
from llm_service.protocol.core.loggers import NoOpLogger
from llm_service.protocol.core.types import TokenDelta


class ModelTokenSource(TokenSource):
    """
    Adapts your model.generate(...) stream to TokenDelta.
    """
    def __init__(self, out_iter, logger: Optional[Logger] = None, config: Optional[ConfigProvider] = None):
        self.out_iter = out_iter
        self._logger = logger or NoOpLogger()
        self._config = config or EnvironmentConfigProvider()
        self._async_timeout = self._config.get_async_timeout()
        # current async iterator for streaming, used for abort
        self._current_aiter: Optional[AsyncIterator] = None

    def _is_async_iter(self, obj) -> bool:
        return hasattr(obj, "__aiter__") or inspect.isasyncgen(obj)

    def _aiter_sync(self, gen):
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=8)
        SENTINEL = object()
        stop = threading.Event()

        def produce():
            try:
                for item in gen:
                    if stop.is_set():
                        break
                    try:
                        loop.call_soon_threadsafe(q.put_nowait, item)
                    except asyncio.QueueFull:
                        self._logger.warning("Queue full, waiting before adding more items")
                        time.sleep(0.1)  # Brief pause to let queue clear
            except Exception as e:
                self._logger.exception("Error in producer thread: %s", str(e))
                try:
                    loop.call_soon_threadsafe(q.put_nowait, e)
                except:
                    pass  # If we can't queue the exception, we'll rely on timeouts
            finally:
                try:
                    loop.call_soon_threadsafe(q.put_nowait, SENTINEL)
                except:
                    pass  # Best effort

        t = threading.Thread(target=produce, name="sync-gen-producer", daemon=True)
        t.start()

        async def _aiter():
            try:
                while True:
                    try:
                        # Add timeout to prevent hanging
                        item = await asyncio.wait_for(q.get(), timeout=self._async_timeout)
                        if item is SENTINEL:
                            break
                        if isinstance(item, Exception):
                            raise item
                        yield item
                    except asyncio.TimeoutError:
                        self._logger.warning("Timeout waiting for next item in stream")
                        break
            finally:
                stop.set()
                try:
                    gen.close()
                except Exception as e:
                    self._logger.debug("Error closing generator: %s", str(e))
        return _aiter()

    async def stream(self) -> AsyncIterator[TokenDelta]:
        self._logger.debug("Starting token stream")
        try:
            aiter = self.out_iter if self._is_async_iter(self.out_iter) else self._aiter_sync(self.out_iter)
            # store for abort
            self._current_aiter = aiter
            
            # Wrap in timeout to avoid hanging
            async with asyncio.timeout(self._async_timeout):
                async for chunk in aiter:
                    try:
                        choice = chunk.choices[0] if hasattr(chunk, 'choices') and chunk.choices else None
                        if not choice:
                            self._logger.warning("Received chunk without choices")
                            continue
                            
                        delta = getattr(choice, "delta", None)
                        token = getattr(delta, "content", None)
                        finish = getattr(choice, "finish_reason", None)
                        
                        if finish:
                            self._logger.debug("Stream finished with reason: %s", finish)
                            
                        yield TokenDelta(token=token, finish_reason=finish)
                    except Exception as e:
                        self._logger.exception("Error processing chunk: %s", str(e))
                        # Continue streaming if possible
        except (asyncio.TimeoutError, asyncio.CancelledError) as e:
            self._logger.warning("Stream operation timed out or was cancelled: %s", str(e))
            yield TokenDelta(token=None, finish_reason="timeout")
        except Exception as e:
            self._logger.exception("Unexpected error in token stream: %s", str(e))
            yield TokenDelta(token=None, finish_reason="error")
        finally:
            # clear current iterator on completion
            self._current_aiter = None

    async def abort_current_stream(self) -> None:
        """
        Abort the current token stream to stop consuming further tokens.
        """
        aiter = self._current_aiter
        self._current_aiter = None
        # if async generator, close it
        if aiter and hasattr(aiter, 'aclose'):
            try:
                # type ignore: aclose may not be on generic AsyncIterator
                await (aiter.aclose())  # type: ignore
            except Exception:
                pass  # best effort