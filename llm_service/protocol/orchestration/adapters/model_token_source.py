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
        
        # Lock to prevent multiple threads from iterating the same generator
        producer_lock = threading.Lock()

        def produce():
            try:
                # Use the lock to prevent concurrent iteration
                if producer_lock.acquire(blocking=False):
                    try:
                        for item in gen:
                            if stop.is_set():
                                break
                            try:
                                loop.call_soon_threadsafe(q.put_nowait, item)
                            except asyncio.QueueFull:
                                self._logger.warning("Queue full, waiting before adding more items")
                                time.sleep(0.1)  # Brief pause to let queue clear
                    finally:
                        producer_lock.release()
                else:
                    # If we couldn't acquire the lock, another thread is already iterating
                    self._logger.warning("Generator is already being iterated, cannot start another iteration")
                    loop.call_soon_threadsafe(q.put_nowait, 
                                            ValueError("Generator already executing"))
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
        # Track if we're already streaming to prevent concurrent access
        stream_id = f"stream_{int(time.time() * 1000)}"
        self._logger.debug(f"Creating stream {stream_id}")
        
        try:
            # Create the appropriate iterator
            if self._is_async_iter(self.out_iter):
                self._logger.debug(f"Stream {stream_id}: Using async iterator directly")
                aiter = self.out_iter
            else:
                self._logger.debug(f"Stream {stream_id}: Adapting sync iterator")
                aiter = self._aiter_sync(self.out_iter)
                
            # Store for abort - do this atomically to prevent race conditions
            self._current_aiter = aiter
            
            # Wrap in timeout to avoid hanging
            async with asyncio.timeout(self._async_timeout):
                try:
                    async for chunk in aiter:
                        try:
                            # Check if we've been cleared (aborted)
                            if self._current_aiter is None:
                                self._logger.warning(f"Stream {stream_id}: Detected abort during iteration")
                                break
                                
                            choice = chunk.choices[0] if hasattr(chunk, 'choices') and chunk.choices else None
                            if not choice:
                                self._logger.warning(f"Stream {stream_id}: Received chunk without choices")
                                continue
                                
                            delta = getattr(choice, "delta", None)
                            token = getattr(delta, "content", None)
                            finish = getattr(choice, "finish_reason", None)
                            if finish == "abort":
                                finish = "error"
                            
                            if finish:
                                self._logger.debug(f"Stream {stream_id}: Finished with reason: {finish}")
                                
                            yield TokenDelta(token=token, finish_reason=finish)
                        except Exception as e:
                            self._logger.exception(f"Stream {stream_id}: Error processing chunk: {str(e)}")
                            # Continue streaming if possible
                except asyncio.CancelledError:
                    self._logger.warning(f"Stream {stream_id}: Cancelled during iteration")
                    raise
        except (asyncio.TimeoutError, asyncio.CancelledError) as e:
            self._logger.warning(f"Stream {stream_id}: Timed out or cancelled: {str(e)}")
            yield TokenDelta(token=None, finish_reason="timeout")
        except ValueError as e:
            if "generator already executing" in str(e):
                self._logger.warning(f"Stream {stream_id}: Generator already executing error")
                # Try to recover by yielding a special token
                yield TokenDelta(token="\n[The AI is already processing another request]", finish_reason="error")
            else:
                self._logger.exception(f"Stream {stream_id}: ValueError: {str(e)}")
                yield TokenDelta(token=None, finish_reason="error")
        except Exception as e:
            self._logger.exception(f"Stream {stream_id}: Unexpected error: {str(e)}")
            yield TokenDelta(token=None, finish_reason="error")
        finally:
            # Clear current iterator only if it's still the one we set
            if self._current_aiter is aiter:
                self._logger.debug(f"Stream {stream_id}: Clearing current iterator reference")
                self._current_aiter = None

    async def abort_current_stream(self) -> None:
        """
        Abort the current token stream to stop consuming further tokens.
        """
        # Store the current iterator and immediately clear it to prevent reuse
        aiter = self._current_aiter
        self._current_aiter = None
        
        # Don't do anything if there's no active iterator
        if not aiter:
            self._logger.debug("No active stream to abort")
            return
            
        # For async generators, use aclose
        if hasattr(aiter, 'aclose'):
            try:
                self._logger.debug("Aborting async stream with aclose()")
                # type ignore: aclose may not be on generic AsyncIterator
                await asyncio.wait_for(aiter.aclose(), timeout=2.0)  # type: ignore
                self._logger.debug("Stream successfully aborted with aclose()")
            except asyncio.TimeoutError:
                self._logger.warning("Timeout while aborting stream")
            except Exception as e:
                self._logger.warning(f"Error aborting stream with aclose(): {str(e)}")
        else:
            # For other types of iterators, just let them be garbage collected
            self._logger.debug("Stream doesn't have aclose() method, relying on garbage collection")
            
        # Add a small delay to allow any pending operations to complete
        await asyncio.sleep(0.1)