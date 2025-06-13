import importlib
import json
import re
import os
import sys
import time
import pandas as pd
import numpy as np
import importlib.util
import copy
import psutil
import signal
import threading
import queue
from collections import deque
from contextlib import contextmanager

import torch
from tqdm import tqdm
from typing import Union, Optional, List, Callable, Any, Tuple, Dict
import multiprocessing as mp
from framework.kernel_gen_config import config


class SegmentationFaultError(Exception):
    """Custom exception for segmentation fault errors."""
    pass


class ResourceExhaustionError(Exception):
    """Custom exception for resource exhaustion."""
    pass


class BatchProcessConfig:
    """Configuration class for batch processing parameters optimized for I/O intensive tasks."""

    def __init__(self):
        # Process management - optimized for I/O intensive workloads
        self.max_processes = 200  # Increased for I/O bound tasks
        self.min_processes = 10  # Higher minimum to maintain concurrency
        self.process_start_delay = 0.01  # Faster process startup

        # Thread awareness configuration
        self.threads_per_process = 8  # Expected threads per process (e.g., make -j8)
        self.thread_sleep_factor = 0.9  # High sleep factor for I/O waiting
        self.use_thread_aware_scaling = True  # Enable thread-aware process scaling

        # Resource monitoring - relaxed for systems with ample resources
        self.memory_threshold_gb = 1400000.0  # 1.4TB threshold
        self.cpu_threshold_percent = 99.0  # Very high threshold
        self.resource_check_interval = 5.0  # Less frequent checks
        self.resource_recovery_time = 2.0  # Faster recovery

        # Modified thresholds for thread-aware mode
        self.active_thread_threshold = 5000  # High threshold for active threads
        self.total_thread_threshold = 20000  # Very high total thread limit
        self.load_average_threshold = None  # Will be set based on CPU count

        # Timeout and retry
        self.timeout_seconds = 300
        self.process_cleanup_timeout = 5.0
        self.max_retries = 1
        self.retry_delay = 2.0

        # Queue management
        self.queue_size_limit = 10000
        self.batch_size = 100  # Process tasks in batches

        # Process monitoring
        self.health_check_interval = 30.0
        self.max_process_lifetime = 3600  # 1 hour max per process

        # File and memory management
        self.max_open_files = 10000
        self.gc_interval = 100  # Run garbage collection every N tasks

    def adjust_for_system(self):
        """Automatically adjust configuration based on system resources."""
        total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        cpu_count = mp.cpu_count()

        # Use 95% of available memory for I/O intensive tasks
        self.memory_threshold_gb = min(self.memory_threshold_gb, total_memory_gb * 0.95)

        # Adjust max processes based on CPU count and thread awareness
        if self.use_thread_aware_scaling:
            # For I/O intensive tasks, allow many more processes
            effective_cpu_multiplier = 1.0 / (1.0 - self.thread_sleep_factor)
            self.max_processes = min(
                self.max_processes,
                int(cpu_count * effective_cpu_multiplier / self.threads_per_process * 5)
            )
        else:
            # Allow 2x CPU count for non-thread-aware mode
            self.max_processes = min(self.max_processes, cpu_count * 2)

        self.min_processes = max(1, int(self.max_processes * 0.1))

        # Higher load average threshold for I/O intensive workloads
        self.load_average_threshold = cpu_count * 3.0

        # Much higher thread limits for I/O bound tasks
        self.active_thread_threshold = cpu_count * 200
        self.total_thread_threshold = cpu_count * 1000

        return self

    def to_dict(self):
        """Convert config to dictionary for safe serialization."""
        return {
            'max_processes': self.max_processes,
            'min_processes': self.min_processes,
            'process_start_delay': self.process_start_delay,
            'threads_per_process': self.threads_per_process,
            'thread_sleep_factor': self.thread_sleep_factor,
            'use_thread_aware_scaling': self.use_thread_aware_scaling,
            'memory_threshold_gb': self.memory_threshold_gb,
            'cpu_threshold_percent': self.cpu_threshold_percent,
            'resource_check_interval': self.resource_check_interval,
            'resource_recovery_time': self.resource_recovery_time,
            'active_thread_threshold': self.active_thread_threshold,
            'total_thread_threshold': self.total_thread_threshold,
            'load_average_threshold': self.load_average_threshold,
            'timeout_seconds': self.timeout_seconds,
            'process_cleanup_timeout': self.process_cleanup_timeout,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'queue_size_limit': self.queue_size_limit,
            'batch_size': self.batch_size,
            'health_check_interval': self.health_check_interval,
            'max_process_lifetime': self.max_process_lifetime,
            'max_open_files': self.max_open_files,
            'gc_interval': self.gc_interval
        }


class ProcessInfo:
    """Information about an active process."""

    def __init__(self, process, connection, start_time, task_idx):
        self.process = process
        self.connection = connection
        self.start_time = start_time
        self.task_idx = task_idx
        self.last_activity = start_time


class TaskQueue:
    """Manages task queue and retry logic."""

    def __init__(self, task_obj_list: List[Any], max_retries: int):
        self.task_obj_list = task_obj_list
        self.max_retries = max_retries
        self.pending_queue = deque(enumerate(task_obj_list))
        self.retry_queue = deque()
        self.retry_counts = {}  # {idx: retry_count}

    def has_pending_tasks(self) -> bool:
        """Check if there are any pending tasks."""
        return bool(self.pending_queue or self.retry_queue)

    def get_next_task(self) -> Optional[Tuple[int, Any]]:
        """Get the next task to process, prioritizing retries."""
        if self.retry_queue:
            idx, task_obj = self.retry_queue.popleft()
            return idx, task_obj
        elif self.pending_queue:
            idx, task_obj = self.pending_queue.popleft()
            self.retry_counts[idx] = 0
            return idx, task_obj
        return None

    def put_back_task(self, idx: int, task_obj: Any):
        """Put a task back to the appropriate queue."""
        if self.retry_counts.get(idx, 0) > 0:
            self.retry_queue.appendleft((idx, task_obj))
        else:
            self.pending_queue.appendleft((idx, task_obj))

    def should_retry(self, idx: int) -> bool:
        """Check if a task should be retried."""
        return self.retry_counts.get(idx, 0) < self.max_retries

    def increment_retry(self, idx: int) -> int:
        """Increment retry count for a task and return the new count."""
        self.retry_counts[idx] = self.retry_counts.get(idx, 0) + 1
        return self.retry_counts[idx]

    def queue_for_retry(self, idx: int):
        """Queue a task for retry."""
        if idx < len(self.task_obj_list):
            self.retry_queue.append((idx, self.task_obj_list[idx]))

    def get_retry_count(self, idx: int) -> int:
        """Get current retry count for a task."""
        return self.retry_counts.get(idx, 0)

    def pending_count(self) -> int:
        """Get total number of pending tasks."""
        return len(self.pending_queue) + len(self.retry_queue)


class ResourceMonitor:
    """Monitor system resources and manage process scaling."""

    def __init__(self, config_: BatchProcessConfig):
        self.config = config_
        self.last_check = 0
        self.resource_history = deque(maxlen=10)

    def check_resources(self) -> Tuple[bool, str]:
        """Check if system resources are available for new processes."""
        current_time = time.time()

        if current_time - self.last_check < self.config.resource_check_interval:
            return True, "OK"
        self.last_check = current_time

        can_proceed, reason = check_system_resources(self.config.to_dict())

        self.resource_history.append((current_time, can_proceed))
        return can_proceed, reason

    def get_optimal_process_count(self, current_count: int, pending_tasks: int) -> int:
        """Calculate optimal number of processes based on current resources."""
        can_proceed, reason = self.check_resources()

        if not can_proceed:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent < 30:
                return current_count
            else:
                recent_failures = sum(1 for _, success in self.resource_history if not success)
                if recent_failures > len(self.resource_history) * 0.7:
                    return max(self.config.min_processes, current_count // 2)
                else:
                    return current_count

        if self.config.use_thread_aware_scaling:
            return self._get_thread_aware_process_count(current_count, pending_tasks)
        else:
            if pending_tasks > current_count * 2:
                return min(self.config.max_processes, current_count + 1)
            return current_count

    def _get_thread_aware_process_count(self, current_count: int, pending_tasks: int) -> int:
        """Calculate process count with thread awareness."""
        active_threads, total_threads, load_avg = get_system_thread_stats()
        cpu_percent = psutil.cpu_percent(interval=0.1)

        if cpu_percent < 50 and pending_tasks > 0:
            increment = max(5, pending_tasks // 20)
            return min(self.config.max_processes, current_count + increment)

        thread_headroom = (self.config.total_thread_threshold - total_threads) / self.config.threads_per_process
        active_thread_headroom = (self.config.active_thread_threshold - active_threads) / (
                self.config.threads_per_process * (1 - self.config.thread_sleep_factor)
        )
        headroom = int(min(thread_headroom, active_thread_headroom))

        if headroom > 0 and pending_tasks > current_count:
            new_count = min(
                self.config.max_processes,
                current_count + max(1, headroom // 2),
                current_count + max(1, pending_tasks // 10)
            )
            return new_count
        elif headroom < -2:
            return max(self.config.min_processes, current_count - 1)

        return current_count


class ProcessManager:
    """Enhanced process manager with resource awareness and fault tolerance."""

    def __init__(self, config_: BatchProcessConfig):
        self.config = config_
        self.active_processes: Dict[int, ProcessInfo] = {}
        self.resource_monitor = ResourceMonitor(config_)
        self.adaptive_start_delay = config_.process_start_delay

    def can_start_new_process(self, pending_tasks: int) -> Tuple[bool, str]:
        """Check if we can start a new process based on current resources and limits."""
        current_count = len(self.active_processes)
        optimal_count = self.resource_monitor.get_optimal_process_count(current_count, pending_tasks)

        if current_count >= optimal_count:
            return False, "Process limit reached"

        can_proceed, reason = check_system_resources(self.config.to_dict())
        return can_proceed, reason

    def start_process(self, task_idx: int, worker_function: Callable,
                      task_obj: Any, other_args: dict, config_dict: dict) -> bool:
        """Start a new process for the given task."""
        parent_conn, child_conn = mp.Pipe()

        try:
            process = mp.Process(
                target=_process_wrapper,
                args=(_execute_task_robust,
                      (worker_function, task_obj, task_idx, other_args, config_dict),
                      child_conn),
                daemon=True
            )
            process.start()
            child_conn.close()

            self.active_processes[task_idx] = ProcessInfo(
                process=process,
                connection=parent_conn,
                start_time=time.time(),
                task_idx=task_idx
            )

            # Adjust start delay on success
            self.adaptive_start_delay = max(0.001, self.adaptive_start_delay * 0.95)
            return True

        except Exception as e:
            # Clean up connections on failure
            safe_close_connection(parent_conn)
            safe_close_connection(child_conn)
            # Increase start delay on failure
            self.adaptive_start_delay = min(1.0, self.adaptive_start_delay * 1.5)
            print(f"Failed to start process for task {task_idx}: {e}")
            return False

    def check_process_status(self, task_idx: int) -> Tuple[str, Any]:
        """Check the status of a process. Returns (status, result)."""
        if task_idx not in self.active_processes:
            return 'not_found', None

        process_info = self.active_processes[task_idx]
        process = process_info.process
        conn = process_info.connection
        current_time = time.time()

        # Check timeout
        if (current_time - process_info.start_time) > self.config.timeout_seconds:
            return 'timeout', TimeoutError(f"Task {task_idx} timed out after {self.config.timeout_seconds}s")

        # Check if process is still alive
        if process.is_alive():
            # Check for available results
            if safe_poll_connection(conn):
                result_data = safe_recv_connection(conn)
                if result_data is not None:
                    status, result = result_data
                    return 'completed', result
                else:
                    return 'error', RuntimeError(f"Failed to receive result for task {task_idx}")
            return 'running', None
        else:
            # Process has finished
            if process.exitcode == 0:
                # Try to get result
                if safe_poll_connection(conn):
                    result_data = safe_recv_connection(conn)
                    if result_data is not None:
                        status, result = result_data
                        return 'completed', result
                return 'error', RuntimeError(f"No result received for task {task_idx}")
            else:
                # Process crashed
                return 'crashed', SegmentationFaultError(
                    f"Process exited with code {process.exitcode} for task {task_idx}")

    def cleanup_process(self, task_idx: int, force: bool = False) -> bool:
        """Clean up a specific process safely."""
        if task_idx not in self.active_processes:
            return True

        process_info = self.active_processes[task_idx]
        process = process_info.process
        conn = process_info.connection

        try:
            # Close connection
            safe_close_connection(conn)

            # Terminate process
            if process.is_alive():
                if force:
                    safe_kill_process(process)
                else:
                    safe_terminate_process(process, self.config.process_cleanup_timeout)

            # Final cleanup
            if hasattr(process, 'close'):
                try:
                    process.close()
                except:
                    pass

        except Exception as e:
            print(f"Warning: Error cleaning up process {task_idx}: {e}")
        finally:
            # Always remove from tracking
            if task_idx in self.active_processes:
                del self.active_processes[task_idx]

        return True

    def cleanup_all(self):
        """Clean up all active processes."""
        indices = list(self.active_processes.keys())
        for idx in indices:
            self.cleanup_process(idx, force=True)

    def get_start_delay(self) -> float:
        """Get the current adaptive start delay."""
        return self.adaptive_start_delay

    def active_count(self) -> int:
        """Get the number of active processes."""
        return len(self.active_processes)

    def list_active_tasks(self) -> List[int]:
        """Get list of active task indices."""
        return list(self.active_processes.keys())


# Helper functions

def get_system_thread_stats() -> Tuple[int, int, float]:
    """Get system-wide thread statistics."""
    try:
        total_threads = 0
        active_threads = 0

        for proc in psutil.process_iter(['num_threads', 'status']):
            try:
                info = proc.info
                total_threads += info['num_threads']
                if info['status'] in [psutil.STATUS_RUNNING, psutil.STATUS_DISK_SLEEP]:
                    active_threads += info['num_threads']
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        load_average = os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0.0
        return active_threads, total_threads, load_average

    except Exception:
        return 0, 0, 0.0


def check_system_resources(config_dict: dict) -> Tuple[bool, str]:
    """Check if system resources are available for new processes."""
    try:
        # Check memory
        memory = psutil.virtual_memory()
        memory_gb = memory.used / (1024 ** 3)
        if memory_gb > config_dict['memory_threshold_gb']:
            return False, f"Memory usage too high: {memory_gb:.1f}GB"

        # Check CPU and threads
        if config_dict.get('use_thread_aware_scaling', False):
            active_threads, total_threads, load_avg = get_system_thread_stats()

            if total_threads > config_dict.get('total_thread_threshold', 20000):
                return False, f"Total thread count too high: {total_threads}"

            if active_threads > config_dict.get('active_thread_threshold', 5000):
                return False, f"Active thread count too high: {active_threads}"

            if load_avg > config_dict.get('load_average_threshold', mp.cpu_count() * 3):
                cpu_percent = psutil.cpu_percent(interval=0.1)
                if cpu_percent > 50:
                    return False, f"Load average too high with high CPU: {load_avg:.1f}"

        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_threshold = config_dict['cpu_threshold_percent']
        if config_dict.get('use_thread_aware_scaling', False):
            cpu_threshold = min(cpu_threshold * 1.2, 99.9)

        if cpu_percent > cpu_threshold:
            return False, f"CPU usage too high: {cpu_percent:.1f}%"

        return True, "OK"

    except Exception as e:
        return False, f"Resource check failed: {e}"


def safe_poll_connection(conn) -> bool:
    """Safely check if connection has data available."""
    try:
        if conn.closed:
            return False
        return conn.poll()
    except (OSError, ValueError, AttributeError):
        return False


def safe_recv_connection(conn) -> Optional[Tuple[str, Any]]:
    """Safely receive data from connection."""
    try:
        if conn.closed:
            return None
        return conn.recv()
    except (OSError, ValueError, AttributeError, EOFError):
        return None


def safe_close_connection(conn):
    """Safely close a connection."""
    if conn is not None and not conn.closed:
        try:
            conn.close()
        except:
            pass


def safe_terminate_process(process, timeout: float):
    """Safely terminate a process with timeout."""
    try:
        process.terminate()
        process.join(timeout)
        if process.is_alive():
            safe_kill_process(process)
    except:
        pass


def safe_kill_process(process):
    """Safely kill a process."""
    try:
        process.kill()
    except:
        pass


def _signal_handler(signum, frame):
    """Handle termination signals gracefully."""
    sys.exit(0)


def _process_wrapper(target_func, args, conn):
    """Wrapper to catch all exceptions in subprocess."""
    try:
        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)

        result = target_func(*args)
        conn.send(('success', result))
    except Exception as e:
        conn.send(('error', e))
    finally:
        safe_close_connection(conn)


def _execute_task_robust(worker_function, task_obj, idx, other_args, config_dict):
    """Execute task in subprocess with enhanced error handling."""
    try:
        start_time = time.time()
        result = worker_function(task_obj, idx, **other_args)

        execution_time = time.time() - start_time
        if execution_time > config_dict['timeout_seconds'] * 0.9:
            print(f"Warning: Task {idx} took {execution_time:.1f}s, close to timeout")

        return result
    except KeyboardInterrupt:
        raise KeyboardInterrupt("Task interrupted")
    except MemoryError:
        raise MemoryError(f"Out of memory in task {idx}")
    except Exception as e:
        raise e


def _handle_task_completion(task_obj: Any, result: Any, idx: int,
                            callback: Optional[Callable[[Any, Any, int], None]]):
    """Handle task completion with callback."""
    if callback:
        try:
            callback(task_obj, result, idx)
        except Exception as e:
            print(f"Warning: Callback failed for task {idx}: {e}")


def _process_completed_task(task_idx: int, result: Any, status: str,
                            results: List[Any], task_queue: TaskQueue,
                            task_obj_list: List[Any],
                            post_process_callback: Optional[Callable],
                            pbar: tqdm) -> bool:
    """Process a completed task and handle retries if needed. Returns True if task is done."""
    if status == 'completed':
        # Task completed successfully
        if isinstance(result, tuple) and result[0] == 'success':
            results[task_idx] = result[1]
        elif isinstance(result, tuple) and result[0] == 'error':
            # Error from subprocess
            error = result[1]
            if task_queue.should_retry(task_idx) and isinstance(error, (MemoryError, OSError)):
                retry_count = task_queue.increment_retry(task_idx)
                print(f"Retrying task {task_idx} after recoverable error (attempt {retry_count}): {error}")
                task_queue.queue_for_retry(task_idx)
                return False
            else:
                results[task_idx] = error
        else:
            results[task_idx] = result

    elif status in ['timeout', 'crashed', 'error']:
        # Handle failures
        if task_queue.should_retry(task_idx):
            retry_count = task_queue.increment_retry(task_idx)
            print(f"Retrying task {task_idx} after {status} (attempt {retry_count})")
            task_queue.queue_for_retry(task_idx)
            return False
        else:
            results[task_idx] = result
    else:
        # Unknown status
        results[task_idx] = RuntimeError(f"Unknown task status: {status}")

    # Task is done (either success or max retries reached)
    _handle_task_completion(task_obj_list[task_idx], results[task_idx], task_idx, post_process_callback)
    pbar.update(1)
    return True


def batch_execute_tests_base(
        task_obj_list: List[Any],
        worker_function: Callable[..., Any],
        batch_config: BatchProcessConfig = None,
        other_args: dict = None,
        tqdm_desc: str = "",
        post_process_callback: Callable[[Any, Any, int], None] = None,
        update_callback = None
) -> List[Any]:
    """
    Execute batch processing with enhanced resource management and fault tolerance.
    """
    if not task_obj_list:
        return []

    # Initialize configuration
    if batch_config is None:
        batch_config = BatchProcessConfig().adjust_for_system()

    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")

    if other_args is None:
        other_args = {}

    # Convert config to dict for safe serialization
    config_dict = batch_config.to_dict()

    # Initialize components
    task_queue = TaskQueue(task_obj_list, batch_config.max_retries)
    process_manager = ProcessManager(batch_config)
    results = [None] * len(task_obj_list)
    completed_count = 0

    try:
        with tqdm(total=len(task_obj_list), desc=tqdm_desc) as pbar:
            while task_queue.has_pending_tasks() or process_manager.active_count() > 0:

                # Start new processes if possible
                while task_queue.has_pending_tasks():
                    can_start, reason = process_manager.can_start_new_process(task_queue.pending_count())
                    if not can_start:
                        if "resource" in reason.lower():
                            time.sleep(batch_config.resource_recovery_time)
                        break

                    # Get next task
                    task_info = task_queue.get_next_task()
                    if task_info is None:
                        break

                    task_idx, task_obj = task_info

                    # Add start delay for I/O intensive tasks
                    if process_manager.active_count() > 0:
                        time.sleep(process_manager.get_start_delay())

                    # Try to start process
                    success = process_manager.start_process(
                        task_idx, worker_function, task_obj, other_args, config_dict
                    )

                    if not success:
                        # Put task back if we couldn't start the process
                        task_queue.put_back_task(task_idx, task_obj)
                        time.sleep(batch_config.retry_delay)
                        break

                # Check active processes
                for task_idx in process_manager.list_active_tasks():
                    status, result = process_manager.check_process_status(task_idx)

                    if status == 'running':
                        continue

                    # Process has finished or failed
                    task_done = _process_completed_task(
                        task_idx, result, status, results, task_queue,
                        task_obj_list, post_process_callback, pbar
                    )

                    if task_done:
                        completed_count += 1
                        if update_callback is not None:
                            update_callback(completed_count)

                        # Periodic garbage collection
                        if completed_count % batch_config.gc_interval == 0:
                            import gc
                            gc.collect()

                    # Clean up the process
                    process_manager.cleanup_process(task_idx, force=(status == 'timeout'))

                # Small delay to prevent CPU spinning
                time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nInterrupt received, cleaning up processes...")
        process_manager.cleanup_all()
        raise
    except Exception as e:
        print(f"Unexpected error in batch execution: {e}")
        process_manager.cleanup_all()
        raise
    finally:
        # Final cleanup
        process_manager.cleanup_all()

    return results


# Public interface - kept unchanged
def batch_execute_tests(
        task_obj_list: List[Any],
        worker_function: Callable[..., Any],
        timeout: int = 300,
        other_args: dict = None,
        tqdm_desc: str = "",
        post_process_callback: Callable[[Any, Any, int], None] = None,
        num_processes: int = 100,
        threads_per_process: int = 8,
        enable_thread_aware_scaling: bool = True,
        update_callback=None
):
    """
    Execute batch tests with optimized settings for I/O intensive workloads.

    This is the main public interface that maintains backward compatibility.
    """
    batch_config = BatchProcessConfig()
    batch_config.timeout_seconds = timeout
    batch_config.max_processes = num_processes
    batch_config.threads_per_process = threads_per_process
    batch_config.use_thread_aware_scaling = enable_thread_aware_scaling
    batch_config.adjust_for_system()

    print("Batch execution config:")
    print(json.dumps(batch_config.to_dict(), indent=2))

    return batch_execute_tests_base(
        task_obj_list=task_obj_list,
        worker_function=worker_function,
        batch_config=batch_config,
        other_args=other_args,
        tqdm_desc=tqdm_desc,
        post_process_callback=post_process_callback,
        update_callback=update_callback
    )