import os
import sys
import pandas as pd
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from framework.logger import init_logger, add_file_handler, common_logger
from framework.task_object import TaskObject, PerformanceResult
from framework.kernel_gen_config import config
from framework.utils import preprocess_dataframe, load_module_from_path
from framework.batch_utils import batch_execute_tests


class PerformanceTester:
    """Helper class for performance testing operations."""

    def __init__(self, device):
        self.device = device

    def warmup_and_measure(self, model, inputs, num_trials):
        """Warmup model and measure average execution time."""
        # Warmup run
        with torch.no_grad():
            _ = model(*inputs)
        torch.cuda.synchronize(self.device)

        # Timing measurement
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            for _ in range(num_trials):
                _ = model(*inputs)
        end.record()
        torch.cuda.synchronize(self.device)

        return start.elapsed_time(end) / num_trials


def run_performance_single(task_obj: TaskObject, idx: int = 0, logger=None) -> PerformanceResult:
    """Execute performance test for a single task object."""
    num_gpu_devices = len(config.eval.gpu_devices)
    device_id = config.eval.gpu_devices[idx % num_gpu_devices]
    success = True
    n_failed = 0
    n_success = 0

    if logger is None:
        logger = init_logger(__name__ + f"{task_obj.short_id}")
        add_file_handler(log_file_path=task_obj.eval_perf_log_path, level="DEBUG", logger=logger)
        print(f"log_file_path = {task_obj.eval_perf_log_path}")

    try:
        # Setup environment and device
        so_dir = os.path.abspath(task_obj.build_path)
        sys.path.append(so_dir)
        task_obj.set_custom_opp_path()

        device = torch.device(f"cuda:{device_id}")
        torch.npu.set_device(device_id)
        logger.info(f"Start performance test of {task_obj.short_id} on device {device_id}")

        # Load modules
        if not os.path.exists(task_obj.module_path):
            logger.error(f"module.py not found: {task_obj.module_path}")
            raise FileNotFoundError(f"Module file not found: {task_obj.module_path}")

        module = load_module_from_path("model_module", task_obj.module_path)
        if not hasattr(module, "get_init_inputs"):
            prepare_inputs = load_module_from_path("prepare_inputs_module", task_obj.prepare_inputs_path)
        else:
            prepare_inputs = module

        if not os.path.exists(task_obj.test_cases_path):
            logger.error(f"test_cases.csv not found: {task_obj.test_cases_path}")
            raise FileNotFoundError(f"Test cases file not found: {task_obj.test_cases_path}")

        # Process test cases
        df = pd.read_csv(task_obj.test_cases_path, sep=';')
        df = preprocess_dataframe(df)
        results_dict = []

        perf_tester = PerformanceTester(device)

        for i, row in df.iterrows():
            if config.static_shape_mode and row['case_id'] != task_obj.case_id:
                continue

            try:
                # Prepare models and inputs
                init_params = prepare_inputs.get_init_inputs(row, device=device)
                inputs = prepare_inputs.get_inputs(row, device=device)

                model = module.Model(*init_params).to(device)
                model_new = module.ModelNew(*init_params).to(device)

                # Measure performance
                avg_model_time = perf_tester.warmup_and_measure(
                    model, inputs, config.eval.num_perf_trials
                )
                avg_model_new_time = perf_tester.warmup_and_measure(
                    model_new, inputs, config.eval.num_perf_trials
                )

                n_success += 1

                results_dict.append(dict(
                    **row,
                    avg_model_time=avg_model_time,
                    avg_model_new_time=avg_model_new_time
                ))

                logger.info(f"PerfTest op_name={task_obj.op_name}, sample_id={task_obj.sample_id}, "
                           f"testcase={i}, model_time={avg_model_time:.3f}ms, "
                           f"model_new_time={avg_model_new_time:.3f}ms")

            except Exception as e:
                success = False
                n_failed += 1
                logger.error(f"Test case {i}/{len(df)} failed: {e}", exc_info=True)
                continue

        result = PerformanceResult(
            success=success,
            n_success=n_success,
            n_failed=n_failed,
            log_file=task_obj.eval_perf_log_path,
            results_dict=results_dict
        )
        logger.debug(f"Task {task_obj.short_id} completed, result:\n{result}")
        return result

    except Exception as e:
        logger.error(f"Performance test failed: {e}", exc_info=True)
        return PerformanceResult(
            success=False,
            n_success=n_success,
            n_failed=n_failed,
            log_file=task_obj.eval_perf_log_path,
            results_dict=[]
        )


def post_performance_callback(task_obj, result_or_exception, idx):
    """
    Post-processing callback for performance tests.
    Stores results in task object and handles exceptions.
    """
    if isinstance(result_or_exception, PerformanceResult):
        task_obj.perf_result = result_or_exception
    else:
        if isinstance(result_or_exception, Exception):
            common_logger.error(result_or_exception, exc_info=True)
        task_obj.perf_result = PerformanceResult(
            success=False,
            log_file=task_obj.eval_perf_log_path,
            n_success=0,
            n_failed=0,
            results_dict=[]
        )


def batch_performance_test(task_obj_list, num_processes=1, timeout=300):
    """Execute performance tests for multiple task objects in parallel."""
    results = batch_execute_tests(
        task_obj_list=task_obj_list,
        worker_function=run_performance_single,
        num_processes=num_processes,
        timeout=timeout,
        tqdm_desc="Batch Performance Test",
        post_process_callback=post_performance_callback
    )
    return results