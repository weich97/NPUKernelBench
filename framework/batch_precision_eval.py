import os
import sys
import pandas as pd
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from framework.logger import init_logger, add_file_handler, common_logger
from framework.task_object import TaskObject, PrecisionResult
from framework.kernel_gen_config import config
from framework.utils import preprocess_dataframe, load_module_from_path, check_precision
from framework.batch_utils import batch_execute_tests
import copy


def deep_copy_tensors(obj):
    """Create a deep copy of the object, handling tensors and nested structures."""
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    elif isinstance(obj, (tuple, list)):
        return type(obj)(deep_copy_tensors(item) for item in obj)
    elif isinstance(obj, dict):
        return {k: deep_copy_tensors(v) for k, v in obj.items()}
    else:
        return obj


def transfer_to_device(params, device='cpu'):
    """Transfer parameters to specified device recursively."""
    if params is None:
        return None
    elif isinstance(params, torch.Tensor):
        return params.to(device)
    elif isinstance(params, torch.nn.Module):
        return params.to(device)
    elif isinstance(params, (list, tuple)):
        return type(params)(transfer_to_device(x, device) for x in params)
    elif isinstance(params, dict):
        return {k: transfer_to_device(v, device) for k, v in params.items()}
    else:
        return params


def run_precision_single(task_obj: TaskObject, idx: int = 0, logger=None) -> PrecisionResult:
    """Execute precision test for a single task object."""
    num_gpu_devices = len(config.eval.gpu_devices)
    device_id = config.eval.gpu_devices[idx % num_gpu_devices]
    n_success = 0
    n_failed = 0

    if logger is None:
        logger = init_logger(__name__ + f"{task_obj.short_id}")
        add_file_handler(log_file_path=task_obj.eval_precision_log_path, level="DEBUG", logger=logger)
        print(f"log_file_path = {task_obj.eval_precision_log_path}")

    try:
        # Setup environment and device
        so_dir = os.path.abspath(task_obj.build_path)
        sys.path.append(so_dir)
        task_obj.set_custom_opp_path()

        torch.npu.set_device(device_id)
        device = torch.device(f"cuda:{device_id}")
        logger.info(f"Start precision test of {task_obj.short_id} on device {device_id}")

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

        for i, row in df.iterrows():
            if config.static_shape_mode and row['case_id'] != task_obj.case_id:
                continue

            try:
                # Prepare inputs and models
                init_params_npu = prepare_inputs.get_init_inputs(row, device=device)
                init_params_cpu = transfer_to_device(init_params_npu, 'cpu')
                inputs_npu = prepare_inputs.get_inputs(row, device=device)
                inputs_cpu = transfer_to_device(inputs_npu, 'cpu')

                model = module.Model(*init_params_cpu).to('cpu')
                model_new = module.ModelNew(*init_params_npu).to(device)

                # Run precision tests
                all_abs_diff, all_rel_diff, all_acc = [], [], []

                n_acc = 0
                for trial_idx in range(config.eval.num_correct_trials):
                    # Execute reference model
                    inputs_copy = deep_copy_tensors(inputs_cpu)
                    ref_output = model(*inputs_copy)
                    torch_npu.npu.synchronize()
                    # Execute test model
                    inputs_copy = deep_copy_tensors(inputs_npu)
                    test_output = model_new(*inputs_copy)
                    torch_npu.npu.synchronize()
                    # Check precision
                    test_output = transfer_to_device(test_output, 'cpu')
                    if hasattr(prepare_inputs, 'custom_check_precision'):
                        logger.debug("Using custom precision check...")
                        acc, abs_diff, rel_diff = prepare_inputs.custom_check_precision(row, ref_output,
                                                                                        test_output)
                    else:
                        acc, abs_diff, rel_diff = check_precision(
                            ref_output, test_output,
                            config.eval.max_abs_error, config.eval.max_rel_error
                        )

                    n_acc += acc
                    all_acc.append(acc)
                    all_abs_diff.append(abs_diff)
                    all_rel_diff.append(rel_diff)

                    # Log detailed results for first trial
                    if config.eval.verbose:
                        _log_output_details(logger, inputs_cpu, ref_output, test_output, abs_diff, rel_diff)

                # Determine success for this test case
                if n_acc == config.eval.num_correct_trials:
                    n_success += 1
                else:
                    n_failed += 1

                # Calculate statistics
                all_abs_cat = torch.cat(all_abs_diff, dim=0)
                all_rel_cat = torch.cat(all_rel_diff, dim=0)

                stats = {
                    'avg_abs_error': all_abs_cat.float().mean().item(),
                    'avg_rel_error': all_rel_cat.float().mean().item(),
                    'max_abs_error': all_abs_cat.max().item(),
                    'max_rel_error': all_rel_cat.max().item(),
                    'avg_acc': torch.mean(torch.Tensor(all_acc).float()).item()
                }

                logger.info(f"[INFO] PrecTest op_name={task_obj.op_name}, sample_id={task_obj.sample_id}, "
                            f"testcase={i}, avg_abs_error={stats['avg_abs_error']:.4f}, "
                            f"avg_rel_error={stats['avg_rel_error']:.4f}, max_abs_error={stats['max_abs_error']:.4f}, "
                            f"max_rel_error={stats['max_rel_error']:.4f}, avg_acc={stats['avg_acc']:.2f}")

                results_dict.append(dict(**row, **stats))

            except Exception as e:
                logger.error(f"Test case {i}/{len(df)} failed: {e}", exc_info=True)
                n_failed += 1
                continue

        # Determine overall success
        if config.static_shape_mode:
            success = (n_success == 1)
        else:
            success = (n_success == len(df))

        result = PrecisionResult(
            success=success,
            n_success=n_success,
            n_failed=n_failed,
            log_file=task_obj.eval_precision_log_path,
            results_dict=results_dict
        )
        logger.debug(f"Task {task_obj.short_id} completed, result:\n{result}")
        return result

    except Exception as e:
        logger.error(f"Precision test failed: {e}", exc_info=True)
        return PrecisionResult(
            success=False,
            n_success=n_success,
            n_failed=n_failed,
            log_file=task_obj.eval_precision_log_path,
            results_dict=[]
        )


def _log_output_details(logger, inputs, ref_output, test_output, abs_diff, rel_diff):
    """Log detailed output information for debugging."""

    def log_tensor_info(output, name):
        if isinstance(output, (list, tuple)):
            for idx, t in enumerate(output):
                if hasattr(t, 'shape'):
                    logger.debug(f"======{name}[{idx}]======shape={t.shape}==========\n{t}")
                else:
                    logger.debug(f"======{name}[{idx}]======type={type(t)}==========\n{t}")
                if isinstance(t, torch.Tensor):
                    tensor_to_check = t.coalesce().values() if t.is_sparse else t
                    has_non_finite = torch.any(~torch.isfinite(tensor_to_check))
                    logger.debug(f"======{name}[{idx}]======has_non_finite=========={has_non_finite}")
        else:
            if hasattr(output, 'shape'):
                logger.debug(f"======{name}======shape={output.shape}==========\n{output}")
            else:
                logger.debug(f"======{name}======type={type(output)}==========\n{output}")
            if isinstance(output, torch.Tensor):
                tensor_to_check = output.coalesce().values() if output.is_sparse else output
                has_non_finite = torch.any(~torch.isfinite(tensor_to_check))
                logger.debug(f"======{name}======has_non_finite=========={has_non_finite}")

    log_tensor_info(inputs, "inputs")
    log_tensor_info(ref_output, "ref_output")
    log_tensor_info(test_output, "test_output")
    logger.debug(f"======abs_diff===max={abs_diff.float().max()}==========")
    logger.debug(f"======rel_diff===max={rel_diff.float().max()}==========")


def post_precision_callback(task_obj: TaskObject, result_or_exception, idx):
    """
    Post-processing callback for precision tests.
    Stores results in task object and handles exceptions.
    """
    if isinstance(result_or_exception, PrecisionResult):
        task_obj.precision_result = result_or_exception
    else:
        if isinstance(result_or_exception, Exception):
            common_logger.error(result_or_exception, exc_info=True)
        task_obj.precision_result = PrecisionResult(
            success=False,
            n_success=0,
            n_failed=0,
            log_file=task_obj.eval_precision_log_path,
            results_dict=[]
        )


def batch_precision_test(task_obj_list, num_processes=1, timeout=300, update_callback=None):
    """Execute precision tests for multiple task objects in parallel."""
    results = batch_execute_tests(
        task_obj_list=task_obj_list,
        worker_function=run_precision_single,
        num_processes=num_processes,
        timeout=timeout,
        tqdm_desc="Batch Precision Test",
        post_process_callback=post_precision_callback,
        update_callback=update_callback
    )
    return results
