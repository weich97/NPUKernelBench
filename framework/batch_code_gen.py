import os
from typing import Dict

from framework.logger import init_logger, add_file_handler, common_logger
from framework.batch_utils import batch_execute_tests
from framework.task_object import TaskObject, CodeGenResult


def code_gen_single_sample(task_obj: TaskObject, idx: int = 0, logger=None,
                           code_gen_class=None, code_gen_args=None, postfix="") -> CodeGenResult:
    """Generate code for a single sample using the specified code generation class."""
    try:
        os.makedirs(task_obj.log_path, exist_ok=True)
        task_obj.log_prefix = f"_{code_gen_class.__name__}_{postfix}"

        logger = init_logger(__name__ + f"_{task_obj.short_id}{task_obj.log_prefix}")
        add_file_handler(log_file_path=task_obj.code_gen_log_path, level="DEBUG", logger=logger)
        print(f"log_file_path = {task_obj.code_gen_log_path}")

        # Initialize code generator with logger
        code_gen_args['logger'] = logger
        code_gen = code_gen_class(task_obj, **code_gen_args)

        # Generate code
        response = code_gen.process()
        logger.info(f"[SUCCESS] Code generation completed for {task_obj.short_id}")
        success = True

    except Exception as e:
        logger.error(f"Code generation failed for {task_obj.short_id}: {e}", exc_info=True)
        success = False
        response = None

    return CodeGenResult(success, log_file=task_obj.code_gen_log_path, response=response)


def post_code_gen_callback(task_obj: TaskObject, result_or_exception, idx: int):
    """
    Post-processing callback for code generation.
    Required for multiprocessing to copy results from subprocess to main process.
    """
    if isinstance(result_or_exception, CodeGenResult):
        task_obj.code_gen_result = result_or_exception
        task_obj.gen_content = result_or_exception.response
    else:
        if isinstance(result_or_exception, Exception):
            common_logger.error(result_or_exception, exc_info=True)
        task_obj.code_gen_result = CodeGenResult(False, log_file=task_obj.code_gen_log_path, response=None)


def batch_code_gen(task_obj_list: list[TaskObject], num_processes: int = 2, timeout: int = 300,
                   code_gen_args=None, code_gen_class=None, postfix=""):
    """Execute code generation for multiple task objects in parallel."""
    other_args = {
        "postfix": postfix,
        "code_gen_args": code_gen_args,
        "code_gen_class": code_gen_class
    }

    results = batch_execute_tests(
        task_obj_list=task_obj_list,
        worker_function=code_gen_single_sample,
        num_processes=num_processes,
        timeout=timeout,
        other_args=other_args,
        tqdm_desc="Batch Code Generation Progress",
        post_process_callback=post_code_gen_callback
    )
    return results


def group_task_objects_by_problem(task_obj_list: list[TaskObject]) -> Dict[tuple, list[TaskObject]]:
    """Group task objects by problem definition path and case ID."""
    results = {}
    for task_obj in task_obj_list:
        key = (task_obj.problem_def_full_path, task_obj.case_id)
        if key in results:
            results[key].append(task_obj)
        else:
            results[key] = [task_obj]
    return results