import os
import shutil
import subprocess
import logging
from typing import List

from framework.logger import init_logger, add_file_handler, common_logger
from framework.kernel_gen_config import config
from framework.utils import safe_write_file
from framework.batch_utils import batch_execute_tests
from framework.task_object import TaskObject, CompileResult

# Constants for string replacements
CUSTOM_OP_REPLACEMENTS = {
    'aclnnCustomOp': 'aclnn{}',
    'custom_pybind_api': '{}',
    'CustomOp': '{}',
    'custom_op': '{}'
}


def run_subprocess_command(cmd, logger, error_prefix="", env_vars=None):
    """
    Execute subprocess command with optional environment variables.
    Raises RuntimeError if process returns non-zero exit code.
    """
    logger.info(f"[CMD]: {' '.join(cmd)}")

    # Prepare environment variables
    process_env = os.environ.copy() if env_vars else None
    if env_vars:
        process_env.update(env_vars)

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
        executable="/bin/bash",
        env=process_env
    )

    # Log output
    if proc.stdout:
        logger.debug(proc.stdout)
    if proc.stderr:
        if proc.returncode != 0:
            logger.error(proc.stderr)
        else:
            logger.warning(proc.stderr)

    if proc.returncode != 0:
        raise RuntimeError(f"{error_prefix} failed with exit code {proc.returncode}")

    return proc


def replace_template_strings(file_path, replacements, logger):
    """Replace template strings in a file with actual values."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        updated_content = content
        for old_str, new_str in replacements.items():
            if old_str in updated_content:
                updated_content = updated_content.replace(old_str, new_str)
                logger.info(f"Replaced '{old_str}' with '{new_str}' in {file_path}")

        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(updated_content)

    except Exception as e:
        logger.error(f"Failed to update file {file_path}: {e}")
        raise


def validate_single_file_in_directory(directory, expected_file):
    """Check if directory contains only the expected file."""
    if not os.path.exists(directory):
        return False

    items = os.listdir(directory)
    return len(items) == 1 and items[0] == expected_file


def prepare_source_files(task_obj: TaskObject, logger):
    """
    Copy template files and generated content to source directory.
    """
    src_path = task_obj.src_path
    logger.info(f"Preparing source files in {src_path}")

    # Create source directory
    os.makedirs(src_path, exist_ok=True)

    # Copy common files
    common_src = f"{config.compile.common_dir}/{config.compile.mode}"
    shutil.copytree(common_src, src_path, dirs_exist_ok=True)

    # Copy validation files
    validation_src = f"{task_obj.problem_def_full_path}/validation"
    validation_dst = f"{src_path}/validation"
    shutil.copytree(validation_src, validation_dst, dirs_exist_ok=True)

    # Update pybind file with custom op names
    pybind_file = f"{src_path}/validation/pybind.cpp"
    pybind_replacements = {
        'aclnnCustomOp': f'aclnn{task_obj.op_name}',
        'custom_pybind_api': task_obj.kernel_name
    }
    replace_template_strings(pybind_file, pybind_replacements, logger)

    if config.chat.active:
        # Generate content from chat
        if not task_obj.gen_content:
            task_obj.load_code_gen_content(logger)

        task_obj.parse_gen_content()

        files_to_write = [
            (f"op_host/{task_obj.kernel_name}.cpp", task_obj.tiling_content),
            (f"op_kernel/{task_obj.kernel_name}.cpp", task_obj.kernel_content)
        ]

        for file_name, content in files_to_write:
            if not content:
                raise RuntimeError(f"Empty content for {file_name}")
            safe_write_file(os.path.join(src_path, file_name), content)
    else:
        # Copy existing source files
        if config.compile.mode == "cann-ops":
            ops_src = task_obj.fake_path
            ops_dst = f"{src_path}/src/ops/{task_obj.kernel_name}"
            shutil.copytree(ops_src, ops_dst, dirs_exist_ok=True)

            # Update CMakeLists.txt
            cmake_file = f"{ops_dst}/CMakeLists.txt"
            cmake_replacements = {
                'CustomOp': task_obj.op_name,
                'custom_op': task_obj.kernel_name
            }
            replace_template_strings(cmake_file, cmake_replacements, logger)
        else:
            question_src = f"{task_obj.problem_def_full_path}/question"
            shutil.copytree(question_src, src_path, dirs_exist_ok=True)

            # Validate file structure
            host_dir = f"{question_src}/op_host"
            kernel_dir = f"{question_src}/op_kernel"

            if (not validate_single_file_in_directory(host_dir, f"{task_obj.kernel_name}.cpp") or
                not validate_single_file_in_directory(kernel_dir, f"{task_obj.kernel_name}.cpp")):
                raise RuntimeError(f"Invalid file structure in {question_src}")

    # Update CMakePresets.json
    cmake_file = f"{src_path}/CMakePresets.json"
    cmake_replacements = {
        '/usr/local/Ascend/ascend-toolkit/latest': os.environ['ASCEND_HOME_PATH']
    }
    replace_template_strings(cmake_file, cmake_replacements, logger)


def compile_single_sample(task_obj: TaskObject, idx: int = 0, verbose: bool = False, logger=None) -> CompileResult:
    """Compile a single sample with complete workflow."""
    # Setup directories
    try:
        os.makedirs(task_obj.build_path, exist_ok=True)
        os.makedirs(task_obj.log_path, exist_ok=True)

        if logger is None:
            logger = init_logger(__name__ + task_obj.short_id)
            add_file_handler(log_file_path=task_obj.compile_log_path, level="DEBUG", logger=logger)
            print(f"log_file_path = {task_obj.compile_log_path}")

        logger.info(f"Starting compilation: {task_obj.short_id}")

        # Prepare source files
        prepare_source_files(task_obj, logger)

        # Execute compilation
        old_cwd = os.getcwd()
        os.chdir(task_obj.src_path)

        try:
            # Clean previous build
            if os.path.exists("CMakeCache.txt"):
                os.remove("CMakeCache.txt")

            # Run build script
            run_subprocess_command(["bash build.sh"], logger, "Compilation")

        finally:
            os.chdir(old_cwd)

        # Verify compilation success
        success, kernel_gen_exists, custom_opapi_exists, kernel_exists = task_obj.check_compile_success()

        if success:
            logger.info(f"[SUCCESS] Compiled: {task_obj.op_name}, Sample={task_obj.sample_id}")
        else:
            logger.error(f"[FAILED] Compilation failed for {task_obj.op_name}, Sample={task_obj.sample_id}")
            logger.error(f"kernel_gen_ops*.so exists: {kernel_gen_exists}")
            logger.error(f"libcust_opapi.so exists: {custom_opapi_exists}")
            logger.error(f"{task_obj.op_name}*.o exists: {kernel_exists}")

    except Exception as e:
        logger.error(f"[FAILED] Compilation error for {task_obj.op_name}, Sample={task_obj.sample_id}: {e}",
                    exc_info=True)
        success = False

    return CompileResult(success, log_file=task_obj.compile_log_path)


def post_compile_callback(task_obj: TaskObject, result_or_exception, idx: int, verbose=False):
    """
    Post-compilation callback to handle results.
    Required for multiprocessing to copy results from subprocess to main process.
    """
    try:
        if isinstance(result_or_exception, CompileResult):
            task_obj.compile_result = result_or_exception
        else:
            if isinstance(result_or_exception, Exception):
                common_logger.error(f"Task {idx} failed with exception: {result_or_exception}", exc_info=True)

            # 创建失败的编译结果
            log_file = getattr(task_obj, 'compile_log_path', None)
            task_obj.compile_result = CompileResult(False, log_file=log_file)

        # 记录编译状态
        status = "SUCCESS" if task_obj.compile_result.success else "FAILED"
        common_logger.info(f"Task {idx} ({task_obj.short_id}): {status}")

    except Exception as e:
        common_logger.error(f"Error in post-compile callback for task {idx}: {e}", exc_info=True)
        task_obj.compile_result = CompileResult(False, log_file=None)


def batch_compile(task_obj_list: List[TaskObject], num_processes: int = 2, timeout: int = 300, verbose: bool = False):
    """Execute compilation for multiple task objects in parallel."""
    return batch_execute_tests(
        task_obj_list=task_obj_list,
        worker_function=compile_single_sample,
        num_processes=num_processes,
        timeout=timeout,
        other_args={"verbose": verbose},
        tqdm_desc="Batch Compile Progress",
        post_process_callback=post_compile_callback
    )