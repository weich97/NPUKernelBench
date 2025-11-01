import glob
import os
import re
import json
import torch
import shutil
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass, field, fields
from framework.kernel_gen_config import config
from framework.kernel_parser import KernelParser
from framework.utils import safe_read_file, save_json_file


def enhanced_dataclass(cls=None):
    """Enhanced dataclass decorator with custom string representation."""

    def wrap(cls):
        cls = dataclass(cls)

        def custom_repr(self):
            result = ["=" * 150, f"{self.__class__.__name__}:"]

            for field_info in fields(self):
                field_value = getattr(self, field_info.name)

                if field_info.name == "results_dict":
                    with pd.option_context('display.max_rows', None, 'display.max_columns', None,
                                           'display.width', None, 'display.max_colwidth', None):
                        result.append(f"{field_info.name}:\n{pd.DataFrame(field_value)}")
                elif field_info.name in ["success", "n_success", "n_failed"]:
                    result.append(f"{field_info.name} = {field_value}")
                elif field_info.name == "log_file" and config.show_result_log_file:
                    try:
                        with open(field_value, "r", encoding='utf-8') as f:
                            result.append(f"{field_info.name} = {field_value}:\n{f.read()}")
                    except FileNotFoundError:
                        result.append(f"{field_info.name} = {field_value} (file not found)")
                else:
                    result.append(f"{field_info.name}:\n{field_value}")

            result.append("=" * 150)
            return "\n".join(result)

        cls.__repr__ = custom_repr
        cls.__str__ = custom_repr
        return cls

    return wrap if cls is None else wrap(cls)


@enhanced_dataclass
class CodeGenResult:
    """Result of code generation process."""
    success: bool
    log_file: str
    response: str


@enhanced_dataclass
class CompileResult:
    """Result of compilation process."""
    success: bool
    log_file: str


@enhanced_dataclass
class PerformanceResult:
    """Result of performance testing."""
    success: bool
    n_success: int
    n_failed: int
    log_file: str
    results_dict: Dict[int, tuple] = field(default_factory=dict)


@enhanced_dataclass
class PrecisionResult:
    """Result of precision testing."""
    success: bool
    n_success: int
    n_failed: int
    log_file: str
    results_dict: List[Dict[int, tuple]] = field(default_factory=dict)


class TaskObject:
    """Main task object containing all information for kernel generation and testing."""

    def __init__(self, problem_def_full_path, sample_id, case_id=None, kernel_name=None):
        self.sample_id = sample_id
        self.problem_def_full_path = problem_def_full_path
        self.kernel_name = kernel_name

        # Validate dynamic shape mode requirements
        if config.static_shape_mode:
            if case_id is None:
                raise RuntimeError("static_shape_mode is True, but case_id is None")
            self.case_id = case_id
        else:
            self.case_id = None

        # Initialize basic properties
        self.api_desc_json_path = os.path.join(self.problem_def_full_path, "question/api_desc.md")
        self.op_name = None
        self.op_dir_name = None
        self.category = None
        self.level_id = None
        self.problem_desc = None
        self.log_prefix = ""

        # Parse problem definition
        self._parse_problem_definition()

        # Initialize content and parsers
        self.gen_content = None
        self.tiling_def = None
        self.tiling_content = None
        self.kernel_content = None
        self.kernel_parser = KernelParser(self.kernel_name)

        # Initialize result objects
        self.code_gen_result: CodeGenResult = None
        self.compile_result: CompileResult = None
        self.perf_result: PerformanceResult = None
        self.precision_result: PrecisionResult = None

    def __str__(self):
        """String representation of the task object."""
        info = {
            'problem_def_full_path': self.problem_def_full_path,
            'level_id': self.level_id,
            'category': self.category,
            'op_name': self.op_name,
            'sample_id': self.sample_id,
        }
        if config.static_shape_mode:
            info['case_id'] = self.case_id
        return str(info)

    @property
    def short_id(self) -> str:
        """Short identifier for the task."""
        return f"lvl{self.level_id}_category{self.category}_{self.op_name}_sample{self.sample_id}"

    @property
    def short_id_wo_sample(self) -> str:
        """Short identifier without sample ID."""
        return f"lvl{self.level_id}_category{self.category}_{self.op_name}"

    def _parse_problem_definition(self):
        """Parse problem definition from directory structure."""

        def camel_to_snake(name):
            """Convert CamelCase to snake_case."""
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

        # Extract op name and directory info
        self.op_dir_name = os.path.basename(self.problem_def_full_path)
        self.op_name = self.op_dir_name.split('_')[0]
        self.kernel_name = camel_to_snake(self.op_name)

        # Extract category and level from parent directories
        parent_dir = os.path.dirname(self.problem_def_full_path)
        self.category = os.path.basename(parent_dir)

        # Parse level ID from directory name
        match = re.search(r'level(\d+)', parent_dir, re.IGNORECASE)
        self.level_id = match.group(1) if match else None

        print(self)

    def get_file_path_from_definition(self, file_name):
        """Get file path from problem definition, checking common directory if needed."""
        primary_path = f"{self.problem_def_full_path}/{file_name}"
        if os.path.exists(primary_path):
            return primary_path

        common_path = f"{self.problem_def_full_path}/../common/{file_name}"
        if os.path.exists(common_path):
            return common_path

        raise RuntimeError(f"File not found: {file_name}")

    def get_problem_description(self):
        """Load and return problem description from API description file."""
        if self.problem_desc is None:
            if not os.path.exists(self.api_desc_json_path):
                print(f"[WARNING] API description not found: {self.api_desc_json_path}")
                return ""

            try:
                with open(self.api_desc_json_path, 'r', encoding='utf-8') as file:
                    self.problem_desc = file.read()
            except Exception as e:
                print(f"[ERROR] Failed to read API description: {e}")
                return ""

        return self.problem_desc

    @staticmethod
    def find_custom_lib_path(search_dir):
        """
        Find libcust_opapi.so file in directory and return its prefix path.

        Args:
            search_dir: Directory to search in

        Returns:
            str: Prefix path (XXX in XXX/op_api/lib/libcust_opapi.so)
            None: If file not found
        """
        target_file = "libcust_opapi.so"
        target_pattern = os.path.join("op_api", "lib", target_file)

        for root, dirs, files in os.walk(search_dir):
            if target_file in files:
                full_path = os.path.join(root, target_file)
                if target_pattern in full_path:
                    # Find prefix part before /op_api/lib
                    prefix_end_index = full_path.find(os.path.join("op_api", "lib"))
                    if prefix_end_index > 0:
                        return full_path[:prefix_end_index - 1]  # -1 to remove path separator

        return None

    def check_compile_success(self):
        """Check if compilation was successful by verifying required files exist."""
        # Check for kernel_gen_ops.*.so files
        so_pattern = os.path.join(self.build_path, "kernel_gen_ops.*.so")
        ops_so_exists = len(glob.glob(so_pattern)) > 0

        # Check for custom op API library
        custom_opapi_path = self.find_custom_lib_path(self.src_path)
        custom_opapi_exists = custom_opapi_path is not None

        # Check for generated kernel .o files
        if custom_opapi_exists:
            kernel_pattern = os.path.join(custom_opapi_path, "**", f"{self.op_name}*.o")
            kernel_exists = len(glob.glob(kernel_pattern, recursive=True)) > 0
        else:
            kernel_exists = False

        # All three conditions must be met for success
        overall_success = ops_so_exists and custom_opapi_exists and kernel_exists
        return overall_success, ops_so_exists, custom_opapi_exists, kernel_exists

    def save_code_gen_content(self, data, logger):
        save_json_file(data, self.prompt_save_file_path)
        if logger:
            logger.info(f"Saved code generation prompts and results to: {self.prompt_save_file_path}")

    def load_code_gen_content(self, logger):
        # Load existing generation results
        if not os.path.exists(self.prompt_save_file_path):
            if logger:
                raise RuntimeError(f"No existing results found at {self.prompt_save_file_path}")

        with open(self.prompt_save_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        generated_content = data.get('gen_content', '')
        reasoning_content = data.get('reasoning_content', '')
        final_content = f"{reasoning_content}\n{generated_content}"
        if not generated_content:
            raise RuntimeError("Loaded content is empty")

        self.gen_content = final_content
        return generated_content

    def parse_gen_content(self):
        """Parse generated content into tiling and kernel components."""
        self.tiling_def, self.tiling_content, self.kernel_content = self.kernel_parser.parse(gen_content=self.gen_content)

        if config.kernel_only_mode:
            # Load tiling content from template
            tiling_template_path = f"{self.problem_def_full_path}/question/op_host/{self.kernel_name}.cpp"
            self.tiling_content = safe_read_file(tiling_template_path)

            # Add tiling parameter to kernel function
            self.kernel_content = self.kernel_content.replace(
                'GM_ADDR workspace',
                'GM_ADDR workspace, GM_ADDR tiling'
            )

        return [self.tiling_def, self.tiling_content, self.kernel_content]

    def set_custom_opp_path(self):
        """Set environment variables for custom operator paths."""
        lib_path = self.find_custom_lib_path(self.src_path)
        if lib_path:
            os.environ["ASCEND_CUSTOM_OPP_PATH"] = f"{lib_path}/"
            print(f"Set ASCEND_CUSTOM_OPP_PATH to {os.environ['ASCEND_CUSTOM_OPP_PATH']}")

            if config.compile.mode == "cann-ops":
                os.environ["CUST_OPAPI_LIB_PATH"] = f"{self.src_path}/build/libcust_opapi.so"
                print(f"Set CUST_OPAPI_LIB_PATH to {os.environ['CUST_OPAPI_LIB_PATH']}")

    # Path properties
    @property
    def work_path(self):
        """Base working directory path for this task."""
        if config.static_shape_mode:
            return (f"{config.run_dir}/{config.compile.mode}/lvl{self.level_id}/"
                    f"{self.category}/{self.op_dir_name}/fixed_case_{self.case_id}/sample{self.sample_id}")
        else:
            return (f"{config.run_dir}/{config.compile.mode}/lvl{self.level_id}/"
                    f"{self.category}/{self.op_dir_name}/sample{self.sample_id}")

    @property
    def src_path(self):
        return f"{self.work_path}/{config.src_dir}"

    @property
    def build_path(self):
        return f"{self.work_path}/{config.build_dir}"

    @property
    def log_path(self):
        return f"{self.work_path}/{config.log_dir}"

    @property
    def fake_path(self):
        return f"{self.problem_def_full_path}/answer/{self.sample_id}"

    @property
    def template_path(self):
        return f"{self.problem_def_full_path}/question"

    @property
    def fake_path_wo_sample(self):
        return f"{self.problem_def_full_path}/answer"

    # File path properties
    @property
    def module_path(self):
        return self.get_file_path_from_definition("validation/module.py")

    @property
    def prepare_inputs_path(self):
        return self.get_file_path_from_definition("validation/prepare_inputs.py")

    @property
    def test_cases_path(self):
        return self.get_file_path_from_definition("validation/test_cases.csv")

    @property
    def tiling_cases_path(self):
        return self.get_file_path_from_definition("validation/tiling_cases.csv")

    # Log file path properties
    @property
    def compile_log_path(self):
        return f"{self.log_path}/{self.short_id}_compile{self.log_prefix}.log"

    @property
    def eval_perf_log_path(self):
        return f"{self.log_path}/{self.short_id}_performance{self.log_prefix}.log"

    @property
    def eval_precision_log_path(self):
        return f"{self.log_path}/{self.short_id}_precision{self.log_prefix}.log"

    @property
    def code_gen_log_path(self):
        return f"{self.log_path}/{self.short_id}_code_gen{self.log_prefix}.log"

    @property
    def prompt_save_file_path(self):
        return f"{self.log_path}/{self.short_id}_rag_simple_code_gen_prompt.json"

    @staticmethod
    def remove_cache_dir(rm_path, verbose: bool = False):
        """Remove specified cache directory."""
        if os.path.exists(rm_path):
            try:
                shutil.rmtree(rm_path, ignore_errors=True)
                if verbose:
                    print(f"[INFO] Removed cache directory: {rm_path}")
            except Exception as e:
                print(f"[WARNING] Failed to remove {rm_path}: {e}")

    def remove_cache_dirs(self, rm_paths: list = None, verbose: bool = False):
        """Remove multiple cache directories."""
        for rm_path in rm_paths:
            self.remove_cache_dir(rm_path, verbose)