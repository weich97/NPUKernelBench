import os
from enum import Enum
from typing import List, Optional

import torch
import yaml
import pathlib
from dataclasses import dataclass

from framework.arg_parser import parse_arguments


class TestStage(Enum):
    """Enumeration of testing stages."""
    CODE_GEN = "code_gen_result"
    COMPILE = "compile_result"
    PRECISION = "precision_result"
    PERFORMANCE = "perf_result"
    FINAL = "final_result"


@dataclass
class ChatConfig:
    """Configuration for chat-based code generation."""
    active: bool
    model_path: str
    api_model: str
    api_max_retries: int
    api_retry_delay: int
    num_processes: int
    timeout: int
    temperature: float
    max_tokens: int


@dataclass
class CompileConfig:
    """Configuration for compilation process."""
    mode: str
    common_dir: str
    timeout: int
    num_processes: int


@dataclass
class EvalConfig:
    """Configuration for evaluation and testing."""
    verbose: bool
    num_perf_trials: int
    num_correct_trials: int
    gpu_devices: List[int]
    max_rel_error: float
    max_abs_error: float
    timeout_perf: int
    timeout_precision: int


@dataclass
class FullConfig:
    """Complete configuration object containing all settings."""
    show_result_log_file: bool
    run_dir: str
    src_dir: str
    log_dir: str
    encoding: str
    build_dir: str
    chat: ChatConfig
    compile: CompileConfig
    eval: EvalConfig
    n_sample: int
    n_case: int
    static_shape_mode: bool
    kernel_only_mode: bool
    active_stages: List


class ConfigManager:
    """Singleton configuration manager for loading and accessing configuration."""
    _instance: Optional['ConfigManager'] = None
    _config: Optional[FullConfig] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def load_config(self) -> FullConfig:
        """Load YAML configuration file and convert to Config object."""
        if self._config is not None:
            return self._config

        args = parse_arguments()
        current_file_dir = pathlib.Path(__file__).parent.absolute()
        config_path = f"{current_file_dir}/../{args.config_path}"

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        # Create nested configuration objects
        chat_config = ChatConfig(**config_dict.pop('chat'))
        compile_config = CompileConfig(**config_dict.pop('compile'))
        eval_config = EvalConfig(**config_dict.pop('eval'))

        # Create main configuration object
        map_str2stage = dict(code_gen=TestStage.CODE_GEN,
                             compile=TestStage.COMPILE,
                             precision=TestStage.PRECISION,
                             perf=TestStage.PERFORMANCE)
        active_stages = [map_str2stage[item] for item in args.stages]

        self._config = FullConfig(
            **config_dict,
            chat=chat_config,
            compile=compile_config,
            eval=eval_config,
            active_stages=active_stages
        )
        configure_testing_mode(args, self._config)
        return self._config

    @property
    def config(self) -> FullConfig:
        """Get configuration object (must be loaded first)."""
        if self._config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        return self._config


def configure_testing_mode(args, config_):
    """Configure testing mode based on arguments."""
    if args.run_dir:
        config_.run_dir = args.run_dir

    if args.test_mode:
        config_.static_shape_mode = (args.test_mode == 'static')

    if args.template_mode:
        config_.kernel_only_mode = (args.template_mode == 'kernel_only')

    if args.n_sample:
        config_.n_sample = args.n_sample

    if args.n_case:
        config_.n_case = args.n_case

    if args.chat:
        config_.chat.active = True
        config_.static_shape_mode = True
    else:
        config_.chat.active = False
        config_.n_sample = 1
        if not config_.kernel_only_mode:
            config_.n_case = 1
            config_.compile.mode = "cann-ops"


# Create global configuration manager instance
_config_manager = ConfigManager()

# Auto-load default configuration
config = _config_manager.load_config()
